import os
import sqlite3
import logging
import uvicorn
import tempfile
import networkx as nx
import re
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pydantic import BaseModel
from textblob import TextBlob
import asyncio
import hashlib
from datetime import datetime
from contextlib import asynccontextmanager
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
import aiosqlite
import aiohttp
from functools import lru_cache
from pydub import AudioSegment
from typing import Dict, List, Tuple, Optional
import time
import aiohttp
from dataclasses import dataclass

# Setup logging with UTF-8 encoding
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("chatbot.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if os.name == "nt":
    import sys
    console_handler.stream = sys.stdout
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Timeout for long-running tasks (20 minutes)
TIMEOUT_SECONDS = 20 * 60  # Increased to 20 minutes

# Supported languages by Whisper
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "auto": "Auto-detect"
}

# Currency conversion API (using a free API, replace with a paid one for production)
EXCHANGE_RATE_API = "https://api.exchangerate-api.com/v4/latest/INR"

# Data structure for portfolio tracking
@dataclass
class PortfolioItem:
    asset_type: str  # e.g., stock, mutual fund
    name: str  # e.g., "AAPL", "HDFC Mutual Fund"
    quantity: float
    purchase_price: float
    currency: str

# Global portfolio storage (in-memory for now, can be moved to DB)
portfolio: Dict[str, List[PortfolioItem]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    global db
    db = await aiosqlite.connect("sessions.db")
    # Initialize database
    logger.info("Initializing database...")
    await db.execute("""CREATE TABLE IF NOT EXISTS sessions
                        (session_id TEXT, role TEXT, content TEXT, timestamp TEXT)""")
    await db.commit()
    logger.info("Database initialized successfully")
    yield
    await db.close()
    logger.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    whisper_model = WhisperModel("large-v2", compute_type="int8")
    logger.info("Whisper model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Whisper model: {str(e)}")
    raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}")

# LRU cache for transcriptions (max 100 entries)
@lru_cache(maxsize=100)
def get_cached_transcription(audio_hash: str, language: str) -> Tuple[str, str]:
    return transcription_cache.get((audio_hash, language), (None, None))

transcription_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}

class TextPrompt(BaseModel):
    session_id: str
    prompt: str
    model: str

class ChatRequest(BaseModel):
    session_id: str
    user_input: str
    model: str

async def save_session_message(session_id: str, role: str, content: str):
    logger.info(f"Saving message for session {session_id}: {role}")
    try:
        timestamp = datetime.now().isoformat()
        await db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, role, content, timestamp)
        )
        await db.commit()
        logger.info(f"Message saved successfully for session {session_id}")
    except aiosqlite.Error as e:
        logger.error(f"Error saving message to database: {str(e)}")
        raise

async def get_session_history(session_id: str) -> List[Dict[str, str]]:
    logger.info(f"Fetching history for session {session_id}")
    try:
        async with aiosqlite.connect("sessions.db") as conn:
            cursor = await conn.execute(
                "SELECT role, content, timestamp FROM sessions WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            history = [{"role": row[0], "content": row[1], "timestamp": row[2]} async for row in cursor]
            logger.info(f"Successfully fetched {len(history)} messages for session {session_id}")
            return history
    except aiosqlite.Error as e:
        logger.error(f"Database error while fetching history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# Function to extract financial details from prompt and history
def extract_financial_details(prompt: str, history: List[Dict[str, str]]) -> Dict:
    details = {"income": None, "credit_score": None, "loan_amount": None, "currency": "INR", "savings_goal": None}

    # Extract from prompt
    prompt_lower = prompt.lower()
    income_match = re.search(r"earn\s+(\d+)\s*(?:every month|per month|monthly)", prompt_lower)
    if income_match:
        details["income"] = int(income_match.group(1))
    credit_match = re.search(r"credit score of\s+(\d+)", prompt_lower)
    if credit_match:
        details["credit_score"] = int(credit_match.group(1))
    loan_match = re.search(r"loan of\s+(\d+)\s*(rupees|rs)?", prompt_lower)
    if loan_match:
        details["loan_amount"] = int(loan_match.group(1))
        details["currency"] = "INR"
    savings_match = re.search(r"save\s+(\d+)\s*(?:per month|every month|monthly)", prompt_lower)
    if savings_match:
        details["savings_goal"] = int(savings_match.group(1))
    if "dollar" in prompt_lower:
        details["currency"] = "USD"

    # Extract from history if not found in prompt
    for message in reversed(history):
        msg_lower = message["content"].lower()
        if not details["income"]:
            income_match = re.search(r"earn\s+(\d+)\s*(?:every month|per month|monthly)", msg_lower)
            if income_match:
                details["income"] = int(income_match.group(1))
        if not details["credit_score"]:
            credit_match = re.search(r"credit score of\s+(\d+)", msg_lower)
            if credit_match:
                details["credit_score"] = int(credit_match.group(1))
        if not details["loan_amount"] and "loan" in prompt_lower:
            loan_match = re.search(r"loan of\s+(\d+)\s*(rupees|rs)?", msg_lower)
            if loan_match:
                details["loan_amount"] = int(loan_match.group(1))
                details["currency"] = "INR"
        if not details["savings_goal"] and "save" in prompt_lower:
            savings_match = re.search(r"save\s+(\d+)\s*(?:per month|every month|monthly)", msg_lower)
            if savings_match:
                details["savings_goal"] = int(savings_match.group(1))
        if "dollar" in msg_lower and not details["currency"] == "INR":
            details["currency"] = "USD"

    return details

# Currency conversion function
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> Optional[float]:
    logger.info(f"Converting {amount} from {from_currency} to {to_currency}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(EXCHANGE_RATE_API) as response:
                response.raise_for_status()
                data = await response.json()
                rates = data["rates"]
                if from_currency != "INR":
                    amount_in_inr = amount * rates["INR"] / rates[from_currency]
                else:
                    amount_in_inr = amount
                converted_amount = amount_in_inr * rates[to_currency] / rates["INR"]
                logger.info(f"Converted {amount} {from_currency} to {converted_amount} {to_currency}")
                return converted_amount
    except Exception as e:
        logger.error(f"Error converting currency: {str(e)}")
        return None

# Financial calculators
def calculate_emi(principal: float, annual_rate: float, tenure_years: int) -> float:
    monthly_rate = annual_rate / 12 / 100
    tenure_months = tenure_years * 12
    emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
    logger.info(f"Calculated EMI: {emi} for principal={principal}, rate={annual_rate}%, tenure={tenure_years} years")
    return emi

def calculate_compound_interest(principal: float, annual_rate: float, times_compounded: int, years: int) -> float:
    amount = principal * (1 + annual_rate / (times_compounded * 100)) ** (times_compounded * years)
    interest = amount - principal
    logger.info(f"Calculated compound interest: {interest} (total amount: {amount})")
    return interest, amount

def calculate_retirement_corpus(current_age: int, retirement_age: int, monthly_expense: float, inflation_rate: float, life_expectancy: int) -> float:
    years_to_retirement = retirement_age - current_age
    years_in_retirement = life_expectancy - retirement_age
    future_monthly_expense = monthly_expense * (1 + inflation_rate / 100) ** years_to_retirement
    annual_expense = future_monthly_expense * 12
    corpus_needed = annual_expense * years_in_retirement
    logger.info(f"Calculated retirement corpus: {corpus_needed} for {years_in_retirement} years of retirement")
    return corpus_needed

# Budget planner based on 50/30/20 rule
def plan_budget(monthly_income: float) -> Dict[str, float]:
    budget = {
        "needs": monthly_income * 0.5,
        "wants": monthly_income * 0.3,
        "savings_debt": monthly_income * 0.2
    }
    logger.info(f"Budget plan for income {monthly_income}: {budget}")
    return budget

G = nx.DiGraph()

def financial_retriever(query: str) -> str:
    query = query.lower()
    logger.info(f"Financial retriever query: {query}")
    
    if "loan" in query:
        return ("**Loan Application Process**\n\n"
                "Here’s how to apply for a loan:\n"
                "- **Check Credit Score**: Above 700 is ideal for better terms; 600-700 may result in higher interest rates (10-15% p.a.).\n"
                "- **Documents Needed**: Proof of income, ID, bank statements, employment history.\n"
                "- **Application**: Apply via banks or online platforms.\n"
                "- **Eligibility**: Lenders assess your debt-to-income ratio (DTI); below 40% is preferred.\n"
                "- **Approval**: Based on your financial profile, the lender will approve the loan and set terms.\n\n"
                "**Tips**:\n"
                "- Compare interest rates from multiple lenders.\n"
                "- Ensure you can manage the EMI within your budget.")
    elif "investment" in query or "invest" in query:
        return ("**Investment Options**\n\n"
                "Consider these investment avenues:\n"
                "- **Stocks**: Trade via brokerage platforms; high risk, high reward.\n"
                "- **Mutual Funds**: Invest through SIPs for diversified growth.\n"
                "- **Fixed Deposits**: Stable returns from banks (5-7% p.a.).\n"
                "- **Gold**: A hedge against inflation.\n"
                "- **Real Estate**: Long-term investment with potential appreciation.\n\n"
                "**Tips**:\n"
                "- Diversify to manage risk.\n"
                "- Assess your risk tolerance before investing.")
    elif "budget" in query or "budgeting" in query:
        return ("**Budgeting Tips**\n\n"
                "Follow these strategies:\n"
                "- **50/30/20 Rule**: Allocate 50% to needs, 30% to wants, 20% to savings/debt repayment.\n"
                "- **Track Expenses**: Use apps or spreadsheets to monitor spending.\n"
                "- **Emergency Fund**: Save 3-6 months’ worth of expenses.\n"
                "- **Review Monthly**: Adjust your budget as needed.\n\n"
                "**Tips**:\n"
                "- Automate savings to stay consistent.\n"
                "- Cut unnecessary expenses to meet financial goals.")
    elif "saving" in query or "savings" in query:
        return ("**Savings Strategies**\n\n"
                "Build your savings with these steps:\n"
                "- **Set Goals**: Define short-term (e.g., emergency fund) and long-term (e.g., retirement) goals.\n"
                "- **Automate Transfers**: Schedule monthly transfers to a savings account.\n"
                "- **High-Yield Accounts**: Use accounts with higher interest rates (e.g., 4-5% p.a.).\n"
                "- **Cut Expenses**: Reduce discretionary spending to save more.\n\n"
                "**Tips**:\n"
                "- Start small and increase savings over time.\n"
                "- Keep savings separate from your checking account.")
    elif "tax" in query or "taxes" in query:
        return ("**Tax Planning Basics**\n\n"
                "Manage your taxes effectively:\n"
                "- **Understand Slabs**: Know your tax bracket (e.g., in India, 5-30% based on income).\n"
                "- **Deductions**: Claim deductions (e.g., under Section 80C in India for investments up to ₹1.5 lakh).\n"
                "- **File On Time**: Avoid penalties by filing before deadlines (e.g., July 31 in India).\n"
                "- **Consult Professionals**: For complex cases, hire a tax advisor.\n\n"
                "**Tips**:\n"
                "- Keep records of all income and expenses.\n"
                "- Invest in tax-saving instruments like ELSS or PPF.")
    elif "retirement" in query or "pension" in query:
        return ("**Retirement Planning**\n\n"
                "Prepare for retirement with these steps:\n"
                "- **Estimate Needs**: Calculate future expenses (e.g., 70-80% of current expenses).\n"
                "- **Start Early**: Invest in retirement plans like NPS or EPF.\n"
                "- **Diversify Investments**: Include equities, bonds, and fixed-income assets.\n"
                "- **Review Regularly**: Adjust your plan as your income or goals change.\n\n"
                "**Tips**:\n"
                "- Aim to save 15-20% of your income for retirement.\n"
                "- Consider inflation when planning.")
    return "I couldn’t find specific financial information for this query. Please provide more details or ask about loans, investments, budgeting, savings, taxes, or retirement planning."

# New financial tools
def emi_calculator(query: str) -> str:
    loan_match = re.search(r"loan of\s+(\d+)\s*(rupees|rs|dollars|usd)?", query.lower())
    rate_match = re.search(r"interest rate of\s+(\d+\.?\d*)", query.lower())
    tenure_match = re.search(r"for\s+(\d+)\s*years", query.lower())
    
    if not (loan_match and rate_match and tenure_match):
        return "Please provide loan amount, interest rate, and tenure (e.g., 'loan of 500000 rupees at interest rate of 12 for 5 years')."
    
    principal = float(loan_match.group(1))
    annual_rate = float(rate_match.group(1))
    tenure_years = int(tenure_match.group(1))
    currency = "INR" if loan_match.group(2) in ["rupees", "rs"] else "USD"
    
    emi = calculate_emi(principal, annual_rate, tenure_years)
    return f"**EMI Calculation**\n- Loan Amount: {principal} {currency}\n- Annual Interest Rate: {annual_rate}%\n- Tenure: {tenure_years} years\n- Monthly EMI: {emi:.2f} {currency}"

def compound_interest_calculator(query: str) -> str:
    principal_match = re.search(r"invest\s+(\d+)\s*(rupees|rs|dollars|usd)?", query.lower())
    rate_match = re.search(r"interest rate of\s+(\d+\.?\d*)", query.lower())
    years_match = re.search(r"for\s+(\d+)\s*years", query.lower())
    
    if not (principal_match and rate_match and years_match):
        return "Please provide investment amount, interest rate, and duration (e.g., 'invest 10000 rupees at interest rate of 8 for 5 years')."
    
    principal = float(principal_match.group(1))
    annual_rate = float(rate_match.group(1))
    years = int(years_match.group(1))
    currency = "INR" if principal_match.group(2) in ["rupees", "rs"] else "USD"
    
    interest, total = calculate_compound_interest(principal, annual_rate, 12, years)
    return f"**Compound Interest Calculation**\n- Principal: {principal} {currency}\n- Annual Interest Rate: {annual_rate}%\n- Duration: {years} years\n- Interest Earned: {interest:.2f} {currency}\n- Total Amount: {total:.2f} {currency}"

def retirement_calculator(query: str) -> str:
    age_match = re.search(r"i am\s+(\d+)\s*years old", query.lower())
    retire_match = re.search(r"retire at\s+(\d+)", query.lower())
    expense_match = re.search(r"monthly expense of\s+(\d+)\s*(rupees|rs|dollars|usd)?", query.lower())
    
    if not (age_match and retire_match and expense_match):
        return "Please provide your current age, retirement age, and monthly expenses (e.g., 'I am 30 years old, want to retire at 60, with monthly expense of 50000 rupees')."
    
    current_age = int(age_match.group(1))
    retirement_age = int(retire_match.group(1))
    monthly_expense = float(expense_match.group(1))
    currency = "INR" if expense_match.group(2) in ["rupees", "rs"] else "USD"
    
    corpus = calculate_retirement_corpus(current_age, retirement_age, monthly_expense, inflation_rate=5.0, life_expectancy=85)
    return f"**Retirement Corpus Calculation**\n- Current Age: {current_age}\n- Retirement Age: {retirement_age}\n- Monthly Expense: {monthly_expense} {currency}\n- Corpus Needed (adjusted for 5% inflation): {corpus:.2f} {currency}"

def budget_planner(query: str) -> str:
    income_match = re.search(r"income of\s+(\d+)\s*(rupees|rs|dollars|usd)?", query.lower())
    if not income_match:
        return "Please provide your monthly income (e.g., 'income of 50000 rupees')."
    
    income = float(income_match.group(1))
    currency = "INR" if income_match.group(2) in ["rupees", "rs"] else "USD"
    budget = plan_budget(income)
    return (f"**Budget Plan (50/30/20 Rule)**\n- Monthly Income: {income} {currency}\n"
            f"- Needs (50%): {budget['needs']:.2f} {currency}\n"
            f"- Wants (30%): {budget['wants']:.2f} {currency}\n"
            f"- Savings/Debt Repayment (20%): {budget['savings_debt']:.2f} {currency}")

tools = [
    Tool(
        name="FinancialRetriever",
        func=financial_retriever,
        description="Retrieves general financial information on loans, investments, budgeting, savings, taxes, and retirement planning."
    ),
    Tool(
        name="EMICalculator",
        func=emi_calculator,
        description="Calculates EMI for a loan based on principal, interest rate, and tenure."
    ),
    Tool(
        name="CompoundInterestCalculator",
        func=compound_interest_calculator,
        description="Calculates compound interest for an investment."
    ),
    Tool(
        name="RetirementCalculator",
        func=retirement_calculator,
        description="Calculates the retirement corpus needed based on age, expenses, and inflation."
    ),
    Tool(
        name="BudgetPlanner",
        func=budget_planner,
        description="Creates a budget plan based on the 50/30/20 rule."
    ),
]

try:
    logger.info("Initializing Ollama LLM...")
    # Note: The Ollama class is deprecated. To update, install the langchain-ollama package:
    # pip install -U langchain-ollama
    # Then replace the import and initialization with:
    # from langchain_ollama import OllamaLLM
    # llm = OllamaLLM(model="llama3.2-vision:11b", base_url="http://localhost:11434", verbose=True)
    llm = Ollama(
        model="llama3.2-vision:11b",
        base_url="http://localhost:11434",
        verbose=True,
    )
    logger.info("Ollama LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
    raise RuntimeError(f"Failed to initialize Ollama LLM: {str(e)}")

# Note: ConversationBufferMemory migration is required for LangChain 1.0.0.
# Follow the migration guide: https://python.langchain.com/docs/versions/migrating_memory/
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

try:
    logger.info("Initializing LangChain agent...")
    # Note: initialize_agent is deprecated. Consider migrating to LangGraph:
    # - Install LangGraph: pip install langgraph
    # - Follow the migration guide: https://python.langchain.com/docs/how_to/migrate_agent/
    # - Use LangGraph's Pre-built ReAct agent: https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
    )
    logger.info("LangChain agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangChain agent: {str(e)}")
    raise RuntimeError(f"Failed to initialize LangChain agent: {str(e)}")

# Preprocess audio to improve transcription accuracy
def preprocess_audio(audio_path: str) -> str:
    logger.info(f"Preprocessing audio file: {audio_path}")
    retries = 3
    for attempt in range(retries):
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Validate and convert audio format if needed
            audio = AudioSegment.from_file(audio_path)
            if audio_path.endswith(".mp3"):
                logger.info("Converting MP3 to WAV")
                audio_path = audio_path.replace(".mp3", ".wav")
                audio.export(audio_path, format="wav")
            
            audio = AudioSegment.from_wav(audio_path)
            # Normalize audio to improve quality
            audio = audio.normalize()
            # Export processed audio
            processed_path = audio_path.replace(".wav", "_processed.wav")
            audio.export(processed_path, format="wav")
            logger.info(f"Audio preprocessing completed: {processed_path}")
            # Verify the processed file exists
            if not os.path.exists(processed_path):
                raise FileNotFoundError(f"Processed audio file not created: {processed_path}")
            return processed_path
        except Exception as e:
            logger.error(f"Error preprocessing audio (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                raise HTTPException(status_code=500, detail=f"Error preprocessing audio after {retries} attempts: {str(e)}")
            time.sleep(0.5)  # Wait before retrying
    return audio_path  # Fallback to original path if preprocessing fails

async def postprocess_transcription(transcription: str, language: str) -> str:
    logger.info(f"Postprocessing transcription for language: {language}")
    # Basic cleanup: remove extra spaces, fix capitalization
    transcription = re.sub(r'\s+', ' ', transcription.strip())
    if language in ["en", "es", "fr", "de"]:
        transcription = transcription.capitalize()
    # Language-specific fixes (example for Hindi)
    if language == "hi":
        # Replace common misheard characters (e.g., "ए" and "ऐ" confusion)
        transcription = transcription.replace("ऐ", "ए").replace("ॉ", "ो")
    logger.info(f"Postprocessed transcription: {transcription}")
    return transcription

async def transcribe(audio_file: str, language: str = None) -> Tuple[str, str]:
    logger.info(f"Transcribing audio file: {audio_file}, language: {language}")
    
    # Preprocess audio
    processed_audio = preprocess_audio(audio_file)
    
    # Compute audio hash for caching
    with open(processed_audio, "rb") as f:
        audio_hash = hashlib.md5(f.read()).hexdigest()

    # Check cache (language-specific)
    cached_transcription, cached_language = get_cached_transcription(audio_hash, language or "auto")
    if cached_transcription:
        logger.info("Using cached transcription")
        return cached_transcription, cached_language

    try:
        kwargs = {
            "beam_size": 10,  # Increased for better accuracy
            "vad_filter": True,  # Enable voice activity detection to reduce noise
            "vad_parameters": {"min_silence_duration_ms": 500}
        }
        if language and language != "auto":
            if language not in SUPPORTED_LANGUAGES:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
            kwargs["language"] = language
            logger.info(f"Forcing transcription language to {language}")
        else:
            logger.info("Auto-detecting language")

        segments, info = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: whisper_model.transcribe(processed_audio, **kwargs)
            ),
            timeout=TIMEOUT_SECONDS
        )
        transcription = ''.join(segment.text for segment in segments)
        detected_language = info.language

        # Postprocess transcription
        transcription = await postprocess_transcription(transcription, detected_language)

        # Cache the result
        transcription_cache[(audio_hash, language or "auto")] = (transcription, detected_language)
        logger.info(f"Transcribed audio (lang: {detected_language}): {transcription}")
        return transcription, detected_language
    except asyncio.TimeoutError:
        logger.error("Transcription timed out")
        raise HTTPException(status_code=504, detail="Transcription timed out after 20 minutes")
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")

async def generate_response(session_id: str, prompt: str, model: str, language: str = "en") -> str:
    logger.info(f"Generating response for session {session_id}, prompt: {prompt}, model: {model}, language: {language}")
    
    # Sentiment analysis for tone
    logger.info("Performing sentiment analysis...")
    blob = TextBlob(prompt)
    sentiment = blob.sentiment.polarity
    sentiment_prefix = ""
    if sentiment < -0.1:
        sentiment_prefix = "I understand your concern. Let's address that:\n\n"
    elif sentiment > 0.1:
        sentiment_prefix = "You're optimistic about your plans! Here's my advice:\n\n"
    logger.info(f"Sentiment analysis completed: polarity={sentiment}")

    # Retrieve session history
    messages = await get_session_history(session_id)
    messages.append({"role": "user", "content": prompt})
    await save_session_message(session_id, "user", prompt)

    # Extract financial details dynamically
    financial_details = extract_financial_details(prompt, messages)
    logger.info(f"Extracted financial details: {financial_details}")

    # Handle currency conversion
    if "convert" in prompt.lower() and "to" in prompt.lower():
        amount_match = re.search(r"convert\s+(\d+\.?\d*)\s*(rupees|rs|dollars|usd)?", prompt.lower())
        to_currency_match = re.search(r"to\s+(rupees|rs|dollars|usd)", prompt.lower())
        if amount_match and to_currency_match:
            amount = float(amount_match.group(1))
            from_currency = "INR" if amount_match.group(2) in ["rupees", "rs"] else "USD"
            to_currency = "INR" if to_currency_match.group(1) in ["rupees", "rs"] else "USD"
            converted_amount = await convert_currency(amount, from_currency, to_currency)
            if converted_amount:
                return f"**Currency Conversion**\n{amount} {from_currency} = {converted_amount:.2f} {to_currency}"

    # Handle portfolio tracking
    if "add to portfolio" in prompt.lower():
        asset_match = re.search(r"add\s+(\d+\.?\d*)\s*(stock|mutual fund)\s+([a-zA-Z\s]+)\s+at\s+(\d+\.?\d*)\s*(rupees|rs|dollars|usd)?", prompt.lower())
        if asset_match:
            quantity = float(asset_match.group(1))
            asset_type = asset_match.group(2)
            name = asset_match.group(3).strip()
            purchase_price = float(asset_match.group(4))
            currency = "INR" if asset_match.group(5) in ["rupees", "rs"] else "USD"
            if session_id not in portfolio:
                portfolio[session_id] = []
            portfolio[session_id].append(PortfolioItem(asset_type, name, quantity, purchase_price, currency))
            return f"Added to portfolio: {quantity} {asset_type} of {name} at {purchase_price} {currency}"

    if "show portfolio" in prompt.lower():
        if session_id not in portfolio or not portfolio[session_id]:
            return "Your portfolio is empty."
        result = "**Your Portfolio**\n"
        for item in portfolio[session_id]:
            result += f"- {item.quantity} {item.asset_type} of {item.name} at {item.purchase_price} {item.currency}\n"
        return result

    # Agentic AI: Add reasoning step for Qwen model
    reasoning = ""
    if "llama3.2-vision:11b" not in model:
        currency_symbol = "₹" if financial_details["currency"] == "INR" else "$"
        reasoning = "**Reasoning**:\n"
        reasoning += "The user asked about a financial topic. I will tailor my response based on their financial details:\n"
        if financial_details["income"]:
            reasoning += f"- Monthly Income: {currency_symbol}{financial_details['income']}\n"
        if financial_details["credit_score"]:
            reasoning += f"- Credit Score: {financial_details['credit_score']}\n"
        if financial_details["loan_amount"]:
            reasoning += f"- Requested Loan Amount: {currency_symbol}{financial_details['loan_amount']}\n"
            if financial_details["income"]:
                annual_interest_rate = 0.12
                monthly_rate = annual_interest_rate / 12
                tenure_months = 60  # 5 years
                emi = (financial_details["loan_amount"] * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
                dti = (emi / financial_details["income"]) * 100
                reasoning += f"- Estimated EMI: {currency_symbol}{emi:.2f} (at 12% interest over 5 years)\n"
                reasoning += f"- Debt-to-Income Ratio (DTI): {dti:.2f}% (lenders prefer below 40%)\n"
        if financial_details["savings_goal"]:
            reasoning += f"- Savings Goal: {currency_symbol}{financial_details['savings_goal']} per month\n"
        reasoning += "I’ll now provide a detailed response based on this analysis.\n\n"

    # Prepare payload for Ollama API, include language hint
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "language": language  # Pass language hint to Ollama
        }
    }
    logger.info(f"Sending request to Ollama API: {payload}")

    # Use aiohttp for async HTTP requests with retry logic
    async with aiohttp.ClientSession() as session:
        for attempt in range(3):  # Retry up to 3 times
            try:
                async with session.post(
                    "http://localhost:11434/v1/chat/completions",
                    json=payload,
                    timeout=TIMEOUT_SECONDS
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    reply = result["choices"][0]["message"]["content"]
                    reply = sentiment_prefix + reasoning + reply
                    logger.info(f"Received response from Ollama: {reply}")
                    break
            except aiohttp.ClientError as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # Last attempt
                    logger.error(f"Error connecting to Ollama API after retries: {str(e)}")
                    raise HTTPException(status_code=503, detail="Ollama API unavailable. Please ensure the service is running.")
                await asyncio.sleep(1)  # Wait before retrying
            except asyncio.TimeoutError:
                logger.error(f"Response generation timed out for session {session_id}")
                raise HTTPException(status_code=504, detail="Response generation timed out after 20 minutes")
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    await save_session_message(session_id, "assistant", reply)

    # Async GraphRAG update
    async def update_graph():
        G.add_node(prompt)
        G.add_node(reply)
        if len(messages) >= 4:
            G.add_edge(messages[-4]["content"], prompt)
            G.add_edge(prompt, reply)
        else:
            G.add_edge(prompt, reply)

    asyncio.create_task(update_graph())

    logger.info(f"[{session_id}] Prompt: {prompt}")
    logger.info(f"[{session_id}] Reply: {reply}")

    return reply

@app.post("/text-chat/")
async def text_chat(data: TextPrompt):
    logger.info(f"Received text-chat request: session_id={data.session_id}, prompt={data.prompt}, model={data.model}")
    if not data.prompt.strip():
        logger.warning("Prompt is empty")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if data.model not in ["qwen:0.5b", "llama3.2-vision:11b"]:
        raise HTTPException(status_code=400, detail="Invalid model selected")
    
    start_time = datetime.now()
    response = await generate_response(data.session_id, data.prompt, data.model, language="en")  # Default to English for text chat
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Response generated in {duration} seconds")
    return {"response": response, "duration": duration}

@app.post("/voice-chat/")
async def voice_chat(file: UploadFile, session_id: str = Form(...), language: str = Form(""), model: str = Form(...)):
    if model not in ["qwen:0.5b", "llama3.2-vision:11b"]:
        raise HTTPException(status_code=400, detail="Invalid model selected")

    audio_path = None
    processed_audio_path = None
    try:
        # Save the uploaded audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            audio_path = f.name
            f.write(await file.read())

        logger.info(f"Processing voice input for session {session_id}, file: {audio_path}, model: {model}")

        # Transcribe the audio
        user_input, detected_language = await transcribe(audio_path, language if language else None)
        logger.info(f"Transcribed: {user_input} (lang: {detected_language})")

        if not user_input.strip():
            raise HTTPException(status_code=400, detail="Transcribed text cannot be empty")

        # Generate response
        start_time = datetime.now()
        # Use detected language for response generation
        reply = await generate_response(session_id, user_input, model, language=detected_language if language == "" else language)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"Generated reply in {duration} seconds: {reply}")

        return JSONResponse(
            content={"response": reply, "transcription": user_input, "language": detected_language, "duration": duration}
        )
    except Exception as e:
        logger.error(f"Error in voice-chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing voice input: {str(e)}")
    finally:
        # Clean up temporary files with retry logic
        for file_path in [audio_path, audio_path.replace(".wav", "_processed.wav") if audio_path else None]:
            if file_path and os.path.exists(file_path):
                retries = 3
                for attempt in range(retries):
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted temporary audio file: {file_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1}/{retries} failed to delete temporary audio file {file_path}: {str(e)}")
                        if attempt == retries - 1:
                            logger.error(f"Failed to delete temporary audio file {file_path} after {retries} attempts: {str(e)}")
                        time.sleep(0.5)  # Wait before retrying

@app.get("/session-history/{session_id}")
async def get_history(session_id: str):
    logger.info(f"Received get_history request for session_id={session_id}")
    try:
        history = await asyncio.wait_for(
            get_session_history(session_id),
            timeout=5
        )
        logger.info(f"Returning history: {history}")
        return history
    except asyncio.TimeoutError:
        logger.error(f"Timeout while fetching history for session {session_id}")
        raise HTTPException(status_code=504, detail="Fetching history timed out after 5 seconds")
    except Exception as e:
        logger.error(f"Error fetching history for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.delete("/clear-history/{session_id}")
async def clear_history(session_id: str):
    logger.info(f"Received clear_history request for session_id={session_id}")
    await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    await db.commit()
    memory.clear()
    if session_id in portfolio:
        del portfolio[session_id]
    logger.info(f"Cleared history for session {session_id}")
    return {"status": "History cleared"}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, timeout_keep_alive=TIMEOUT_SECONDS)