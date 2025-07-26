import React, { createContext, useState, useEffect, useRef, useCallback } from 'react';
import Recorder from 'recorder-js';

export const ChatContext = createContext();

const ChatProvider = ({ children, apiEndpoints, initialModel = 'qwen:0.5b' }) => {
  const [sessionId, setSessionId] = useState('');
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState(initialModel);
  const [selectedLanguage, setSelectedLanguage] = useState(''); // Empty string for auto-detect
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [responseTime, setResponseTime] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const messagesEndRef = useRef(null);
  const timerRef = useRef(null);

  const fetchHistory = useCallback(async (sid) => {
    try {
      const response = await fetch(`${apiEndpoints.history}/${sid}`, { timeout: 10000 });
      if (!response.ok) throw new Error(`Failed to load conversation history: ${response.statusText}`);
      const data = await response.json();
      setHistory(data);
      setMessages(data);
    } catch (err) {
      console.error('Error fetching history:', err);
      setError('Failed to load conversation history. Please try again.');
    }
  }, [apiEndpoints.history]);

  useEffect(() => {
    let storedSessionId = localStorage.getItem('session_id');
    if (!storedSessionId) {
      storedSessionId = 'session-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('session_id', storedSessionId);
    }
    setSessionId(storedSessionId);
    fetchHistory(storedSessionId);
  }, [fetchHistory]);

  useEffect(() => {
    if (loading) {
      timerRef.current = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
      setElapsedTime(0);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  const handleTextChat = async () => {
    if (!input.trim()) {
      setError('Prompt cannot be empty.');
      return;
    }
    const newMessage = { role: 'user', content: input, timestamp: new Date().toISOString() };
    setMessages([...messages, newMessage]);
    setLoading(true);
    setError(null);
    setResponseTime(null);
    try {
      const controller = new AbortController();
      const timeout = 1000000; // 1000 seconds for all models
      const timeoutId = setTimeout(() => {
        controller.abort();
        setError('Request timed out. The server is taking too long to respond.');
      }, timeout);

      const response = await fetch(apiEndpoints.textChat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, prompt: input, model: selectedModel }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to send prompt: ${errorText}`);
      }
      const data = await response.json();
      let assistantResponse = data.response;


      setMessages([...messages, newMessage, { role: 'assistant', content: assistantResponse, timestamp: new Date().toISOString() }]);
      setResponseTime(data.duration);
      setInput('');
      await fetchHistory(sessionId);
    } catch (err) {
      console.error('Error sending text message:', err);
      setError(err.message || 'Failed to get response from the server. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleVoiceRecording = async () => {
    if (!isRecording) {
      try {
        setError(null);
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const recorder = new Recorder(audioContext);
        recorderRef.current = recorder;
        await recorder.init(stream);
        await recorder.start();
        setIsRecording(true);
      } catch (err) {
        console.error('Error starting recording:', err);
        setError('Failed to start recording. Please ensure microphone access is granted.');
      }
    } else {
      try {
        const { blob } = await recorderRef.current.stop();
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setIsRecording(false);
        if (blob.type !== 'audio/wav') {
          throw new Error('Recorded audio is not in WAV format');
        }
        await handleVoiceChat(blob);
      } catch (err) {
        console.error('Error stopping recording:', err);
        setError('Failed to process recording: ' + err.message);
        setIsRecording(false);
      }
    }
  };

  const handleVoiceChat = async (audioBlob) => {
    setLoading(true);
    setError(null);
    setResponseTime(null);
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.wav');
      formData.append('session_id', sessionId);
      formData.append('language', selectedLanguage || ''); // Empty string for auto-detect
      formData.append('model', selectedModel);

      const controller = new AbortController();
      const timeout = 6000000; // 60 seconds timeout
      const timeoutId = setTimeout(() => {
        controller.abort();
        setError('Request timed out. The server is taking too long to respond.');
      }, timeout);

      const response = await fetch(apiEndpoints.voiceChat, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      if (!response.ok) {
        const errorText = await response.text();
        if (errorText.includes('Error preprocessing audio')) {
          throw new Error('Failed to process audio. Please try recording again.');
        }
        throw new Error(`Failed to send voice prompt: ${errorText}`);
      }
      const data = await response.json();
      const newMessage = { role: 'user', content: data.transcription, timestamp: new Date().toISOString() };
      let assistantResponse = data.response;


      setMessages([...messages, newMessage, { role: 'assistant', content: assistantResponse, timestamp: new Date().toISOString() }]);
      setResponseTime(data.duration);
      await fetchHistory(sessionId);
    } catch (err) {
      console.error('Error sending voice message:', err);
      setError(err.message || 'Failed to process voice input. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiEndpoints.clearHistory}/${sessionId}`, {
        method: 'DELETE',
        timeout: 10000,
      });
      if (!response.ok) throw new Error(`Failed to clear history: ${response.statusText}`);
      localStorage.removeItem('session_id');
      const newSessionId = 'session-' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('session_id', newSessionId);
      setSessionId(newSessionId);
      setMessages([]);
      setHistory([]);
    } catch (err) {
      console.error('Error clearing history:', err);
      setError('Failed to clear conversation history. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ChatContext.Provider
      value={{
        sessionId,
        input,
        setInput,
        selectedModel,
        setSelectedModel,
        selectedLanguage,
        setSelectedLanguage,
        messages,
        history,
        loading,
        error,
        setError,
        isRecording,
        responseTime,
        elapsedTime,
        messagesEndRef,
        handleTextChat,
        handleVoiceRecording,
        clearHistory,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
};

export default ChatProvider;