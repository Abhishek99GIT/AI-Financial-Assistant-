import React, { useContext, useRef, useEffect } from 'react';
import ThemeProvider, { ThemeContext } from './ThemeProvider';
import ChatProvider from './ChatProvider';
import NavbarComponent from './NavbarComponent';
import SidebarComponent from './SidebarComponent';
import ChatAreaComponent from './ChatAreaComponent';
import InputSectionComponent from './InputSectionComponent';

// API Endpoints Configuration
const apiEndpoints = {
  history: 'http://localhost:8000/session-history',
  textChat: 'http://localhost:8000/text-chat/',
  voiceChat: 'http://localhost:8000/voice-chat/',
  clearHistory: 'http://localhost:8000/clear-history',
};

// Add keyframe animation for fade-in effect
const keyframes = `
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;

// Inject keyframes into the document
const styleSheet = document.createElement('style');
styleSheet.innerText = keyframes;
document.head.appendChild(styleSheet);

const Chatbot = () => {
  const getGreetingMessage = () => {
    const now = new Date();
    const istOffset = 5.5 * 60 * 60 * 1000; // IST offset in milliseconds
    const istTime = new Date(now.getTime() + istOffset);
    const hours = istTime.getUTCHours();
    const minutes = istTime.getUTCMinutes();
    const totalHours = hours + minutes / 60;

    let greeting;
    if (totalHours >= 0 && totalHours < 12) {
      greeting = "Good Morning!";
    } else if (totalHours >= 12 && totalHours < 17) {
      greeting = "Good Afternoon!";
    } else if (totalHours >= 17 && totalHours < 21) {
      greeting = "Good Evening!";
    } else {
      greeting = "Good Night!";
    }

    return `${greeting} How can I help you today?`;
  };

  return (
    <ThemeProvider>
      <ChatProvider apiEndpoints={apiEndpoints}>
        <ChatbotContent getGreetingMessage={getGreetingMessage} />
      </ChatProvider>
    </ThemeProvider>
  );
};

const ChatbotContent = ({ getGreetingMessage }) => {
  const { styles, theme } = useContext(ThemeContext);
  const navbarRef = useRef(null);
  const [navbarHeight, setNavbarHeight] = React.useState(0);

  useEffect(() => {
    if (navbarRef.current) {
      const height = navbarRef.current.getBoundingClientRect().height;
      setNavbarHeight(height);
    }
  }, []);

  return (
    <div
      style={{
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: styles[theme + 'Theme'].backgroundColor,
        fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        overflow: 'hidden',
      }}
    >
      <div ref={navbarRef}>
        <NavbarComponent title="Financial Assistant" />
      </div>
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'row',
          height: `calc(100vh - ${navbarHeight}px)`,
          minHeight: 0,
        }}
      >
        <SidebarComponent />
        <div
          style={{
            width: '75%',
            display: 'flex',
            flexDirection: 'column',
            flex: 1,
            minHeight: 0,
          }}
        >
          <ChatAreaComponent getGreetingMessage={getGreetingMessage} />
          <InputSectionComponent placeholder="Ask your financial question..." />
        </div>
      </div>
    </div>
  );
};

export default Chatbot;