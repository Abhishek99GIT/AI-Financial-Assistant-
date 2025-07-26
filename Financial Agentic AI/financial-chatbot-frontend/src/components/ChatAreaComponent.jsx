import React, { useContext, useEffect, useRef } from 'react';
import { Card, Spinner } from 'react-bootstrap';
import { ThemeContext } from './ThemeProvider.jsx';
import { ChatContext } from './ChatProvider.jsx';
import MessageComponent from './MessageComponent.jsx';
import AlertComponent from './AlertComponent.jsx';

const ChatAreaComponent = ({ getGreetingMessage }) => {
  const { theme, styles } = useContext(ThemeContext);
  const { messages, history, loading, error, responseTime, elapsedTime, messagesEndRef, selectedModel } = useContext(ChatContext);
  const chatContainerRef = useRef(null);
  const shouldAutoScroll = useRef(true);

  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    const handleScroll = () => {
      if (chatContainer) {
        const { scrollTop, scrollHeight, clientHeight } = chatContainer;
        const isNearBottom = scrollTop + clientHeight >= scrollHeight - 100;
        shouldAutoScroll.current = isNearBottom;
      }
    };

    handleScroll();
    chatContainer?.addEventListener('scroll', handleScroll);
    return () => chatContainer?.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (shouldAutoScroll.current && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, messagesEndRef]);

  return (
    <div
      style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
    >
      <div
        ref={chatContainerRef}
        style={{
          flex: 1,
          padding: '25px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          background: styles[theme + 'Theme'].chatAreaBackground,
          minHeight: 0,
          boxSizing: 'border-box',
        }}
      >
        {messages.length === 0 && history.length === 0 && !loading && !error && (
          <Card
            style={{
              ...styles.messageCard,
              backgroundColor: styles[theme + 'Theme'].cardBackground,
              borderLeft: styles.assistantMessageBorder.borderLeft,
              width: '100%',
              maxWidth: '600px',
              marginBottom: '15px',
              minHeight: '60px',
            }}
            onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.messageCardHover)}
            onMouseLeave={(e) => Object.assign(e.currentTarget.style, styles.messageCard)}
          >
            <Card.Body style={{ padding: '20px', color: styles[theme + 'Theme'].textColor }}>
              <strong>Assistant:</strong>{' '}
              <span>{getGreetingMessage()}</span>
            </Card.Body>
          </Card>
        )}
        {messages.map((msg, index) => (
          <MessageComponent
            key={index}
            message={msg}
            style={{
              width: '100%',
              maxWidth: '600px',
              marginBottom: '15px',
              minHeight: '60px',
            }}
          />
        ))}
        {loading && (
          <Card
            style={{
              ...styles.loadingCard,
              width: '100%',
              maxWidth: '600px',
              marginBottom: '15px',
              minHeight: '60px',
            }}
          >
            <Card.Body style={{ padding: '20px' }}>
              <Spinner animation="border" size="sm" style={{ color: '#FF7043' }} />{' '}
              <span>Processing your request... Elapsed time: {elapsedTime} seconds</span>
              {selectedModel === 'llama3.2-vision:11b' && (
                <div style={{ marginTop: '14px', fontStyle: 'italic', color: '#FF7043' }}>
                  Note: The Vision model may take up to 20 minutes to respond.
                </div>
              )}
            </Card.Body>
          </Card>
        )}
        {responseTime && (
          <AlertComponent
            variant="success"
            style={{
              width: '100%',
              maxWidth: '600px',
              marginBottom: '15px',
              minHeight: '60px',
            }}
          >
            Response generated in {responseTime.toFixed(2)} seconds
          </AlertComponent>
        )}
        {error && (
          <AlertComponent
            variant="danger"
            style={{
              width: '100%',
              maxWidth: '600px',
              marginBottom: '15px',
              minHeight: '60px',
            }}
          >
            {error}
          </AlertComponent>
        )}
        <div ref={messagesEndRef} style={{ height: '1px' }} />
      </div>
    </div>
  );
};

export default ChatAreaComponent;