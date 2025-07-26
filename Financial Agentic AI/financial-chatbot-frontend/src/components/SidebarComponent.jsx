import React, { useContext, useEffect, useRef } from 'react';
import { ListGroup, Button, Alert } from 'react-bootstrap';
import { ThemeContext } from './ThemeProvider';
import { ChatContext } from './ChatProvider';

const SidebarComponent = () => {
  const { theme, styles } = useContext(ThemeContext);
  const { history, loading, error, clearHistory } = useContext(ChatContext);
  const sidebarContainerRef = useRef(null);
  const historyEndRef = useRef(null);
  const shouldAutoScroll = useRef(true);

  useEffect(() => {
    const sidebarContainer = sidebarContainerRef.current;
    const handleScroll = () => {
      if (sidebarContainer) {
        const { scrollTop, scrollHeight, clientHeight } = sidebarContainer;
        const isNearBottom = scrollTop + clientHeight >= scrollHeight - 100;
        shouldAutoScroll.current = isNearBottom;
      }
    };

    handleScroll();

    sidebarContainer?.addEventListener('scroll', handleScroll);
    return () => sidebarContainer?.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (shouldAutoScroll.current && historyEndRef.current) {
      historyEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history]);

  // Helper function to truncate text at word boundaries
  const truncateText = (text, maxLength) => {
    if (text.length <= maxLength) return text;
    const truncated = text.slice(0, maxLength);
    const lastSpace = truncated.lastIndexOf(' ');
    if (lastSpace > 0) {
      return truncated.slice(0, lastSpace) + '...';
    }
    return truncated + '...';
  };

  // Helper function to detect language (simplified)
  const detectLanguage = (text) => {
    if (/[\u0900-\u097F]/.test(text)) return 'hi'; // Hindi
    if (/[äöüßÄÖÜ]/.test(text)) return 'de'; // German
    return 'en'; // Default to English
  };

  return (
    <div
      style={{
        width: '25%',
        borderRight: `1px solid ${theme === 'light' ? '#dfe4ea' : '#3B486B'}`,
        padding: '15px',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: styles[theme + 'Theme'].sidebarBackground,
        minHeight: 0,
      }}
    >
      <div style={theme === 'light' ? styles.sidebarHeaderLight : styles.sidebarHeaderDark}>
        Chat History
      </div>
      <div
        ref={sidebarContainerRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          marginBottom: '15px',
          minHeight: 0,
          boxSizing: 'border-box',
        }}
      >
        {error && (
          <Alert variant="danger" style={{ marginBottom: '15px', minHeight: '40px' }}>
            {error.includes('history') ? error : 'Error loading history.'}
          </Alert>
        )}
        {!error && history.length === 0 && !loading && (
          <div
            style={{
              color: styles[theme + 'Theme'].textColor,
              textAlign: 'center',
              padding: '15px',
              minHeight: '40px',
            }}
          >
            No chat history available.
          </div>
        )}
        <ListGroup>
          {history.map((msg, index) => {
            const lang = detectLanguage(msg.content);
            const timestamp = msg.timestamp
              ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : 'Unknown time';

            return (
              <ListGroup.Item
                key={index}
                style={{
                  ...styles.sidebarItem,
                  backgroundColor: index % 2 === 0
                    ? (theme === 'light' ? '#F7F9FC' : '#2A3553')
                    : (theme === 'light' ? '#FFFFFF' : '#3B486B'),
                  color: styles[theme + 'Theme'].textColor,
                  marginBottom: '8px',
                  minHeight: '40px',
                  padding: '12px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  border: 'none',
                  borderRadius: '5px',
                  borderLeft: msg.role === 'user'
                    ? `3px solid ${theme === 'light' ? '#1976D2' : '#64B5F6'}`
                    : `3px solid ${theme === 'light' ? '#D81B60' : '#F06292'}`,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = theme === 'light' ? '#E6F0FA' : '#5C6F9A';
                  Object.assign(e.currentTarget.style, {
                    transform: 'scale(1.005)',
                    transition: 'all 0.2s ease',
                  });
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = index % 2 === 0
                    ? (theme === 'light' ? '#F7F9FC' : '#2A3553')
                    : (theme === 'light' ? '#FFFFFF' : '#3B486B');
                  e.currentTarget.style.transform = 'scale(1)';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px' }}>
                  <strong
                    style={{
                      color: msg.role === 'user'
                        ? (theme === 'light' ? '#1976D2' : '#64B5F6')
                        : (theme === 'light' ? '#D81B60' : '#F06292'),
                      fontSize: '14px',
                      marginRight: '8px',
                    }}
                  >
                    {msg.role === 'user' ? 'You' : 'Assistant'}
                  </strong>
                  <span
                    style={{
                      color: styles[theme + 'Theme'].secondaryTextColor,
                      fontSize: '12px',
                      marginRight: '8px',
                    }}
                  >
                    {timestamp}
                  </span>
                  <span
                    style={{
                      color: styles[theme + 'Theme'].secondaryTextColor,
                      fontSize: '10px',
                      backgroundColor: theme === 'light' ? '#E0E0E0' : '#5C6F9A',
                      padding: '2px 5px',
                      borderRadius: '3px',
                    }}
                  >
                    {lang}
                  </span>
                </div>
                <span
                  style={{
                    color: styles[theme + 'Theme'].textColor,
                    fontSize: '13px',
                    lineHeight: '1.4',
                    wordBreak: 'break-word',
                  }}
                >
                  {truncateText(msg.content, 60)}
                </span>
              </ListGroup.Item>
            );
          })}
        </ListGroup>
        <div ref={historyEndRef} style={{ height: '1px' }} />
      </div>
      <Button
        style={styles.clearButton}
        onClick={clearHistory}
        disabled={loading}
        onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.clearButtonHover)}
        onMouseLeave={(e) => Object.assign(e.currentTarget.style, styles.clearButton)}
      >
        Clear History
      </Button>
    </div>
  );
};

export default SidebarComponent;