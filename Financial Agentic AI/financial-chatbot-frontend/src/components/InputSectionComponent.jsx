import React, { useContext } from 'react';
import { Form, Button } from 'react-bootstrap';
import { Mic, Send } from 'react-bootstrap-icons';
import { ThemeContext } from './ThemeProvider';
import { ChatContext } from './ChatProvider';

const InputSectionComponent = ({ placeholder = 'Ask your question...' }) => {
  const { theme, styles } = useContext(ThemeContext);
  const {
    input,
    setInput,
    selectedModel,
    setSelectedModel,
    selectedLanguage,
    setSelectedLanguage,
    loading,
    isRecording,
    handleTextChat,
    handleVoiceRecording,
  } = useContext(ChatContext);

  return (
    <div
      style={{
        ...styles.inputSection,
        ...(theme === 'dark' && styles.inputSectionDark),
        position: 'sticky',
        bottom: 0,
        zIndex: 10,
        backgroundColor: styles[theme + 'Theme'].backgroundColor,
        padding: '10px',
        boxSizing: 'border-box',
      }}
    >
      <Form.Select
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        style={{
          ...styles.select,
          ...(theme === 'dark' && styles.selectDark),
        }}
        disabled={loading}
        onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.selectHover)}
        onMouseLeave={(e) => Object.assign(e.currentTarget.style, { borderColor: theme === 'dark' ? '#4DB6AC' : '#26A69A' })}
      >
        <option value="qwen:0.5b">Qwen Model</option>
        <option value="llama3.2-vision:11b">Vision Model</option>
      </Form.Select>
      <Form.Select
        value={selectedLanguage}
        onChange={(e) => setSelectedLanguage(e.target.value)}
        style={{
          ...styles.select,
          ...(theme === 'dark' && styles.selectDark),
        }}
        disabled={loading}
        onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.selectHover)}
        onMouseLeave={(e) => Object.assign(e.currentTarget.style, { borderColor: theme === 'dark' ? '#4DB6AC' : '#26A69A' })}
      >
        <option value="">Auto-detect Language</option>
        <option value="en">English</option>
        <option value="es">Spanish</option>
        <option value="fr">French</option>
        <option value="de">German</option>
        <option value="hi">Hindi</option>
        <option value="zh">Chinese</option>
        <option value="ja">Japanese</option>
      </Form.Select>
      <Form.Control
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder={placeholder}
        disabled={loading}
        onKeyPress={(e) => e.key === 'Enter' && handleTextChat()}
        style={{
          ...styles.input,
          ...(theme === 'dark' && styles.inputDark),
        }}
        onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.inputHover)}
        onMouseLeave={(e) => Object.assign(e.currentTarget.style, { borderColor: theme === 'dark' ? '#4DB6AC' : '#26A69A' })}
      />
      <Button
        style={{
          ...styles.button,
          ...(isRecording && styles.buttonRecording),
        }}
        onClick={handleVoiceRecording}
        disabled={loading}
        onMouseEnter={(e) => Object.assign(e.currentTarget.style, isRecording ? styles.buttonRecordingHover : styles.buttonHover)}
        onMouseLeave={(e) => Object.assign(e.currentTarget.style, isRecording ? styles.buttonRecording : styles.button)}
      >
        <Mic size={24} />
      </Button>
      <Button
        style={styles.button}
        onClick={handleTextChat}
        disabled={loading}
        onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.buttonHover)}
        onMouseLeave={(e) => Object.assign(e.currentTarget.style, styles.button)}
      >
        <Send size={24} />
      </Button>
    </div>
  );
};

export default InputSectionComponent;