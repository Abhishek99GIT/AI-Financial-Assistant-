import React, { useContext } from 'react';
import { Card } from 'react-bootstrap';
import { ThemeContext } from './ThemeProvider.jsx';
import AlertComponent from './AlertComponent.jsx';

const MessageComponent = ({ message }) => {
  const { theme, styles } = useContext(ThemeContext);

  const parseResponse = (content) => {
    if (!content || typeof content !== 'string') {
      return <p style={{ color: styles[theme + 'Theme'].textColor }}>No response content available.</p>;
    }

    const sections = content.split('\n\n').filter(section => section);
    return sections.map((section, index) => {
      if (section.startsWith('**') && section.endsWith('**') && !section.includes(':')) {
        const title = section.replace(/\*\*/g, '');
        return (
          <h5 key={index} style={{ marginTop: '15px', fontWeight: 'bold', color: styles[theme + 'Theme'].textColor }}>
            {title}
          </h5>
        );
      } else if (section.startsWith('- **')) {
        const items = section.split('\n').map((item, i) => {
          const text = item.replace('- **', '').replace('**:', ':');
          const [label, ...description] = text.split(':');
          const descriptionText = description.length > 0 ? description.join(':').trim() : '';

          if (!descriptionText) {
            return (
              <li key={i} style={{ marginBottom: '5px', color: styles[theme + 'Theme'].textColor }}>
                <strong>{label || 'Item'}</strong>
              </li>
            );
          }

          let formattedDescription = descriptionText;
          const boldRegex = /\*\*(.*?)\*\*/g;
          const boldParts = [];
          let lastIndex = 0;
          let boldMatch;
          while ((boldMatch = boldRegex.exec(descriptionText)) !== null) {
            const before = descriptionText.slice(lastIndex, boldMatch.index);
            if (before) boldParts.push(before);
            boldParts.push(<strong key={`bold-${i}-${boldMatch.index}`}>{boldMatch[1]}</strong>);
            lastIndex = boldMatch.index + boldMatch[0].length;
          }
          if (lastIndex < descriptionText.length) {
            boldParts.push(descriptionText.slice(lastIndex));
          }
          formattedDescription = boldParts.length > 0 ? boldParts : descriptionText;

          if (typeof formattedDescription === 'string') {
            const italicRegex = /\*(.*?)\*/g;
            const italicParts = [];
            lastIndex = 0;
            let italicMatch;
            while ((italicMatch = italicRegex.exec(formattedDescription)) !== null) {
              const before = formattedDescription.slice(lastIndex, italicMatch.index);
              if (before) italicParts.push(before);
              italicParts.push(<em key={`italic-${i}-${italicMatch.index}`}>{italicMatch[1]}</em>);
              lastIndex = italicMatch.index + italicMatch[0].length;
            }
            if (lastIndex < formattedDescription.length) {
              italicParts.push(formattedDescription.slice(lastIndex));
            }
            formattedDescription = italicParts.length > 0 ? italicParts : formattedDescription;
          }

          return (
            <li key={i} style={{ marginBottom: '5px', color: styles[theme + 'Theme'].textColor }}>
              <strong>{label || 'Item'}:</strong>{' '}
              {Array.isArray(formattedDescription) ? (
                formattedDescription
              ) : (
                <span>{formattedDescription || 'No description available.'}</span>
              )}
            </li>
          );
        });
        return <ul key={index} style={{ paddingLeft: '20px' }}>{items}</ul>;
      } else if (/^\d+\.\s/.test(section)) {
        const items = section.split('\n').map((item, i) => {
          const text = item.replace(/^\d+\.\s/, '');

          if (text.startsWith('**')) {
            const cleanedText = text.replace('**:', ':');
            const [label, ...description] = cleanedText.split(':');
            const labelText = label.replace(/\*\*/g, '').trim();
            const descriptionText = description.length > 0 ? description.join(':').trim() : '';

            if (!descriptionText) {
              return (
                <li key={i} style={{ marginBottom: '5px', color: styles[theme + 'Theme'].textColor }}>
                  <strong>{labelText || 'Item'}</strong>
                </li>
              );
            }

            let formattedDescription = descriptionText;
            const boldRegex = /\*\*(.*?)\*\*/g;
            const boldParts = [];
            let lastIndex = 0;
            let boldMatch;
            while ((boldMatch = boldRegex.exec(descriptionText)) !== null) {
              const before = descriptionText.slice(lastIndex, boldMatch.index);
              if (before) boldParts.push(before);
              boldParts.push(<strong key={`bold-num-${i}-${boldMatch.index}`}>{boldMatch[1]}</strong>);
              lastIndex = boldMatch.index + boldMatch[0].length;
            }
            if (lastIndex < descriptionText.length) {
              boldParts.push(descriptionText.slice(lastIndex));
            }
            formattedDescription = boldParts.length > 0 ? boldParts : descriptionText;

            if (typeof formattedDescription === 'string') {
              const italicRegex = /\*(.*?)\*/g;
              const italicParts = [];
              lastIndex = 0;
              let italicMatch;
              while ((italicMatch = italicRegex.exec(formattedDescription)) !== null) {
                const before = formattedDescription.slice(lastIndex, italicMatch.index);
                if (before) italicParts.push(before);
                italicParts.push(<em key={`italic-num-${i}-${italicMatch.index}`}>{italicMatch[1]}</em>);
                lastIndex = italicMatch.index + italicMatch[0].length;
              }
              if (lastIndex < formattedDescription.length) {
                italicParts.push(formattedDescription.slice(lastIndex));
              }
              formattedDescription = italicParts.length > 0 ? italicParts : formattedDescription;
            }

            return (
              <li key={i} style={{ marginBottom: '5px', color: styles[theme + 'Theme'].textColor }}>
                <strong>{labelText || 'Item'}:</strong>{' '}
                {Array.isArray(formattedDescription) ? (
                  formattedDescription
                ) : (
                  <span>{formattedDescription || 'No description available.'}</span>
                )}
              </li>
            );
          } else {
            let formattedText = text.trim();
            const boldRegex = /\*\*(.*?)\*\*/g;
            const boldParts = [];
            let lastIndex = 0;
            let boldMatch;
            while ((boldMatch = boldRegex.exec(formattedText)) !== null) {
              const before = formattedText.slice(lastIndex, boldMatch.index);
              if (before) boldParts.push(before);
              boldParts.push(<strong key={`bold-num-simple-${i}-${boldMatch.index}`}>{boldMatch[1]}</strong>);
              lastIndex = boldMatch.index + boldMatch[0].length;
            }
            if (lastIndex < formattedText.length) {
              boldParts.push(formattedText.slice(lastIndex));
            }
            formattedText = boldParts.length > 0 ? boldParts : formattedText;

            if (typeof formattedText === 'string') {
              const italicRegex = /\*(.*?)\*/g;
              const italicParts = [];
              lastIndex = 0;
              let italicMatch;
              while ((italicMatch = italicRegex.exec(formattedText)) !== null) {
                const before = formattedText.slice(lastIndex, italicMatch.index);
                if (before) italicParts.push(before);
                italicParts.push(<em key={`italic-num-simple-${i}-${italicMatch.index}`}>{italicMatch[1]}</em>);
                lastIndex = italicMatch.index + italicMatch[0].length;
              }
              if (lastIndex < formattedText.length) {
                italicParts.push(formattedText.slice(lastIndex));
              }
              formattedText = italicParts.length > 0 ? italicParts : formattedText;
            }

            return (
              <li key={i} style={{ marginBottom: '5px', color: styles[theme + 'Theme'].textColor }}>
                {Array.isArray(formattedText) ? formattedText : formattedText || 'No description available.'}
              </li>
            );
          }
        });
        return <ol key={index} style={{ paddingLeft: '20px' }}>{items}</ol>;
      } else if (section.startsWith('**Reasoning**:')) {
        const reasoningContent = section.split('\n').map((line, i) => {
          if (line.startsWith('- ')) {
            let bulletText = line.replace('- ', '• ').trim();
            let formattedBullet = bulletText;
            const boldRegex = /\*\*(.*?)\*\*/g;
            const boldParts = [];
            let lastIndex = 0;
            let boldMatch;
            while ((boldMatch = boldRegex.exec(bulletText)) !== null) {
              const before = bulletText.slice(lastIndex, boldMatch.index);
              if (before) boldParts.push(before);
              boldParts.push(<strong key={`bold-reason-${i}-${boldMatch.index}`}>{boldMatch[1]}</strong>);
              lastIndex = boldMatch.index + boldMatch[0].length;
            }
            if (lastIndex < bulletText.length) {
              boldParts.push(bulletText.slice(lastIndex));
            }
            formattedBullet = boldParts.length > 0 ? boldParts : bulletText;

            if (typeof formattedBullet === 'string') {
              const italicRegex = /\*(.*?)\*/g;
              const italicParts = [];
              lastIndex = 0;
              let italicMatch;
              while ((italicMatch = italicRegex.exec(formattedBullet)) !== null) {
                const before = formattedBullet.slice(lastIndex, italicMatch.index);
                if (before) italicParts.push(before);
                italicParts.push(<em key={`italic-reason-${i}-${italicMatch.index}`}>{italicMatch[1]}</em>);
                lastIndex = italicMatch.index + italicMatch[0].length;
              }
              if (lastIndex < formattedBullet.length) {
                italicParts.push(formattedBullet.slice(lastIndex));
              }
              formattedBullet = italicParts.length > 0 ? italicParts : formattedBullet;
            }

            return (
              <div key={i} style={{ marginLeft: '15px', color: theme === 'light' ? '#26A69A' : '#E0E0E0' }}>
                {Array.isArray(formattedBullet) ? formattedBullet : formattedBullet || 'No content available.'}
              </div>
            );
          }

          let formattedLine = line;
          const boldRegex = /\*\*(.*?)\*\*/g;
          const boldParts = [];
          let lastIndexLine = 0;
          let boldMatchLine;
          while ((boldMatchLine = boldRegex.exec(line)) !== null) {
            const before = line.slice(lastIndexLine, boldMatchLine.index);
            if (before) boldParts.push(before);
            boldParts.push(<strong key={`bold-line-${i}-${boldMatchLine.index}`}>{boldMatchLine[1]}</strong>);
            lastIndexLine = boldMatchLine.index + boldMatchLine[0].length;
          }
          if (lastIndexLine < line.length) {
            boldParts.push(line.slice(lastIndexLine));
          }
          formattedLine = boldParts.length > 0 ? boldParts : line;

          if (typeof formattedLine === 'string') {
            const italicRegex = /\*(.*?)\*/g;
            const italicParts = [];
            lastIndexLine = 0;
            let italicMatchLine;
            while ((italicMatchLine = italicRegex.exec(formattedLine)) !== null) {
              const before = formattedLine.slice(lastIndexLine, italicMatchLine.index);
              if (before) italicParts.push(before);
              italicParts.push(<em key={`italic-line-${i}-${italicMatchLine.index}`}>{italicMatchLine[1]}</em>);
              lastIndexLine = italicMatchLine.index + italicMatchLine[0].length;
            }
            if (lastIndexLine < formattedLine.length) {
              italicParts.push(formattedLine.slice(lastIndexLine));
            }
            formattedLine = italicParts.length > 0 ? italicParts : formattedLine;
          }

          return (
            <div key={i} style={{ color: theme === 'light' ? '#26A69A' : '#E0E0E0' }}>
              {Array.isArray(formattedLine) ? formattedLine : formattedLine || 'No content available.'}
            </div>
          );
        });
        return <AlertComponent key={index} variant="success">{reasoningContent}</AlertComponent>;
      } else {
        let formattedParagraph = section;
        const boldRegex = /\*\*(.*?)\*\*/g;
        const boldParts = [];
        let lastIndex = 0;
        let boldMatch;
        while ((boldMatch = boldRegex.exec(section)) !== null) {
          const before = section.slice(lastIndex, boldMatch.index);
          if (before) boldParts.push(before);
          boldParts.push(<strong key={`bold-para-${index}-${boldMatch.index}`}>{boldMatch[1]}</strong>);
          lastIndex = boldMatch.index + boldMatch[0].length;
        }
        if (lastIndex < section.length) {
          boldParts.push(section.slice(lastIndex));
        }
        formattedParagraph = boldParts.length > 0 ? boldParts : section;

        if (typeof formattedParagraph === 'string') {
          const italicRegex = /\*(.*?)\*/g;
          const italicParts = [];
          lastIndex = 0;
          let italicMatch;
          while ((italicMatch = italicRegex.exec(formattedParagraph)) !== null) {
            const before = formattedParagraph.slice(lastIndex, italicMatch.index);
            if (before) italicParts.push(before);
            italicParts.push(<em key={`italic-para-${index}-${italicMatch.index}`}>{italicMatch[1]}</em>);
            lastIndex = italicMatch.index + italicMatch[0].length;
          }
          if (lastIndex < formattedParagraph.length) {
            italicParts.push(formattedParagraph.slice(lastIndex));
          }
          formattedParagraph = italicParts.length > 0 ? italicParts : formattedParagraph;
        }

        return (
          <p key={index} style={{ lineHeight: '1.6', color: styles[theme + 'Theme'].textColor }}>
            {Array.isArray(formattedParagraph) ? formattedParagraph : formattedParagraph || 'No paragraph content available.'}
          </p>
        );
      }
    });
  };

  return (
    <Card
      style={{
        ...styles.messageCard,
        backgroundColor: styles[theme + 'Theme'].cardBackground,
        borderLeft: message.role === 'user' ? styles.userMessageBorder.borderLeft : styles.assistantMessageBorder.borderLeft,
      }}
      onMouseEnter={(e) => Object.assign(e.currentTarget.style, styles.messageCardHover)}
      onMouseLeave={(e) => Object.assign(e.currentTarget.style, styles.messageCard)}
    >
      <Card.Body style={{ padding: '20px', color: styles[theme + 'Theme'].textColor }}>
        <strong style={{ color: message.role === 'user' ? '#42A5F5' : '#26A69A' }}>
          {message.role === 'user' ? 'You' : 'Assistant'}:
        </strong>{' '}
        {message.role === 'assistant' ? parseResponse(message.content) : (
          <span>{message.content || 'No content available.'}</span>
        )}
      </Card.Body>
    </Card>
  );
};

export default MessageComponent;