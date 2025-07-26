import React, { useContext } from 'react';
import { Alert } from 'react-bootstrap';
import { ThemeContext } from './ThemeProvider.jsx';

const AlertComponent = ({ variant, children }) => {
  const { theme, styles } = useContext(ThemeContext);

  const alertStyles = {
    success: theme === 'light' ? styles.alertSuccess : styles.alertSuccessDark,
    danger: styles.alertDanger,
  };

  return (
    <Alert variant={variant === 'success' ? 'info' : 'danger'} style={alertStyles[variant]}>
      {children}
    </Alert>
  );
};

export default AlertComponent;