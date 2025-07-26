import React, { createContext, useState, useEffect } from 'react';
import styles from './styles';



// Create Theme Context
export const ThemeContext = createContext();

const ThemeProvider = ({ children }) => {
  const [theme, setTheme] = useState('light');

  useEffect(() => {
    document.documentElement.setAttribute('data-bs-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, styles}}>
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;