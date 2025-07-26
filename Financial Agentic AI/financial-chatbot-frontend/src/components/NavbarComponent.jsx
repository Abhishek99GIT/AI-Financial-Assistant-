import React, { useContext } from 'react';
import { Navbar, Container, Form } from 'react-bootstrap';
import { Sun, Moon } from 'react-bootstrap-icons';
import { ThemeContext } from './ThemeProvider';

const NavbarComponent = ({ title }) => {
  const { theme, toggleTheme, styles } = useContext(ThemeContext);

  return (
    <Navbar expand="lg" style={theme === 'light' ? styles.navbarLight : styles.navbarDark}>
      <Container fluid>
        <Navbar.Brand style={styles.navbarBrand}>{title}</Navbar.Brand>
        <Form className="ms-auto d-flex align-items-center">
          <Form.Check
            type="switch"
            id="theme-switch"
            label={
              <span style={styles.themeSwitch}>
                {theme === 'light' ? (
                  <Sun size={20} style={{ ...styles.themeSwitchIcon, transform: theme === 'light' ? 'rotate(0deg)' : 'rotate(180deg)' }} />
                ) : (
                  <Moon size={20} style={{ ...styles.themeSwitchIcon, transform: theme === 'dark' ? 'rotate(0deg)' : 'rotate(180deg)' }} />
                )}
              </span>
            }
            checked={theme === 'dark'}
            onChange={toggleTheme}
            className="d-flex align-items-center gap-2"
          />
        </Form>
      </Container>
    </Navbar>
  );
};

export default NavbarComponent;