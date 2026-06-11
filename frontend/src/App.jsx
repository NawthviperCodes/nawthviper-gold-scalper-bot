import React, { useState } from 'react';
import LandingPage from './LandingPage';
import { Dashboard, LoginPage } from './DashboardPage';

const AppState = {
  LANDING: 'LANDING',
  LOGIN: 'LOGIN',
  DASHBOARD: 'DASHBOARD',
};

function App() {
  const [appState, setAppState] = useState(AppState.LANDING);
  const [user, setUser] = useState(null);

  const handleAccessDashboard = () => {
    if (user) {
      setAppState(AppState.DASHBOARD);
    } else {
      setAppState(AppState.LOGIN);
    }
  };
  
  const handleLogin = (userData) => {
    setUser(userData);
    setAppState(AppState.DASHBOARD);
  };
  
  const handleLogout = () => {
    setUser(null);
    setAppState(AppState.LANDING);
  };

  // NEW: Handler to go back to the landing page
  const handleBackToLanding = () => {
    setAppState(AppState.LANDING);
  };

  switch (appState) {
    case AppState.LOGIN:
      // Pass the new onBack prop here
      return <LoginPage onLogin={handleLogin} onBack={handleBackToLanding} />;
    
    case AppState.DASHBOARD:
      return <Dashboard user={user} onLogout={handleLogout} />;
      
    case AppState.LANDING:
    default:
      return <LandingPage onAccessDashboard={handleAccessDashboard} />;
  }
}

export default App;