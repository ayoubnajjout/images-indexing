import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const location = useLocation();
  const [apiStatus, setApiStatus] = useState('online');
  
  const navItems = [
    { path: '/', label: 'Collection' },
    { path: '/create', label: 'Create' },
    { path: '/descriptors', label: 'Descriptors' },
    { path: '/search', label: 'Search' },
  ];
  
  useEffect(() => {
    // Check API status (mock for now)
    const checkApiStatus = async () => {
      // In production, this would ping the actual API
      setApiStatus('online');
    };
    
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  const isActive = (path) => {
    return location.pathname === path;
  };
  
  return (
    <nav className="sticky top-0 z-50 bg-white border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <span className="text-xl font-semibold text-slate-900">ImageExplorer</span>
            </Link>
          </div>
          
          {/* Navigation Links */}
          <div className="flex items-center space-x-8 h-full">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center h-full text-sm font-medium transition ${
                  isActive(item.path)
                    ? 'text-indigo-600 border-b-2 border-indigo-600'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
          
          {/* API Status */}
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${
              apiStatus === 'online' ? 'bg-green-500' : 'bg-red-500'
            }`} />
            <span className="text-sm text-slate-600">
              API {apiStatus === 'online' ? 'Online' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
