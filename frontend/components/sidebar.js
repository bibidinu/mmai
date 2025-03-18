import React, { useState } from 'react';
import Link from 'next/link';
import { 
  HomeIcon, 
  ChartBarIcon, 
  CurrencyDollarIcon, 
  CogIcon, 
  BellIcon, 
  ArrowPathIcon, 
  MoonIcon, 
  SunIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const Sidebar = ({ activeEnvironment, onEnvironmentChange }) => {
  const [darkMode, setDarkMode] = useState(true);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    
    // Toggle dark mode class on document
    if (newMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const toggleMobileSidebar = () => {
    setIsMobileSidebarOpen(!isMobileSidebarOpen);
  };

  const closeMobileSidebar = () => {
    setIsMobileSidebarOpen(false);
  };

  return (
    <>
      {/* Mobile menu button */}
      <div className="fixed top-0 left-0 z-40 m-4 md:hidden">
        <button
          type="button"
          className="flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
          onClick={toggleMobileSidebar}
        >
          <span className="sr-only">Open sidebar</span>
          <Bars3Icon className="h-6 w-6" aria-hidden="true" />
        </button>
      </div>

      {/* Mobile sidebar */}
      <div className={`fixed inset-0 flex z-40 md:hidden transform ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out`}>
        <div className="fixed inset-0 bg-gray-600 bg-opacity-75" onClick={closeMobileSidebar} aria-hidden="true"></div>
        
        <div className="relative flex-1 flex flex-col max-w-xs w-full bg-gray-800">
          <div className="absolute top-0 right-0 -mr-12 pt-2">
            <button
              type="button"
              className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
              onClick={closeMobileSidebar}
            >
              <span className="sr-only">Close sidebar</span>
              <XMarkIcon className="h-6 w-6 text-white" aria-hidden="true" />
            </button>
          </div>
          
          {/* Mobile sidebar content */}
          <div className="flex-1 h-0 pt-5 pb-4 overflow-y-auto">
            <div className="flex items-center flex-shrink-0 px-4">
              <span className="text-xl font-bold text-white">Crypto Bot</span>
            </div>
            <nav className="mt-5 px-2 space-y-1">
              {renderNavItems()}
            </nav>
          </div>
          
          {/* Mobile bottom section */}
          <div className="flex-shrink-0 flex border-t border-gray-700 p-4">
            {renderEnvironmentSwitch()}
          </div>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden md:flex h-screen w-64 flex-col fixed inset-y-0 left-0 bg-gray-800">
        <div className="flex flex-col h-full justify-between">
          <div>
            {/* Logo */}
            <div className="h-16 flex items-center px-4 bg-gray-900">
              <span className="text-xl font-bold text-white">Crypto Bot</span>
            </div>
            
            {/* Navigation */}
            <nav className="mt-8 flex-1 px-4 space-y-1">
              {renderNavItems()}
            </nav>
          </div>
          
          {/* Environment Switch & Settings */}
          <div className="p-4 border-t border-gray-700">
            {renderEnvironmentSwitch()}
            
            {/* Dark Mode Toggle */}
            <div className="mt-4">
              <button 
                onClick={toggleDarkMode}
                className="flex items-center w-full px-4 py-2 text-sm text-gray-300 rounded-md hover:bg-gray-700"
              >
                {darkMode ? (
                  <>
                    <SunIcon className="h-5 w-5 mr-3 text-gray-400" />
                    Light Mode
                  </>
                ) : (
                  <>
                    <MoonIcon className="h-5 w-5 mr-3 text-gray-400" />
                    Dark Mode
                  </>
                )}
              </button>
            </div>
            
            {/* Version Info */}
            <div className="mt-4 text-xs text-gray-500 text-center">
              v1.0.0 - Alpha
            </div>
          </div>
        </div>
      </div>
    </>
  );

  function renderNavItems() {
    const navItems = [
      { name: 'Dashboard', href: '/', icon: HomeIcon, current: true },
      { name: 'Trading Bots', href: '/bots', icon: ArrowPathIcon, current: false },
      { name: 'Performance', href: '/performance', icon: ChartBarIcon, current: false },
      { name: 'Positions', href: '/positions', icon: CurrencyDollarIcon, current: false },
      { name: 'Notifications', href: '/notifications', icon: BellIcon, current: false },
      { name: 'Settings', href: '/settings', icon: CogIcon, current: false },
    ];

    return navItems.map((item) => (
      <Link href={item.href} key={item.name}>
        <a
          className={`flex items-center px-4 py-2 text-sm font-medium rounded-md ${
            item.current
              ? 'bg-gray-900 text-white'
              : 'text-gray-300 hover:bg-gray-700 hover:text-white'
          }`}
        >
          <item.icon
            className={`mr-3 h-5 w-5 ${
              item.current ? 'text-white' : 'text-gray-400'
            }`}
            aria-hidden="true"
          />
          {item.name}
        </a>
      </Link>
    ));
  }

  function renderEnvironmentSwitch() {
    return (
      <div className="px-2 space-y-1">
        <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Environment
        </div>
        <div className="flex space-x-1 bg-gray-700 p-1 rounded-md">
          <button
            className={`flex-1 py-1 px-3 text-sm font-medium rounded-md ${
              activeEnvironment === 'virtual'
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-600'
            }`}
            onClick={() => onEnvironmentChange('virtual')}
          >
            Virtual
          </button>
          <button
            className={`flex-1 py-1 px-3 text-sm font-medium rounded-md ${
              activeEnvironment === 'mainnet'
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-600'
            }`}
            onClick={() => onEnvironmentChange('mainnet')}
          >
            Mainnet
          </button>
        </div>
      </div>
    );
  }
};

export default Sidebar;
