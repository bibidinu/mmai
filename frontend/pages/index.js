import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Dashboard from '../components/Dashboard';
import Sidebar from '../components/Sidebar';
import PositionsList from '../components/PositionsList';
import PerformanceMetrics from '../components/PerformanceMetrics';
import StrategyPerformance from '../components/StrategyPerformance';
import TradingChart from '../components/TradingChart';
import AlertCenter from '../components/AlertCenter';

export default function Home() {
  const [activeEnvironment, setActiveEnvironment] = useState('virtual');
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');

  // Fetch dashboard data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // In a real implementation, this would be an API call
        const response = await fetch(`/api/dashboard?environment=${activeEnvironment}`);
        if (!response.ok) {
          throw new Error('Failed to fetch dashboard data');
        }
        const data = await response.json();
        setDashboardData(data);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
    // Set up polling interval (every 10 seconds)
    const interval = setInterval(fetchData, 10000);
    
    return () => clearInterval(interval);
  }, [activeEnvironment]);

  const handleEnvironmentChange = (env) => {
    setActiveEnvironment(env);
  };

  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
      <Head>
        <title>Adaptive Crypto Trading Bot - Dashboard</title>
        <meta name="description" content="Multi-strategy cryptocurrency trading bot dashboard" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="flex">
        {/* Sidebar */}
        <Sidebar 
          activeEnvironment={activeEnvironment} 
          onEnvironmentChange={handleEnvironmentChange} 
        />

        {/* Main content */}
        <main className="flex-1 p-6">
          {loading ? (
            <div className="flex items-center justify-center h-screen">
              <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
          ) : error ? (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              <p>{error}</p>
            </div>
          ) : dashboardData ? (
            <div className="space-y-6">
              {/* Dashboard header */}
              <Dashboard 
                data={dashboardData.overview} 
                environment={activeEnvironment} 
              />
              
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Trading chart */}
                <div className="lg:col-span-8 bg-white dark:bg-gray-800 rounded-lg shadow">
                  <TradingChart 
                    symbol={selectedSymbol} 
                    timeframe="1h"
                    environment={activeEnvironment}
                  />
                </div>
                
                {/* Active positions */}
                <div className="lg:col-span-4 bg-white dark:bg-gray-800 rounded-lg shadow">
                  <PositionsList 
                    positions={dashboardData.positions} 
                    onSymbolSelect={handleSymbolChange}
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                {/* Performance metrics */}
                <div className="lg:col-span-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                  <PerformanceMetrics 
                    metrics={dashboardData.performance}
                  />
                </div>
                
                {/* Strategy performance */}
                <div className="lg:col-span-6 bg-white dark:bg-gray-800 rounded-lg shadow">
                  <StrategyPerformance 
                    strategies={dashboardData.strategies}
                  />
                </div>
              </div>
              
              {/* Alert center */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
                <AlertCenter 
                  alerts={dashboardData.alerts}
                />
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}
