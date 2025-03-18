import React from 'react';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon, CurrencyDollarIcon, ClockIcon } from '@heroicons/react/24/outline';

const Dashboard = ({ data, environment }) => {
  if (!data) return null;

  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'bg-green-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'stopped':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value / 100);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex flex-col md:flex-row justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Trading Dashboard {environment === 'virtual' ? '(Virtual)' : '(Mainnet)'}
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Last updated: {new Date(data.lastUpdate).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center mt-4 md:mt-0">
          <div className="flex items-center">
            <span className="mr-2">Status:</span>
            <span className="flex items-center">
              <span className={`w-3 h-3 rounded-full ${getStatusColor(data.status)} mr-1`}></span>
              <span className="capitalize">{data.status}</span>
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Balance card */}
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-blue-600 bg-opacity-30">
              <CurrencyDollarIcon className="h-6 w-6 text-white" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-blue-100">Total Balance</p>
              <p className="text-xl font-semibold text-white">{formatCurrency(data.balance)}</p>
            </div>
          </div>
        </div>

        {/* Daily PnL card */}
        <div className={`bg-gradient-to-r ${data.dailyPnL >= 0 ? 'from-green-500 to-green-600' : 'from-red-500 to-red-600'} rounded-lg shadow p-4`}>
          <div className="flex items-center">
            <div className={`p-3 rounded-full ${data.dailyPnL >= 0 ? 'bg-green-600' : 'bg-red-600'} bg-opacity-30`}>
              {data.dailyPnL >= 0 ? (
                <ArrowTrendingUpIcon className="h-6 w-6 text-white" />
              ) : (
                <ArrowTrendingDownIcon className="h-6 w-6 text-white" />
              )}
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-green-100">Daily P&L</p>
              <p className="text-xl font-semibold text-white">{formatCurrency(data.dailyPnL)}</p>
            </div>
          </div>
        </div>

        {/* Monthly PnL card */}
        <div className={`bg-gradient-to-r ${data.monthlyPnL >= 0 ? 'from-emerald-500 to-emerald-600' : 'from-rose-500 to-rose-600'} rounded-lg shadow p-4`}>
          <div className="flex items-center">
            <div className={`p-3 rounded-full ${data.monthlyPnL >= 0 ? 'bg-emerald-600' : 'bg-rose-600'} bg-opacity-30`}>
              {data.monthlyPnL >= 0 ? (
                <ArrowTrendingUpIcon className="h-6 w-6 text-white" />
              ) : (
                <ArrowTrendingDownIcon className="h-6 w-6 text-white" />
              )}
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-emerald-100">Monthly P&L</p>
              <p className="text-xl font-semibold text-white">{formatCurrency(data.monthlyPnL)}</p>
              <p className="text-sm text-emerald-100">{formatPercentage(data.monthlyPnLPercentage)}</p>
            </div>
          </div>
        </div>

        {/* Active positions card */}
        <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="p-3 rounded-full bg-purple-600 bg-opacity-30">
              <ClockIcon className="h-6 w-6 text-white" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-purple-100">Active Positions</p>
              <p className="text-xl font-semibold text-white">{data.activePositions}</p>
              <p className="text-sm text-purple-100">Win rate: {formatPercentage(data.winRate)}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Drawdown</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{formatPercentage(data.drawdown)}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Sharpe Ratio</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{data.sharpeRatio.toFixed(2)}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Best Strategy</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{data.bestStrategy || "N/A"}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Active Since</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{data.activeSince || "N/A"}</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
