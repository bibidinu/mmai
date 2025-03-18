import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PerformanceMetrics = ({ metrics }) => {
  if (!metrics) return null;

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value / 100);
  };

  // Prepare chart data
  const chartData = metrics.dailyPerformance.map(day => ({
    date: new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    balance: day.balance,
    pnl: day.pnl
  }));

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Performance Metrics</h2>
        <div className="flex space-x-2">
          <select className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-white px-2 py-1 rounded">
            <option value="7d">7 Days</option>
            <option value="14d">14 Days</option>
            <option value="30d">30 Days</option>
            <option value="90d">90 Days</option>
          </select>
        </div>
      </div>

      {/* Performance Chart */}
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.2} />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              tickLine={{ stroke: '#4B5563' }}
              axisLine={{ stroke: '#4B5563' }}
            />
            <YAxis 
              tick={{ fontSize: 12, fill: '#9CA3AF' }}
              tickLine={{ stroke: '#4B5563' }}
              axisLine={{ stroke: '#4B5563' }}
              tickFormatter={formatCurrency}
            />
            <Tooltip
              formatter={(value) => [formatCurrency(value), "Balance"]}
              labelFormatter={(label) => `Date: ${label}`}
              contentStyle={{
                backgroundColor: '#1F2937',
                borderColor: '#374151',
                color: '#F9FAFB',
                borderRadius: '0.375rem'
              }}
              itemStyle={{ color: '#F9FAFB' }}
              labelStyle={{ color: '#F9FAFB' }}
            />
            <Line 
              type="monotone" 
              dataKey="balance" 
              stroke="#3B82F6" 
              strokeWidth={2}
              dot={{ r: 3, fill: '#3B82F6', stroke: '#3B82F6', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Key Performance Indicators */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Total Trades</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{metrics.totalTrades}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Win Rate</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{formatPercentage(metrics.winRate)}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">{metrics.wins}W / {metrics.losses}L</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Profit Factor</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{metrics.profitFactor.toFixed(2)}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Avg. Trade</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{formatCurrency(metrics.averageTrade)}</p>
          <p className="text-xs text-gray-500 dark:text-gray-400">{metrics.averageTradePercentage.toFixed(2)}%</p>
        </div>
      </div>

      {/* More Performance Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Max Drawdown</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{formatPercentage(metrics.maxDrawdown)}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Monthly Return</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{formatPercentage(metrics.monthlyReturn)}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Sharpe Ratio</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{metrics.sharpeRatio.toFixed(2)}</p>
        </div>
        <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded-lg">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Avg. Trade Duration</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">{metrics.avgTradeDuration}</p>
        </div>
      </div>
    </div>
  );
};

export default PerformanceMetrics;
