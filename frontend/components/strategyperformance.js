import React from 'react';
import { PieChart, Pie, Cell, Legend, ResponsiveContainer, Tooltip } from 'recharts';

const StrategyPerformance = ({ strategies }) => {
  if (!strategies || strategies.length === 0) return null;

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

  // Define chart colors
  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#6366F1', '#14B8A6', '#F97316', '#4F46E5'];

  // Prepare pie chart data for usage count
  const usageData = strategies.map((strategy, index) => ({
    name: strategy.name,
    value: strategy.usageCount,
    color: COLORS[index % COLORS.length]
  }));

  // Prepare pie chart data for profit
  const profitData = strategies.map((strategy, index) => ({
    name: strategy.name,
    value: Math.max(0, strategy.profit), // Only positive values
    color: COLORS[index % COLORS.length]
  }));

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Strategy Performance</h2>
        <div className="flex space-x-2">
          <select className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-white px-2 py-1 rounded">
            <option value="7d">7 Days</option>
            <option value="14d">14 Days</option>
            <option value="30d">30 Days</option>
            <option value="90d">90 Days</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        {/* Strategy Usage Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Strategy Usage</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={usageData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  labelLine={false}
                >
                  {usageData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [`${value} uses`, "Usage Count"]}
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    borderColor: '#374151',
                    color: '#F9FAFB',
                    borderRadius: '0.375rem'
                  }}
                  itemStyle={{ color: '#F9FAFB' }}
                  labelStyle={{ color: '#F9FAFB' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Strategy Profit Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Strategy Profit</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={profitData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                  labelLine={false}
                >
                  {profitData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [formatCurrency(value), "Profit"]}
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    borderColor: '#374151',
                    color: '#F9FAFB',
                    borderRadius: '0.375rem'
                  }}
                  itemStyle={{ color: '#F9FAFB' }}
                  labelStyle={{ color: '#F9FAFB' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Strategy performance table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm text-left text-gray-700 dark:text-gray-300">
          <thead className="text-xs uppercase bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
            <tr>
              <th className="px-4 py-2 rounded-l-lg">Strategy</th>
              <th className="px-4 py-2">Win Rate</th>
              <th className="px-4 py-2">Trades</th>
              <th className="px-4 py-2">Profit</th>
              <th className="px-4 py-2 rounded-r-lg">ROI</th>
            </tr>
          </thead>
          <tbody>
            {strategies.map((strategy, index) => (
              <tr key={index} className="border-b border-gray-200 dark:border-gray-700">
                <td className="px-4 py-3">
                  <div className="flex items-center">
                    <div className="w-3 h-3 rounded-full mr-2" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                    {strategy.name}
                  </div>
                </td>
                <td className="px-4 py-3">{formatPercentage(strategy.winRate)}</td>
                <td className="px-4 py-3">{strategy.totalTrades}</td>
                <td className={`px-4 py-3 ${strategy.profit >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {formatCurrency(strategy.profit)}
                </td>
                <td className={`px-4 py-3 ${strategy.roi >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {formatPercentage(strategy.roi)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StrategyPerformance;
