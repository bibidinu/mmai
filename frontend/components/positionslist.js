import React from 'react';
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';

const PositionsList = ({ positions, onSymbolSelect }) => {
  if (!positions || positions.length === 0) {
    return (
      <div className="p-6">
        <h2 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Active Positions</h2>
        <div className="text-center py-10 text-gray-500 dark:text-gray-400">
          No active positions
        </div>
      </div>
    );
  }

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
      maximumFractionDigits: 2,
      signDisplay: 'always'
    }).format(value / 100);
  };

  const calculateTimeDifference = (timestamp) => {
    const now = new Date();
    const entryTime = new Date(timestamp);
    const diffMs = now - entryTime;
    const diffHrs = Math.floor(diffMs / (1000 * 60 * 60));
    const diffMins = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    
    if (diffHrs > 0) {
      return `${diffHrs}h ${diffMins}m`;
    } else {
      return `${diffMins}m`;
    }
  };

  const handlePositionClick = (symbol) => {
    if (onSymbolSelect) {
      onSymbolSelect(symbol);
    }
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Active Positions</h2>
        <span className="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded dark:bg-blue-900 dark:text-blue-300">
          {positions.length} Positions
        </span>
      </div>
      
      <div className="overflow-y-auto max-h-96">
        {positions.map((position) => (
          <div 
            key={position.id}
            className="border-b border-gray-200 dark:border-gray-700 py-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700"
            onClick={() => handlePositionClick(position.symbol)}
          >
            <div className="flex justify-between items-center mb-1">
              <div className="flex items-center">
                <span className={`w-2 h-2 rounded-full ${position.direction === 'long' ? 'bg-green-500' : 'bg-red-500'} mr-2`}></span>
                <span className="font-medium text-gray-900 dark:text-white">{position.symbol}</span>
                <span className={`ml-2 px-2 py-0.5 text-xs rounded ${position.direction === 'long' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300' : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'}`}>
                  {position.direction === 'long' ? 'LONG' : 'SHORT'}
                </span>
              </div>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {calculateTimeDifference(position.entryTime)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Entry: {position.entryPrice}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Strategy: {position.strategy}</div>
              </div>
              <div className="text-right">
                <div className={`font-medium flex items-center justify-end ${position.pnlPercentage >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                  {position.pnlPercentage >= 0 ? (
                    <ArrowUpIcon className="h-3 w-3 mr-1" />
                  ) : (
                    <ArrowDownIcon className="h-3 w-3 mr-1" />
                  )}
                  {formatPercentage(position.pnlPercentage)}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {formatCurrency(position.pnl)}
                </div>
              </div>
            </div>
            
            {/* Progress bar for TPs */}
            <div className="mt-2">
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                <span>Entry</span>
                <span>TP1</span>
                <span>TP2</span>
                <span>TP3</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                {position.tpHit1 && (
                  <div className="bg-green-500 h-full" style={{ width: '33.33%' }}></div>
                )}
                {position.tpHit2 && (
                  <div className="bg-green-500 h-full" style={{ width: '66.66%' }}></div>
                )}
                {position.tpHit3 && (
                  <div className="bg-green-500 h-full" style={{ width: '100%' }}></div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PositionsList;
