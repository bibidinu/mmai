import React, { useState } from 'react';
import { BellAlertIcon, CheckCircleIcon, ExclamationTriangleIcon, XCircleIcon, InformationCircleIcon } from '@heroicons/react/24/outline';

const AlertCenter = ({ alerts }) => {
  const [filter, setFilter] = useState('all');

  if (!alerts || alerts.length === 0) {
    return (
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Notifications</h2>
        </div>
        <div className="text-center py-10 text-gray-500 dark:text-gray-400">
          No notifications
        </div>
      </div>
    );
  }

  const getAlertIcon = (type) => {
    switch (type) {
      case 'success':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />;
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
      case 'info':
      default:
        return <InformationCircleIcon className="h-5 w-5 text-blue-500" />;
    }
  };

  const getAlertBgClass = (type) => {
    switch (type) {
      case 'success':
        return 'bg-green-50 dark:bg-green-900 dark:bg-opacity-10';
      case 'warning':
        return 'bg-yellow-50 dark:bg-yellow-900 dark:bg-opacity-10';
      case 'error':
        return 'bg-red-50 dark:bg-red-900 dark:bg-opacity-10';
      case 'info':
      default:
        return 'bg-blue-50 dark:bg-blue-900 dark:bg-opacity-10';
    }
  };

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const alertTime = new Date(timestamp);
    const diffMs = now - alertTime;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHrs = Math.floor(diffMin / 60);
    const diffDays = Math.floor(diffHrs / 24);

    if (diffDays > 0) {
      return `${diffDays}d ago`;
    } else if (diffHrs > 0) {
      return `${diffHrs}h ago`;
    } else if (diffMin > 0) {
      return `${diffMin}m ago`;
    } else {
      return 'Just now';
    }
  };

  // Filter alerts based on selected filter
  const filteredAlerts = filter === 'all' 
    ? alerts 
    : alerts.filter(alert => alert.type === filter);

  return (
    <div className="p-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 gap-3 sm:gap-0">
        <div className="flex items-center">
          <BellAlertIcon className="h-5 w-5 mr-2 text-gray-700 dark:text-gray-300" />
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Notifications</h2>
          <span className="ml-2 bg-gray-100 text-gray-800 text-xs font-medium px-2.5 py-0.5 rounded dark:bg-gray-700 dark:text-gray-300">
            {alerts.length}
          </span>
        </div>
        
        <div className="flex space-x-2">
          <button 
            className={`px-3 py-1 text-xs font-medium rounded-full ${filter === 'all' ? 'bg-gray-200 dark:bg-gray-600' : 'bg-gray-100 dark:bg-gray-700'}`}
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button 
            className={`px-3 py-1 text-xs font-medium rounded-full ${filter === 'info' ? 'bg-blue-200 text-blue-800 dark:bg-blue-900 dark:text-blue-300' : 'bg-gray-100 dark:bg-gray-700'}`}
            onClick={() => setFilter('info')}
          >
            Info
          </button>
          <button 
            className={`px-3 py-1 text-xs font-medium rounded-full ${filter === 'success' ? 'bg-green-200 text-green-800 dark:bg-green-900 dark:text-green-300' : 'bg-gray-100 dark:bg-gray-700'}`}
            onClick={() => setFilter('success')}
          >
            Success
          </button>
          <button 
            className={`px-3 py-1 text-xs font-medium rounded-full ${filter === 'warning' ? 'bg-yellow-200 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300' : 'bg-gray-100 dark:bg-gray-700'}`}
            onClick={() => setFilter('warning')}
          >
            Warning
          </button>
          <button 
            className={`px-3 py-1 text-xs font-medium rounded-full ${filter === 'error' ? 'bg-red-200 text-red-800 dark:bg-red-900 dark:text-red-300' : 'bg-gray-100 dark:bg-gray-700'}`}
            onClick={() => setFilter('error')}
          >
            Error
          </button>
        </div>
      </div>
      
      <div className="space-y-4 max-h-96 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-10 text-gray-500 dark:text-gray-400">
            No notifications match the selected filter
          </div>
        ) : (
          filteredAlerts.map((alert, index) => (
            <div 
              key={index} 
              className={`p-4 rounded-lg flex items-start ${getAlertBgClass(alert.type)}`}
            >
              <div className="flex-shrink-0 mr-3 mt-0.5">
                {getAlertIcon(alert.type)}
              </div>
              <div className="flex-1">
                <div className="flex justify-between">
                  <p className="font-medium text-sm text-gray-900 dark:text-white">{alert.title}</p>
                  <span className="text-xs text-gray-500 dark:text-gray-400">{formatTimeAgo(alert.timestamp)}</span>
                </div>
                <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">{alert.message}</p>
                {alert.actions && (
                  <div className="mt-2 flex space-x-2">
                    {alert.actions.map((action, actionIndex) => (
                      <button 
                        key={actionIndex}
                        className="text-xs font-medium text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300"
                      >
                        {action}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default AlertCenter;
