import React, { useState, useEffect, useRef } from 'react';
import { createChart, CrosshairMode } from 'lightweight-charts';

const TradingChart = ({ symbol, timeframe, environment }) => {
  const chartContainerRef = useRef(null);
  const [chartInstance, setChartInstance] = useState(null);
  const [candleSeries, setCandleSeries] = useState(null);
  const [volumeSeries, setVolumeSeries] = useState(null);
  const [positions, setPositions] = useState([]);
  const [activeTimeframe, setActiveTimeframe] = useState(timeframe || '1h');

  // Available timeframes
  const timeframes = ['5m', '15m', '1h', '4h', '1d'];

  useEffect(() => {
    // Initialize chart when component mounts
    if (chartContainerRef.current && !chartInstance) {
      const chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height: 500,
        layout: {
          backgroundColor: '#1E293B',
          textColor: '#D1D5DB',
        },
        grid: {
          vertLines: {
            color: '#334155',
          },
          horzLines: {
            color: '#334155',
          },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
        },
        rightPriceScale: {
          borderColor: '#475569',
        },
        timeScale: {
          borderColor: '#475569',
        },
      });

      // Create candlestick series
      const newCandleSeries = chart.addCandlestickSeries({
        upColor: '#10B981',
        downColor: '#EF4444',
        borderDownColor: '#EF4444',
        borderUpColor: '#10B981',
        wickDownColor: '#EF4444',
        wickUpColor: '#10B981',
      });

      // Create volume series
      const newVolumeSeries = chart.addHistogramSeries({
        color: '#60A5FA',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
          top: 0.85,
          bottom: 0,
        },
      });

      setChartInstance(chart);
      setCandleSeries(newCandleSeries);
      setVolumeSeries(newVolumeSeries);

      // Resize handler
      const handleResize = () => {
        if (chartContainerRef.current && chart) {
          chart.resize(
            chartContainerRef.current.clientWidth,
            chartContainerRef.current.clientHeight
          );
        }
      };

      window.addEventListener('resize', handleResize);

      // Cleanup function
      return () => {
        window.removeEventListener('resize', handleResize);
        if (chart) {
          chart.remove();
        }
      };
    }
  }, [chartContainerRef, chartInstance]);

  // Update data when symbol, timeframe, or environment changes
  useEffect(() => {
    if (candleSeries && volumeSeries) {
      // Fetch candle data
      fetchCandleData();
      // Fetch positions
      fetchPositionData();
    }
  }, [symbol, activeTimeframe, environment, candleSeries, volumeSeries]);

  // Fetch candle data from API
  const fetchCandleData = async () => {
    try {
      // In a real implementation, this would fetch from your API
      const response = await fetch(`/api/candles?symbol=${symbol}&timeframe=${activeTimeframe}&environment=${environment}`);
      if (!response.ok) {
        throw new Error('Failed to fetch candle data');
      }
      const data = await response.json();
      
      // Format data for lightweight-charts
      const candleData = data.map(candle => ({
        time: candle.timestamp / 1000, // Convert to seconds if in milliseconds
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close
      }));
      
      const volumeData = data.map(candle => ({
        time: candle.timestamp / 1000, // Convert to seconds if in milliseconds
        value: candle.volume,
        color: candle.close >= candle.open ? 'rgba(76, 175, 80, 0.5)' : 'rgba(255, 82, 82, 0.5)'
      }));
      
      candleSeries.setData(candleData);
      volumeSeries.setData(volumeData);
    } catch (error) {
      console.error('Error fetching candle data:', error);
    }
  };

  // Fetch position data from API
  const fetchPositionData = async () => {
    try {
      // In a real implementation, this would fetch from your API
      const response = await fetch(`/api/positions?symbol=${symbol}&environment=${environment}`);
      if (!response.ok) {
        throw new Error('Failed to fetch position data');
      }
      const data = await response.json();
      
      setPositions(data);
      
      // Add position markers to chart
      if (candleSeries) {
        // Clear existing markers
        candleSeries.setMarkers([]);
        
        const markers = [];
        
        data.forEach(position => {
          // Entry marker
          if (position.entryTime) {
            markers.push({
              time: position.entryTime / 1000,
              position: 'belowBar',
              color: position.direction === 'long' ? '#10B981' : '#EF4444',
              shape: position.direction === 'long' ? 'arrowUp' : 'arrowDown',
              text: `${position.direction.toUpperCase()} Entry`,
              size: 1
            });
          }
          
          // TP1 marker
          if (position.tp1Hit) {
            markers.push({
              time: position.tp1Time / 1000,
              position: 'aboveBar',
              color: '#10B981',
              shape: 'circle',
              text: 'TP1',
              size: 1
            });
          }
          
          // TP2 marker
          if (position.tp2Hit) {
            markers.push({
              time: position.tp2Time / 1000,
              position: 'aboveBar',
              color: '#10B981',
              shape: 'circle',
              text: 'TP2',
              size: 1
            });
          }
          
          // TP3/Exit marker
          if (position.tp3Hit || position.exitTime) {
            markers.push({
              time: (position.tp3Hit ? position.tp3Time : position.exitTime) / 1000,
              position: 'aboveBar',
              color: position.pnl >= 0 ? '#10B981' : '#EF4444',
              shape: 'circle',
              text: position.tp3Hit ? 'TP3' : 'Exit',
              size: 1
            });
          }
        });
        
        candleSeries.setMarkers(markers);
      }
    } catch (error) {
      console.error('Error fetching position data:', error);
    }
  };

  const handleTimeframeChange = (tf) => {
    setActiveTimeframe(tf);
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mr-2">{symbol}</h2>
          <div className="flex space-x-1">
            {timeframes.map(tf => (
              <button
                key={tf}
                className={`px-2 py-1 text-xs font-medium rounded ${
                  activeTimeframe === tf 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-white'
                }`}
                onClick={() => handleTimeframeChange(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <select className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-white px-2 py-1 rounded">
            <option value="indicators">+ Add Indicator</option>
            <option value="rsi">RSI</option>
            <option value="macd">MACD</option>
            <option value="bb">Bollinger Bands</option>
            <option value="ema">EMA</option>
          </select>
          <button className="text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-white px-2 py-1 rounded">
            Fullscreen
          </button>
        </div>
      </div>
      
      <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
        <div className="h-[500px]" ref={chartContainerRef} />
      </div>
      
      {/* Position Information */}
      {positions.length > 0 && (
        <div className="mt-4 bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Active Positions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {positions.map(position => (
              <div key={position.id} className="border border-gray-200 dark:border-gray-700 rounded p-3">
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center">
                    <span className={`w-2 h-2 rounded-full ${position.direction === 'long' ? 'bg-green-500' : 'bg-red-500'} mr-2`}></span>
                    <span className="font-medium">{position.direction.toUpperCase()}</span>
                  </div>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(position.entryTime).toLocaleString()}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>Entry: <span className="font-medium">{position.entryPrice}</span></div>
                  <div>Current: <span className="font-medium">{position.currentPrice}</span></div>
                  <div>TP1: <span className="font-medium">{position.tp1}</span></div>
                  <div>TP2: <span className="font-medium">{position.tp2}</span></div>
                  <div>TP3: <span className="font-medium">{position.tp3}</span></div>
                  <div>SL: <span className="font-medium">{position.stopLoss}</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingChart;
