/**
 * API configuration for the dashboard
 */

// Get API URL from environment or use default
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

/**
 * Fetch data from the API
 * @param {string} endpoint - API endpoint
 * @param {object} params - Query parameters
 * @returns {Promise} - Promise with the API response
 */
export async function fetchFromAPI(endpoint, params = {}) {
  // Build query string
  const queryString = Object.keys(params)
    .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
    .join('&');
    
  // Build URL
  const url = `${API_URL}/api/${endpoint}${queryString ? `?${queryString}` : ''}`;
  
  try {
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error fetching from API: ${error.message}`);
    
    // Return mock data if API is unavailable
    return getMockData(endpoint, params);
  }
}

/**
 * Get mock data for when API is unavailable
 * This is the same data that was originally in the API routes
 */
function getMockData(endpoint, params = {}) {
  // Use the original mock data from the API routes
  const { environment } = params;
  const isVirtual = environment === 'virtual';
  const multiplier = isVirtual ? 1 : 1.5;
  
  switch (endpoint) {
    case 'dashboard':
      // Return dashboard data
      return {
        overview: {
          lastUpdate: new Date().toISOString(),
          status: 'running',
          balance: 10000 * multiplier,
          dailyPnL: 120 * multiplier,
          monthlyPnL: 1450 * multiplier,
          monthlyPnLPercentage: 14.5 * multiplier,
          activePositions: isVirtual ? 3 : 5,
          winRate: 65,
          drawdown: -8.2,
          sharpeRatio: 2.3 * multiplier,
          bestStrategy: 'Breakout',
          activeSince: '2023-01-15'
        },
        positions: generatePositions(
          ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT'],
          isVirtual ? 3 : 5,
          environment
        ),
        performance: generatePerformanceData(multiplier),
        strategies: generateStrategyData(multiplier),
        alerts: generateAlertData(isVirtual)
      };
      
    case 'positions':
      // Return positions data
      const { symbol } = params;
      if (symbol) {
        // Generate 1-2 positions for the requested symbol
        const count = Math.floor(Math.random() * 2) + 1;
        return generatePositions([symbol], count, environment);
      } else {
        // Generate positions for all symbols
        const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT'];
        // Generate more positions for mainnet
        const count = environment === 'mainnet' ? 5 : 3;
        return generatePositions(symbols, count, environment);
      }
      
    case 'candles':
      // Return candle data
      const { symbol: candleSymbol, timeframe } = params;
      return generateCandleData(candleSymbol, timeframe, 200);
      
    default:
      return { error: 'Invalid endpoint' };
  }
}

// Mock data generator functions
function generatePositions(symbols, count, environment) {
  const positions = [];
  const strategies = ['Breakout', 'Trend Following', 'Mean Reversion', 'Range', 'Volatility'];
  const multiplier = environment === 'mainnet' ? 1.5 : 1;
  
  for (let i = 0; i < count; i++) {
    const symbol = symbols[i % symbols.length];
    const direction = Math.random() > 0.5 ? 'long' : 'short';
    
    // Set base price based on symbol
    let currentPrice;
    switch (symbol) {
      case 'BTCUSDT':
        currentPrice = 26500;
        break;
      case 'ETHUSDT':
        currentPrice = 1850;
        break;
      case 'SOLUSDT':
        currentPrice = 32;
        break;
      case 'BNBUSDT':
        currentPrice = 245;
        break;
      case 'ADAUSDT':
        currentPrice = 0.35;
        break;
      default:
        currentPrice = 100;
    }
    
    // Generate entry price with small deviation from current price
    const entryPrice = currentPrice * (1 + (Math.random() * 0.04 - 0.02)); // ±2% from current
    
    // Calculate TPs and SL
    let tp1, tp2, tp3, stopLoss;
    if (direction === 'long') {
      tp1 = entryPrice * 1.015; // +1.5%
      tp2 = entryPrice * 1.03;  // +3%
      tp3 = entryPrice * 1.05;  // +5%
      stopLoss = entryPrice * 0.98; // -2%
    } else {
      tp1 = entryPrice * 0.985; // -1.5%
      tp2 = entryPrice * 0.97;  // -3%
      tp3 = entryPrice * 0.95;  // -5%
      stopLoss = entryPrice * 1.02; // +2%
    }
    
    // Generate random entry time (between 5 min and 24 hours ago)
    const now = Date.now();
    const entryTime = now - (5 * 60 * 1000 + Math.random() * (24 * 60 * 60 * 1000 - 5 * 60 * 1000));
    
    // Generate TP hit times if appropriate
    const minutesSinceEntry = (now - entryTime) / (60 * 1000);
    const tp1Hit = Math.random() > 0.7;
    const tp1Time = tp1Hit ? entryTime + (minutesSinceEntry * 0.3 * 60 * 1000) : null;
    
    const tp2Hit = tp1Hit && Math.random() > 0.7;
    const tp2Time = tp2Hit ? entryTime + (minutesSinceEntry * 0.6 * 60 * 1000) : null;
    
    const tp3Hit = tp2Hit && Math.random() > 0.8;
    const tp3Time = tp3Hit ? entryTime + (minutesSinceEntry * 0.9 * 60 * 1000) : null;
    
    // Calculate PnL
    let pnlPercentage;
    if (direction === 'long') {
      pnlPercentage = ((currentPrice - entryPrice) / entryPrice) * 100;
    } else {
      pnlPercentage = ((entryPrice - currentPrice) / entryPrice) * 100;
    }
    
    positions.push({
      id: `pos-${symbol}-${i}`,
      symbol,
      direction,
      entryPrice: entryPrice.toFixed(symbol === 'ADAUSDT' ? 5 : 2),
      currentPrice: currentPrice.toFixed(symbol === 'ADAUSDT' ? 5 : 2),
      tp1: tp1.toFixed(symbol === 'ADAUSDT' ? 5 : 2),
      tp2: tp2.toFixed(symbol === 'ADAUSDT' ? 5 : 2),
      tp3: tp3.toFixed(symbol === 'ADAUSDT' ? 5 : 2),
      stopLoss: stopLoss.toFixed(symbol === 'ADAUSDT' ? 5 : 2),
      strategy: strategies[Math.floor(Math.random() * strategies.length)],
      entryTime: new Date(entryTime).toISOString(),
      pnl: ((pnlPercentage / 100) * (1000 * multiplier)).toFixed(2),
      pnlPercentage,
      tp1Hit,
      tp1Time: tp1Time ? new Date(tp1Time).toISOString() : null,
      tp2Hit,
      tp2Time: tp2Time ? new Date(tp2Time).toISOString() : null,
      tp3Hit,
      tp3Time: tp3Time ? new Date(tp3Time).toISOString() : null,
      size: (1000 * multiplier).toFixed(2)
    });
  }
  
  return positions;
}

function generatePerformanceData(multiplier) {
  // Generate daily performance data for the chart (last 30 days)
  const dailyPerformance = [];
  let balance = 10000 - (1500 * multiplier); // Starting balance 30 days ago
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    
    // Generate some random daily changes, trending upward
    const dailyChange = (Math.random() * 200 - 50) * multiplier;
    balance += dailyChange;
    
    dailyPerformance.push({
      date: date.toISOString(),
      balance,
      pnl: dailyChange
    });
  }
  
  return {
    totalTrades: 142 * multiplier,
    winRate: 65.3,
    wins: Math.floor(92 * multiplier),
    losses: Math.floor(50 * multiplier),
    profitFactor: 1.87 * multiplier,
    averageTrade: 25.4 * multiplier,
    averageTradePercentage: 0.76 * multiplier,
    maxDrawdown: 12.4,
    monthlyReturn: 14.5 * multiplier,
    sharpeRatio: 2.3 * multiplier,
    avgTradeDuration: '3h 42m',
    dailyPerformance
  };
}

function generateStrategyData(multiplier) {
  return [
    {
      name: 'Breakout',
      winRate: 72.5,
      totalTrades: 40 * multiplier,
      profit: 580 * multiplier,
      roi: 5.8 * multiplier,
      usageCount: 40 * multiplier
    },
    {
      name: 'Trend Following',
      winRate: 68.2,
      totalTrades: 35 * multiplier,
      profit: 420 * multiplier,
      roi: 4.2 * multiplier,
      usageCount: 35 * multiplier
    },
    {
      name: 'Mean Reversion',
      winRate: 60.0,
      totalTrades: 25 * multiplier,
      profit: 320 * multiplier,
      roi: 3.2 * multiplier,
      usageCount: 25 * multiplier
    },
    {
      name: 'Range',
      winRate: 63.6,
      totalTrades: 22 * multiplier,
      profit: 280 * multiplier,
      roi: 2.8 * multiplier,
      usageCount: 22 * multiplier
    },
    {
      name: 'Volatility',
      winRate: 55.0,
      totalTrades: 20 * multiplier,
      profit: 150 * multiplier,
      roi: 1.5 * multiplier,
      usageCount: 20 * multiplier
    }
  ];
}

function generateAlertData(isVirtual) {
  const alerts = [
    {
      type: 'success',
      title: 'Trade Closed',
      message: 'BTCUSDT long position closed with profit of $120.50 (2.3%)',
      timestamp: new Date(Date.now() - 1800000).toISOString(),
      actions: ['View Details']
    },
    {
      type: 'info',
      title: 'New Position Opened',
      message: 'ETHUSDT short position opened at $1,845.20 using Trend Following strategy',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      actions: ['View Position']
    },
    {
      type: 'warning',
      title: 'High Volatility Detected',
      message: 'Market volatility for BTC has increased. Position sizing adjusted accordingly.',
      timestamp: new Date(Date.now() - 7200000).toISOString()
    }
  ];
  
  // If mainnet, add some extra alerts
  if (!isVirtual) {
    alerts.push({
      type: 'error',
      title: 'API Connection Issue',
      message: 'Temporary connection issue with ByBit API. Reconnecting...',
      timestamp: new Date(Date.now() - 5400000).toISOString(),
      actions: ['View Logs', 'Restart']
    });
    alerts.push({
      type: 'success',
      title: 'Take Profit Hit',
      message: 'SOLUSDT long position reached TP1 (32.5%). Moved stop loss to breakeven.',
      timestamp: new Date(Date.now() - 10800000).toISOString(),
      actions: ['View Position']
    });
  }
  
  return alerts;
}

function generateCandleData(symbol, timeframe, count) {
  // Set base price based on symbol
  let basePrice;
  switch (symbol) {
    case 'BTCUSDT':
      basePrice = 26500;
      break;
    case 'ETHUSDT':
      basePrice = 1850;
      break;
    case 'SOLUSDT':
      basePrice = 32;
      break;
    case 'BNBUSDT':
      basePrice = 245;
      break;
    case 'ADAUSDT':
      basePrice = 0.35;
      break;
    default:
      basePrice = 100;
  }
  
  // Calculate interval in milliseconds
  let interval;
  switch (timeframe) {
    case '5m':
      interval = 5 * 60 * 1000;
      break;
    case '15m':
      interval = 15 * 60 * 1000;
      break;
    case '1h':
      interval = 60 * 60 * 1000;
      break;
    case '4h':
      interval = 4 * 60 * 60 * 1000;
      break;
    case '1d':
      interval = 24 * 60 * 60 * 1000;
      break;
    default:
      interval = 60 * 60 * 1000; // Default to 1h
  }
  
  // Set volatility based on symbol
  let volatility;
  switch (symbol) {
    case 'BTCUSDT':
      volatility = 0.015; // 1.5%
      break;
    case 'ETHUSDT':
      volatility = 0.02; // 2%
      break;
    case 'SOLUSDT':
      volatility = 0.035; // 3.5%
      break;
    case 'BNBUSDT':
      volatility = 0.025; // 2.5%
      break;
    case 'ADAUSDT':
      volatility = 0.03; // 3%
      break;
    default:
      volatility = 0.02; // 2%
  }
  
  // Generate candles with random walk
  const candles = [];
  let currentPrice = basePrice;
  let trend = 0; // No initial trend
  let trendStrength = 0;
  let trendDuration = 0;
  
  // End time is now
  const endTime = Date.now();
  
  for (let i = 0; i < count; i++) {
    // Update trend every ~20 candles
    if (i % 20 === 0 || trendDuration <= 0) {
      trend = Math.random() > 0.5 ? 1 : -1; // +1 for uptrend, -1 for downtrend
      trendStrength = Math.random() * 0.01; // 0-1% trend bias per candle
      trendDuration = Math.floor(Math.random() * 40) + 10; // 10-50 candles per trend
    }
    trendDuration--;
    
    // Calculate timestamp (going backward from now)
    const timestamp = endTime - ((count - i) * interval);
    
    // Random price change with trend bias
    const priceChange = (Math.random() * 2 - 1) * volatility * currentPrice;
    const trendChange = trend * trendStrength * currentPrice;
    currentPrice += priceChange + trendChange;
    
    // Ensure price doesn't go negative
    currentPrice = Math.max(currentPrice, basePrice * 0.1);
    
    // Generate candle
    const open = currentPrice;
    const close = currentPrice * (1 + (Math.random() * 0.01 - 0.005)); // ±0.5% from current
    const high = Math.max(open, close) * (1 + Math.random() * 0.005); // 0-0.5% above max
    const low = Math.min(open, close) * (1 - Math.random() * 0.005); // 0-0.5% below min
    
    // Volume increases with volatility
    const priceSwing = Math.abs(high - low) / currentPrice;
    const baseVolume = currentPrice * 100; // Higher price, higher base volume
    const volume = baseVolume * (1 + priceSwing * 10); // Volume increases with volatility
    
    candles.push({
      timestamp,
      open,
      high,
      low,
      close,
      volume
    });
    
    // Update current price to close price for next candle
    currentPrice = close;
  }
  
  return candles;
}
