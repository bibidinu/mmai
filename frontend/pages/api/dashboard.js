// API route for dashboard data
export default function handler(req, res) {
  const { environment } = req.query;
  
  // This would normally fetch data from your backend API
  // For demonstration, we'll return mock data
  
  // Create consistent but different data for virtual vs mainnet
  const isVirtual = environment === 'virtual';
  const multiplier = isVirtual ? 1 : 1.5;  // Make mainnet values higher
  
  // Overview data
  const overview = {
    lastUpdate: new Date().toISOString(),
    status: isVirtual ? 'running' : 'running',
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
  };
  
  // Generate positions
  const positions = [];
  const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT'];
  const strategies = ['Breakout', 'Trend Following', 'Mean Reversion', 'Range', 'Volatility'];
  
  for (let i = 0; i < (isVirtual ? 3 : 5); i++) {
    const direction = Math.random() > 0.5 ? 'long' : 'short';
    const entryPrice = 1000 + Math.random() * 1000;
    const currentPrice = entryPrice * (1 + (direction === 'long' ? 0.05 : -0.05) * Math.random());
    const pnlPercentage = direction === 'long' 
      ? ((currentPrice - entryPrice) / entryPrice) * 100
      : ((entryPrice - currentPrice) / entryPrice) * 100;
    
    positions.push({
      id: `pos-${i}`,
      symbol: symbols[i % symbols.length],
      direction,
      entryPrice: entryPrice.toFixed(2),
      currentPrice: currentPrice.toFixed(2),
      pnl: ((pnlPercentage / 100) * (1000 * multiplier)).toFixed(2),
      pnlPercentage,
      strategy: strategies[i % strategies.length],
      entryTime: new Date(Date.now() - Math.random() * 8640000).toISOString(), // Random time within last 24h
      tpHit1: Math.random() > 0.7,
      tpHit2: Math.random() > 0.8,
      tpHit3: false
    });
  }
  
  // Generate performance data
  const performance = {
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
    dailyPerformance: []
  };
  
  // Generate daily performance data for the chart (last 30 days)
  let balance = 10000 - (1500 * multiplier); // Starting balance 30 days ago
  for (let i = 30; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    
    // Generate some random daily changes, trending upward
    const dailyChange = (Math.random() * 200 - 50) * multiplier;
    balance += dailyChange;
    
    performance.dailyPerformance.push({
      date: date.toISOString(),
      balance,
      pnl: dailyChange
    });
  }
  
  // Generate strategy performance data
  const strategies = [
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
  
  // Generate alerts
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
  
  // Return the complete dashboard data
  res.status(200).json({
    overview,
    positions,
    performance,
    strategies,
    alerts
  });
}
