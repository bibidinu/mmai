// API route for positions data
export default function handler(req, res) {
  const { symbol, environment } = req.query;
  
  // This would normally fetch data from your backend API
  // For demonstration, we'll return mock data
  
  // Generate position data
  let positions = [];
  
  // If symbol is specified, only return positions for that symbol
  if (symbol) {
    // Generate 1-2 positions for the requested symbol
    const count = Math.floor(Math.random() * 2) + 1;
    positions = generatePositions([symbol], count, environment);
  } else {
    // Generate positions for all symbols
    const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT'];
    // Generate more positions for mainnet
    const count = environment === 'mainnet' ? 5 : 3;
    positions = generatePositions(symbols, count, environment);
  }
  
  res.status(200).json(positions);
}

/**
 * Generate realistic position data
 */
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
    
    const position = {
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
    };
    
    positions.push(position);
  }
  
  return positions;
}
