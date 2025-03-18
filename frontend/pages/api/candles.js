// API route for candle data
export default function handler(req, res) {
  const { symbol, timeframe, environment } = req.query;
  
  // This would normally fetch data from your backend API
  // For demonstration, we'll return mock data
  
  // Generate candle data (200 candles)
  const candles = generateCandleData(symbol, timeframe, 200);
  
  res.status(200).json(candles);
}

/**
 * Generate realistic candle data with trends, volatility and volume
 */
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
