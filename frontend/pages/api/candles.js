// API route for candle data
import { fetchFromAPI } from '../../utils/api-config';

export default async function handler(req, res) {
  const { symbol, timeframe, environment } = req.query;
  
  try {
    const data = await fetchFromAPI('candles', { symbol, timeframe, environment });
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
