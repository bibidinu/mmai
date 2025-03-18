// API route for positions data
import { fetchFromAPI } from '../../utils/api-config';

export default async function handler(req, res) {
  const { symbol, environment } = req.query;
  
  try {
    const data = await fetchFromAPI('positions', { symbol, environment });
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
