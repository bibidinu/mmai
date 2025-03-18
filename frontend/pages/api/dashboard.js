// API route for dashboard data
import { fetchFromAPI } from '../../utils/api-config';

export default async function handler(req, res) {
  const { environment } = req.query;
  
  try {
    const data = await fetchFromAPI('dashboard', { environment });
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}
