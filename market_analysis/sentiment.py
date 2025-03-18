"""
Sentiment Analysis Module with proper class naming and method implementation
"""
import logging
import time
import json
import requests
from datetime import datetime, timedelta
import threading
import random

class SentimentAnalyzer:
    """
    Analyzes news sentiment for crypto assets with improved error handling and fallback
    """
    
    def __init__(self, config):
        """
        Initialize the sentiment analyzer
        
        Args:
            config (dict): Sentiment analysis configuration
        """
        self.logger = logging.getLogger("sentiment_analyzer")
        self.config = config
        
        # API configurations
        self.apis = config.get("apis", [])
        self.enabled_apis = [api for api in self.apis if api.get("enabled", False)]
        
        # Check if we have enabled APIs
        if not self.enabled_apis:
            self.logger.warning("No enabled sentiment APIs - using simulated sentiment data")
        
        # Update settings
        self.update_interval = config.get("update_interval", 3600)  # Default: 1 hour
        self.max_news_age = config.get("max_news_age", 86400)  # Default: 24 hours
        
        # Symbol settings
        self.common_symbols = config.get("common_symbols", ["BTC", "ETH"])
        
        # Data storage
        self.news_data = []
        self.sentiment_scores = {}
        self.last_update = 0
        
        # Cache sentiment data to prevent excessive updates
        self.sentiment_cache = {}
        self.sentiment_cache_timestamp = 0
        
        # API rate limiting
        self.request_times = {}
        self.max_requests_per_minute = 10
        
        # Set up HTTP session with retry
        self.session = requests.Session()
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=5,
            pool_maxsize=5
        )
        self.session.mount('https://', retry_adapter)
        
        # Initialize data
        self.update()
        
        self.logger.info("Sentiment analyzer initialized")
    
    def update(self):
        """Update sentiment data if needed"""
        current_time = time.time()
        
        # Check if update is needed
        if current_time - self.last_update < self.update_interval:
            return
        
        self.logger.info("Updating sentiment data")
        
        try:
            # Update news data
            self._fetch_news()
            
            # Calculate sentiment scores
            self._calculate_sentiment()
            
            self.last_update = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating sentiment data: {str(e)}", exc_info=True)
            # Fall back to simulated data if real data not available
            if not self.sentiment_scores:
                self._generate_simulated_sentiment()
    
    def _fetch_news(self):
        """Fetch news data from configured APIs with improved error handling"""
        # Clear old news data
        self.news_data = []
        
        # If no enabled APIs, generate simulated data
        if not self.enabled_apis:
            self._generate_simulated_news()
            return
        
        # Try each API until we get data
        for api_config in self.enabled_apis:
            try:
                news = self._fetch_from_api(api_config)
                
                if news:
                    self.news_data.extend(news)
                    self.logger.info(f"Fetched {len(news)} news items from {api_config['name']}")
                    
                    # If we have enough data, stop
                    if len(self.news_data) >= 50:
                        break
                        
            except Exception as e:
                self.logger.error(f"Error fetching news from {api_config['name']}: {str(e)}")
        
        # If we couldn't get any news data, generate simulated data
        if not self.news_data:
            self.logger.warning("No news data fetched from APIs, using simulated data")
            self._generate_simulated_news()
        
        # Filter out old news
        current_time = time.time()
        self.news_data = [
            news for news in self.news_data 
            if current_time - news.get("timestamp", current_time) < self.max_news_age
        ]
    
    def _fetch_from_api(self, api_config):
        """
        Fetch news from a specific API with rate limiting and error handling
        
        Args:
            api_config (dict): API configuration
            
        Returns:
            list: List of news items
        """
        api_name = api_config.get("name", "unknown")
        api_type = api_config.get("type", "rest")
        url = api_config.get("url", "")
        params = api_config.get("params", {})
        news_path = api_config.get("news_path", "")
        
        # Check rate limiting
        if not self._check_rate_limit(api_name):
            self.logger.warning(f"Rate limit reached for {api_name}, skipping")
            return []
        
        # Make API request
        if api_type == "rest":
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract news items
                if news_path:
                    path_parts = news_path.split(".")
                    for part in path_parts:
                        if part in data:
                            data = data[part]
                        else:
                            self.logger.error(f"News path {news_path} not found in API response")
                            return []
                
                # Record request time for rate limiting
                api_key = api_name
                if api_key not in self.request_times:
                    self.request_times[api_key] = []
                self.request_times[api_key].append(time.time())
                
                # Process news items
                return self._process_news_items(data, api_name)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request error for {api_name}: {str(e)}")
                return []
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON from {api_name}: {str(e)}")
                return []
        
        # Other API types not implemented
        self.logger.error(f"Unsupported API type: {api_type}")
        return []
    
    def _process_news_items(self, items, source):
        """
        Process raw news items into standardized format
        
        Args:
            items (list): Raw news items
            source (str): News source name
            
        Returns:
            list: Processed news items
        """
        processed = []
        
        try:
            # Ensure items is a list
            if isinstance(items, dict):
                items = [items]
            
            for item in items:
                # Extract relevant fields
                title = item.get("title", "")
                if not title:
                    continue
                    
                # Extract timestamp
                published_at = item.get("published_at") or item.get("created_at") or item.get("date", "")
                if published_at:
                    try:
                        # Try parsing ISO format
                        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                        timestamp = int(dt.timestamp())
                    except (ValueError, TypeError):
                        timestamp = int(time.time())
                else:
                    timestamp = int(time.time())
                
                # Extract URL
                url = item.get("url", "")
                
                # Extract content or description
                content = item.get("content", "") or item.get("description", "") or ""
                
                # Extract sentiment if available
                sentiment = item.get("sentiment", None)
                if sentiment is None:
                    # Simulate sentiment if not provided
                    sentiment = self._analyze_text_sentiment(title + " " + content)
                
                # Extract mentioned symbols
                symbols = self._extract_symbols(title + " " + content)
                
                processed.append({
                    "title": title,
                    "content": content,
                    "url": url,
                    "timestamp": timestamp,
                    "source": source,
                    "sentiment": sentiment,
                    "symbols": symbols
                })
                
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing news items: {str(e)}")
            return []
    
    def _analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score (-1.0 to 1.0)
        """
        # Simple rule-based sentiment analysis
        positive_words = [
            "bullish", "buy", "gain", "profit", "increase", "up", "high", "rise", "surge", "rally",
            "breakthrough", "success", "good", "great", "best", "positive", "optimistic", "promising",
            "growth", "growing", "soaring", "skyrocket", "impressive", "outperform", "excel"
        ]
        
        negative_words = [
            "bearish", "sell", "loss", "decrease", "down", "low", "fall", "plunge", "crash", "drop",
            "decline", "negative", "bad", "worst", "poor", "terrible", "pessimistic", "risk", "danger",
            "bankrupt", "liquidation", "underperform", "problem", "issue", "trouble", "warning", "bear"
        ]
        
        text = text.lower()
        
        # Count word occurrences
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
            
        # Calculate sentiment score
        return (positive_count - negative_count) / total
    
    def _extract_symbols(self, text):
        """
        Extract mentioned symbols from text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: Mentioned symbols
        """
        text = text.upper()
        mentioned = []
        
        for symbol in self.common_symbols:
            # Check if symbol is mentioned
            if symbol in text:
                mentioned.append(symbol)
                
        return mentioned
    
    def _check_rate_limit(self, api_key):
        """
        Check if we're within rate limits for an API
        
        Args:
            api_key (str): API key
            
        Returns:
            bool: True if within limits, False otherwise
        """
        if api_key not in self.request_times:
            return True
            
        current_time = time.time()
        
        # Remove old requests (older than 1 minute)
        self.request_times[api_key] = [t for t in self.request_times[api_key] if current_time - t < 60]
        
        # Check if we're at the limit
        return len(self.request_times[api_key]) < self.max_requests_per_minute
    
    def _calculate_sentiment(self):
        """Calculate sentiment scores for symbols"""
        # Clear old scores
        self.sentiment_scores = {}
        
        # Calculate max age for weighted scores
        current_time = time.time()
        max_age = self.max_news_age
        
        for news in self.news_data:
            sentiment = news.get("sentiment", 0.0)
            symbols = news.get("symbols", [])
            timestamp = news.get("timestamp", current_time)
            
            # Calculate age and weight
            age = current_time - timestamp
            weight = 1.0 - (age / max_age)  # Newer news has higher weight
            
            # Update sentiment for each symbol
            for symbol in symbols:
                if symbol not in self.sentiment_scores:
                    self.sentiment_scores[symbol] = {
                        "total_sentiment": 0.0,
                        "total_weight": 0.0,
                        "news_count": 0,
                        "recent_news": []
                    }
                
                # Update weighted sentiment
                self.sentiment_scores[symbol]["total_sentiment"] += sentiment * weight
                self.sentiment_scores[symbol]["total_weight"] += weight
                self.sentiment_scores[symbol]["news_count"] += 1
                
                # Keep track of most recent news
                self.sentiment_scores[symbol]["recent_news"].append({
                    "title": news.get("title", ""),
                    "timestamp": timestamp,
                    "sentiment": sentiment,
                    "source": news.get("source", "")
                })
                
                # Keep only the 5 most recent news items
                self.sentiment_scores[symbol]["recent_news"] = sorted(
                    self.sentiment_scores[symbol]["recent_news"],
                    key=lambda x: x["timestamp"],
                    reverse=True
                )[:5]
        
        # Calculate final scores
        for symbol, data in self.sentiment_scores.items():
            if data["total_weight"] > 0:
                data["score"] = data["total_sentiment"] / data["total_weight"]
            else:
                data["score"] = 0.0
                
        # Update cache
        self.sentiment_cache = self.sentiment_scores.copy()
        self.sentiment_cache_timestamp = current_time
    
    def _generate_simulated_news(self):
        """Generate simulated news data when APIs fail"""
        self.logger.info("Generating simulated news data")
        
        current_time = time.time()
        simulated_news = []
        
        # Generate news for common symbols
        for symbol in self.common_symbols:
            # Generate 1-3 news items per symbol
            for _ in range(random.randint(1, 3)):
                # Generate random sentiment
                sentiment = random.uniform(-0.8, 0.8)
                
                # Generate title based on sentiment
                if sentiment > 0.3:
                    templates = [
                        f"{symbol} shows bullish signals as market recovers",
                        f"Analysts predict gains for {symbol} in coming weeks",
                        f"{symbol} breaks resistance level, indicating potential rally",
                        f"Positive developments for {symbol} ecosystem boosting confidence",
                        f"{symbol} gaining momentum with increased adoption"
                    ]
                elif sentiment < -0.3:
                    templates = [
                        f"{symbol} faces selling pressure amid market uncertainty",
                        f"Bearish pattern emerges for {symbol}, traders cautious",
                        f"{symbol} drops following negative market sentiment",
                        f"Analysts warn of potential correction for {symbol}",
                        f"Regulatory concerns affect {symbol} price action"
                    ]
                else:
                    templates = [
                        f"{symbol} trades sideways as market consolidates",
                        f"Mixed signals for {symbol} as traders await direction",
                        f"{symbol} shows low volatility amid market indecision",
                        f"Technical analysis shows neutral stance for {symbol}",
                        f"Traders monitor {symbol} for breakout signals"
                    ]
                
                title = random.choice(templates)
                
                # Generate random timestamp (within the last 24 hours)
                timestamp = current_time - random.randint(0, self.max_news_age)
                
                # Create news item
                news_item = {
                    "title": title,
                    "content": f"Simulated news content for {symbol}.",
                    "url": "",
                    "timestamp": timestamp,
                    "source": "simulated",
                    "sentiment": sentiment,
                    "symbols": [symbol]
                }
                
                simulated_news.append(news_item)
        
        # Add some market-wide news
        market_news_templates = [
            "Crypto market shows signs of recovery after recent correction",
            "Bitcoin dominance affects altcoin performance",
            "Market volatility increases as trading volume surges",
            "Institutional investors continue to enter crypto space",
            "Regulatory developments impact market sentiment",
            "DeFi tokens gain attention amid protocol innovations",
            "NFT market shows renewed interest after correction",
            "Layer-2 solutions gaining traction for scaling blockchains"
        ]
        
        for _ in range(3):
            title = random.choice(market_news_templates)
            sentiment = random.uniform(-0.5, 0.5)
            timestamp = current_time - random.randint(0, self.max_news_age)
            
            # Extract mentioned symbols
            mentioned = [symbol for symbol in self.common_symbols if symbol in title.upper()]
            if not mentioned:
                mentioned = random.sample(self.common_symbols, 2)  # Include 2 random symbols
            
            market_news = {
                "title": title,
                "content": "Simulated market news content.",
                "url": "",
                "timestamp": timestamp,
                "source": "simulated",
                "sentiment": sentiment,
                "symbols": mentioned
            }
            
            simulated_news.append(market_news)
        
        self.news_data = simulated_news
    
    def _generate_simulated_sentiment(self):
        """Generate simulated sentiment scores when calculation fails"""
        self.logger.info("Generating simulated sentiment scores")
        
        # Generate scores for common symbols
        for symbol in self.common_symbols:
            # Generate random sentiment score (-0.7 to 0.7)
            score = random.uniform(-0.7, 0.7)
            
            # Create simulated sentiment data
            self.sentiment_scores[symbol] = {
                "score": score,
                "news_count": random.randint(1, 5),
                "total_sentiment": score,
                "total_weight": 1.0,
                "recent_news": [
                    {
                        "title": f"Simulated news for {symbol}",
                        "timestamp": int(time.time()),
                        "sentiment": score,
                        "source": "simulated"
                    }
                ]
            }
        
        # Update cache
        self.sentiment_cache = self.sentiment_scores.copy()
        self.sentiment_cache_timestamp = time.time()
    
    def get_sentiment(self, symbol):
        """
        Get sentiment score for a symbol
        
        Args:
            symbol (str): Symbol to get sentiment for
            
        Returns:
            float: Sentiment score (-1.0 to 1.0)
        """
        # Convert to standard format (BTC, ETH, etc.)
        std_symbol = symbol.replace("USDT", "").replace("USD", "")
        
        # Use cached value if available and recent
        current_time = time.time()
        if std_symbol in self.sentiment_cache and current_time - self.sentiment_cache_timestamp < self.update_interval:
            return self.sentiment_cache[std_symbol].get("score", 0.0)
        
        # Update if needed
        if current_time - self.last_update >= self.update_interval:
            self.update()
        
        # Return sentiment score
        if std_symbol in self.sentiment_scores:
            return self.sentiment_scores[std_symbol].get("score", 0.0)
            
        # Return 0 for unknown symbols
        return 0.0
    
    def get_sentiment_score(self, symbol):
        """
        Alias for get_sentiment() for backward compatibility
        
        Args:
            symbol (str): Symbol to get sentiment for
            
        Returns:
            float: Sentiment score (-1.0 to 1.0)
        """
        return self.get_sentiment(symbol)
    
    def get_symbol_sentiment_data(self, symbol):
        """
        Get detailed sentiment data for a symbol
        
        Args:
            symbol (str): Symbol to get sentiment for
            
        Returns:
            dict: Sentiment data including score, news count, and recent news
        """
        # Convert to standard format (BTC, ETH, etc.)
        std_symbol = symbol.replace("USDT", "").replace("USD", "")
        
        # Update if needed
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.update()
        
        # Return sentiment data
        if std_symbol in self.sentiment_scores:
            return self.sentiment_scores[std_symbol]
            
        # Return empty data for unknown symbols
        return {
            "score": 0.0,
            "news_count": 0,
            "total_sentiment": 0.0,
            "total_weight": 0.0,
            "recent_news": []
        }
    
    def get_market_sentiment(self):
        """
        Get overall market sentiment
        
        Returns:
            float: Overall market sentiment score (-1.0 to 1.0)
        """
        # Update if needed
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.update()
        
        # Calculate average sentiment across all symbols
        total_score = 0.0
        total_weight = 0.0
        
        for symbol, data in self.sentiment_scores.items():
            score = data.get("score", 0.0)
            weight = data.get("news_count", 0)
            
            total_score += score * weight
            total_weight += weight
        
        # Return average sentiment
        if total_weight > 0:
            return total_score / total_weight
            
        # Return neutral sentiment if no data
        return 0.0
    
    def get_trending_symbols(self, limit=5):
        """
        Get symbols with the most news activity
        
        Args:
            limit (int): Maximum number of symbols to return
            
        Returns:
            list: List of symbols with the most news activity
        """
        # Update if needed
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.update()
        
        # Sort symbols by news count
        trending = sorted(
            self.sentiment_scores.items(),
            key=lambda x: x[1].get("news_count", 0),
            reverse=True
        )
        
        # Return top symbols
        return [symbol for symbol, _ in trending[:limit]]
