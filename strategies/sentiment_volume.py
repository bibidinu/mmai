"""
Sentiment-Volume Correlation Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd

class SentimentVolumeStrategy:
    """
    Strategy that trades based on correlation between sentiment and volume
    """
    
    def __init__(self, config, sentiment_analyzer, volume_analyzer):
        """
        Initialize the sentiment-volume strategy
        
        Args:
            config (dict): Strategy configuration
            sentiment_analyzer: Sentiment analysis module
            volume_analyzer: Volume analysis module
        """
        self.logger = logging.getLogger("sentiment_volume_strategy")
        self.config = config
        self.sentiment = sentiment_analyzer
        self.volume = volume_analyzer
        
        # Configuration parameters with defaults
        self.min_sentiment_score = config.get("min_sentiment_score", 0.5)
        self.min_volume_increase = config.get("min_volume_increase", 1.3)  # 30% increase
        self.lookback_periods = config.get("lookback_periods", 12)
        self.preferred_timeframe = config.get("preferred_timeframe", "1h")
        self.profit_target = config.get("profit_target", 0.02)  # 2%
        self.stop_loss_multiplier = config.get("stop_loss_multiplier", 1.5)
        
        self.logger.info("Sentiment-volume strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a sentiment-volume entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get sentiment data
        sentiment_data = self.sentiment.get_sentiment(symbol)
        sentiment_score = sentiment_data.get("score", 0)
        recent_change = sentiment_data.get("recent_change", 0)
        
        # Get volume data
        volume_increase = self.volume.get_volume_increase(symbol)
        
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < self.lookback_periods:
            return None
        
        # Current price
        current_price = candles["close"].iloc[-1]
        
        # Check if we have a significant sentiment score (positive or negative)
        has_significant_sentiment = abs(sentiment_score) >= self.min_sentiment_score
        
        # Check if volume is increasing
        has_volume_increase = volume_increase >= self.min_volume_increase
        
        # Track significant news recently
        has_recent_news = len(sentiment_data.get("latest_news", [])) > 0
        
        # Check for anomalous volume
        has_volume_anomaly = self.volume.has_volume_anomaly(symbol)
        
        # Entry logic for positive sentiment + volume increase (long)
        if (sentiment_score > self.min_sentiment_score and 
            (has_volume_increase or has_volume_anomaly) and
            has_recent_news):
            
            # Calculate stop loss based on recent volatility
            atr = self._calculate_atr(candles)
            stop_loss = current_price - (atr * self.stop_loss_multiplier)
            target_price = current_price * (1 + self.profit_target)
            
            self.logger.info(f"Bullish sentiment-volume signal for {symbol} at {current_price}, sentiment: {sentiment_score:.2f}, volume increase: {volume_increase:.2f}x")
            
            return {
                "direction": "long",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "sentiment_score": sentiment_score,
                "volume_increase": volume_increase,
                "news_count": len(sentiment_data.get("latest_news", []))
            }
        
        # Entry logic for negative sentiment + volume increase (short)
        elif (sentiment_score < -self.min_sentiment_score and 
              (has_volume_increase or has_volume_anomaly) and
              has_recent_news):
            
            # Calculate stop loss based on recent volatility
            atr = self._calculate_atr(candles)
            stop_loss = current_price + (atr * self.stop_loss_multiplier)
            target_price = current_price * (1 - self.profit_target)
            
            self.logger.info(f"Bearish sentiment-volume signal for {symbol} at {current_price}, sentiment: {sentiment_score:.2f}, volume increase: {volume_increase:.2f}x")
            
            return {
                "direction": "short",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "sentiment_score": sentiment_score,
                "volume_increase": volume_increase,
                "news_count": len(sentiment_data.get("latest_news", []))
            }
        
        return None
    
    def should_exit(self, symbol, direction, position, market_data):
        """
        Check if we should exit an existing position
        
        Args:
            symbol (str): Symbol to check
            direction (str): Position direction ("long" or "short")
            position (dict): Position details
            market_data (dict): Market data
            
        Returns:
            bool: True if we should exit, False otherwise
        """
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < 5:
            return False
        
        # Get current price
        current_price = candles["close"].iloc[-1]
        
        # Get sentiment data
        sentiment_data = self.sentiment.get_sentiment(symbol)
        sentiment_score = sentiment_data.get("score", 0)
        
        # Get target price from position
        target_price = position.get("target_price")
        entry_price = position.get("entry_price", current_price)
        entry_sentiment = position.get("sentiment_score", 0)
        
        # Exit if target reached
        if direction == "long" and target_price and current_price >= target_price:
            self.logger.info(f"Target reached for {symbol} long at {current_price}, target: {target_price:.5f}")
            return True
        elif direction == "short" and target_price and current_price <= target_price:
            self.logger.info(f"Target reached for {symbol} short at {current_price}, target: {target_price:.5f}")
            return True
        
        # Exit if sentiment shifts significantly
        if direction == "long" and sentiment_score < 0:
            # Sentiment has turned negative
            self.logger.info(f"Sentiment turned negative for {symbol} long at {current_price}, sentiment: {sentiment_score:.2f}")
            return True
        elif direction == "short" and sentiment_score > 0:
            # Sentiment has turned positive
            self.logger.info(f"Sentiment turned positive for {symbol} short at {current_price}, sentiment: {sentiment_score:.2f}")
            return True
        
        # Exit if significant price move against position
        if self._check_adverse_price_action(candles, direction):
            self.logger.info(f"Adverse price action for {symbol} {direction} at {current_price}")
            return True
        
        return False
    
    def _get_candles(self, symbol, market_data):
        """Get candle data for the symbol"""
        if symbol not in market_data:
            return None
            
        # Select the preferred timeframe
        if self.preferred_timeframe in market_data[symbol]:
            return market_data[symbol][self.preferred_timeframe]
        
        # Fall back to any available timeframe
        for tf in market_data[symbol]:
            return market_data[symbol][tf]
            
        return None
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if not all(col in df.columns for col in ["high", "low", "close"]) or len(df) < period:
            return df["close"].iloc[-1] * 0.01  # Default to 1% of current price
            
        try:
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            
            # Calculate true range
            tr1 = abs(high[1:] - low[1:])
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            
            # Calculate ATR
            atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return df["close"].iloc[-1] * 0.01  # Default to 1% of current price
    
    def _check_adverse_price_action(self, candles, direction):
        """
        Check for price action that goes against our position
        
        Args:
            candles (pd.DataFrame): OHLCV data
            direction (str): Position direction ("long" or "short")
            
        Returns:
            bool: True if adverse price action is detected
        """
        if len(candles) < 3:
            return False
            
        # For long positions, check for bearish price action
        if direction == "long":
            # Check for bearish engulfing pattern
            if (candles["close"].iloc[-1] < candles["open"].iloc[-1] and  # Current candle is bearish
                candles["close"].iloc[-1] < candles["open"].iloc[-2] and  # Closes below prior open
                candles["open"].iloc[-1] > candles["close"].iloc[-2]):    # Opens above prior close
                return True
                
            # Check for price closing below recent lows
            recent_low = candles["low"].iloc[-5:-1].min()
            if candles["close"].iloc[-1] < recent_low:
                return True
        
        # For short positions, check for bullish price action
        elif direction == "short":
            # Check for bullish engulfing pattern
            if (candles["close"].iloc[-1] > candles["open"].iloc[-1] and  # Current candle is bullish
                candles["close"].iloc[-1] > candles["open"].iloc[-2] and  # Closes above prior open
                candles["open"].iloc[-1] < candles["close"].iloc[-2]):    # Opens below prior close
                return True
                
            # Check for price closing above recent highs
            recent_high = candles["high"].iloc[-5:-1].max()
            if candles["close"].iloc[-1] > recent_high:
                return True
        
        return False
