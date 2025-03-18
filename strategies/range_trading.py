"""
Range Trading Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd
import talib as ta

class RangeStrategy:
    """
    Strategy that trades within established price ranges
    """
    
    def __init__(self, config, technical_analyzer):
        """
        Initialize the range trading strategy
        
        Args:
            config (dict): Strategy configuration
            technical_analyzer: Technical analysis module
        """
        self.logger = logging.getLogger("range_strategy")
        self.config = config
        self.technical = technical_analyzer
        
        # Configuration parameters with defaults
        self.min_range_periods = config.get("min_range_periods", 20)
        self.max_range_width = config.get("max_range_width", 0.05)  # 5% range
        self.entry_distance = config.get("entry_distance", 0.01)    # Enter 1% from range edge
        self.preferred_timeframe = config.get("preferred_timeframe", "4h")
        self.exit_opposite_band = config.get("exit_opposite_band", True)
        
        # Bollinger Bands parameters
        self.use_bollinger_bands = config.get("use_bollinger_bands", True)
        self.bollinger_period = config.get("bollinger_period", 20)
        self.bollinger_std = config.get("bollinger_std", 2.0)
        
        self.logger.info("Range trading strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a range trading entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < self.min_range_periods:
            return None
        
        # Check if the market is in a range
        if not self._is_ranging(symbol, candles):
            return None
        
        # Detect range boundaries
        if self.use_bollinger_bands:
            upper, middle, lower = self._calculate_bollinger_bands(candles)
            
            if upper is None or middle is None or lower is None:
                return None
                
            range_high = upper[-1]
            range_low = lower[-1]
            range_center = middle[-1]
        else:
            # Use recent high/low as range
            recent_candles = candles.tail(self.min_range_periods)
            range_high = recent_candles["high"].max()
            range_low = recent_candles["low"].min()
            range_center = (range_high + range_low) / 2
        
        # Get current price
        current_price = candles["close"].iloc[-1]
        range_width = (range_high - range_low) / range_center
        
        # Check if range is tight enough
        if range_width > self.max_range_width:
            return None
        
        # Entry logic
        entry_signal = None
        
        # Near lower band - long opportunity
        lower_entry = range_low * (1 + self.entry_distance)
        if current_price <= lower_entry:
            # Calculate stop loss (below range low)
            stop_loss = range_low * 0.99
            
            # Calculate target (range center or upper band)
            target = range_center if not self.exit_opposite_band else range_high
            
            entry_signal = {
                "direction": "long",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target,
                "range_low": range_low,
                "range_high": range_high,
                "range_width": range_width
            }
            
            self.logger.info(f"Range trading long signal for {symbol} at {current_price}, range: {range_low:.5f}-{range_high:.5f}")
        
        # Near upper band - short opportunity
        upper_entry = range_high * (1 - self.entry_distance)
        if current_price >= upper_entry:
            # Calculate stop loss (above range high)
            stop_loss = range_high * 1.01
            
            # Calculate target (range center or lower band)
            target = range_center if not self.exit_opposite_band else range_low
            
            entry_signal = {
                "direction": "short",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target,
                "range_low": range_low,
                "range_high": range_high,
                "range_width": range_width
            }
            
            self.logger.info(f"Range trading short signal for {symbol} at {current_price}, range: {range_low:.5f}-{range_high:.5f}")
        
        return entry_signal
    
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
        
        if candles is None or len(candles) < self.bollinger_period:
            return False
        
        # Get current price
        current_price = candles["close"].iloc[-1]
        
        # Get target price from position
        target_price = position.get("target_price")
        range_high = position.get("range_high")
        range_low = position.get("range_low")
        
        # Check if we're still ranging
        is_ranging = self._is_ranging(symbol, candles)
        
        # Exit logic for long positions
        if direction == "long":
            # Exit if price reaches target
            if target_price and current_price >= target_price:
                self.logger.info(f"Range target reached for {symbol} long at {current_price}, target: {target_price:.5f}")
                return True
            
            # Exit if range is broken (price moves below range)
            if range_low and current_price < range_low * 0.98:
                self.logger.info(f"Range broken for {symbol} long at {current_price}, range low: {range_low:.5f}")
                return True
            
            # Exit if we're no longer in a range
            if not is_ranging:
                self.logger.info(f"No longer ranging for {symbol} long at {current_price}")
                return True
        
        # Exit logic for short positions
        elif direction == "short":
            # Exit if price reaches target
            if target_price and current_price <= target_price:
                self.logger.info(f"Range target reached for {symbol} short at {current_price}, target: {target_price:.5f}")
                return True
            
            # Exit if range is broken (price moves above range)
            if range_high and current_price > range_high * 1.02:
                self.logger.info(f"Range broken for {symbol} short at {current_price}, range high: {range_high:.5f}")
                return True
            
            # Exit if we're no longer in a range
            if not is_ranging:
                self.logger.info(f"No longer ranging for {symbol} short at {current_price}")
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
    
    def _is_ranging(self, symbol, candles):
        """
        Check if the market is in a range
        
        Args:
            symbol (str): Symbol to check
            candles (pd.DataFrame): OHLCV data
            
        Returns:
            bool: True if ranging, False otherwise
        """
        # Check if the technical analyzer says it's ranging
        is_ranging = self.technical.is_ranging(symbol)
        
        # If technical analyzer provides a result, use it
        if is_ranging is not None:
            return is_ranging
        
        # Otherwise, perform our own check
        if len(candles) < self.min_range_periods:
            return False
        
        # Calculate Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(candles)
        
        if upper is None or middle is None or lower is None:
            return False
        
        # Check if bandwidth is narrow
        bandwidth = (upper[-1] - lower[-1]) / middle[-1]
        
        # Check for flat moving average (middle band)
        ma_slope = abs((middle[-1] - middle[-5]) / middle[-5]) if len(middle) >= 5 else 1
        
        # Consider it ranging if bandwidth is within range and MA is relatively flat
        return bandwidth <= self.max_range_width and ma_slope < 0.01
    
    def _calculate_bollinger_bands(self, df, period=None, std_dev=None):
        """Calculate Bollinger Bands"""
        if period is None:
            period = self.bollinger_period
            
        if std_dev is None:
            std_dev = self.bollinger_std
            
        if "close" not in df.columns or len(df) < period:
            return None, None, None
            
        try:
            upper, middle, lower = ta.BBANDS(
                df["close"].values,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev
            )
            return upper, middle, lower
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return None, None, None
