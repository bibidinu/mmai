"""
Trend Following Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd
import talib as ta

class TrendFollowingStrategy:
    """
    Strategy that follows established market trends
    """
    
    def __init__(self, config, technical_analyzer):
        """
        Initialize the trend following strategy
        
        Args:
            config (dict): Strategy configuration
            technical_analyzer: Technical analysis module
        """
        self.logger = logging.getLogger("trend_following_strategy")
        self.config = config
        self.technical = technical_analyzer
        
        # Configuration parameters with defaults
        self.ema_short = config.get("ema_short", 8)
        self.ema_long = config.get("ema_long", 21)
        self.preferred_timeframe = config.get("preferred_timeframe", "1h")
        self.confirmation_periods = config.get("confirmation_periods", 3)
        self.exit_reversal_bars = config.get("exit_reversal_bars", 2)
        self.use_adx = config.get("use_adx", True)
        self.adx_threshold = config.get("adx_threshold", 25)
        
        self.logger.info("Trend following strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a trend entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < max(self.ema_short, self.ema_long) + self.confirmation_periods:
            return None
        
        # Calculate EMAs
        ema_short = self._calculate_ema(candles, self.ema_short)
        ema_long = self._calculate_ema(candles, self.ema_long)
        
        if ema_short is None or ema_long is None:
            return None
        
        # Calculate ADX if enabled
        adx_value = None
        if self.use_adx:
            adx_value = self._calculate_adx(candles)
            
            if adx_value is None or adx_value[-1] < self.adx_threshold:
                return None
        
        # Check for trend based on EMA crossover
        current_short = ema_short[-1]
        current_long = ema_long[-1]
        
        # Check previous bars for confirmation
        trend_confirmed = True
        for i in range(1, self.confirmation_periods + 1):
            if i >= len(ema_short) or i >= len(ema_long):
                trend_confirmed = False
                break
                
            if ema_short[-i-1] <= ema_long[-i-1]:
                trend_confirmed = False
                break
        
        # Entry logic for uptrend
        if current_short > current_long and trend_confirmed:
            # Determine entry price
            entry_price = candles["close"].iloc[-1]
            
            # Calculate trend strength
            trend_strength = (current_short - current_long) / current_long
            
            self.logger.info(f"Trend following long signal for {symbol} at {entry_price}, strength: {trend_strength:.4f}")
            
            return {
                "direction": "long",
                "entry_price": entry_price,
                "strength": trend_strength
            }
        
        # Check for downtrend
        trend_confirmed = True
        for i in range(1, self.confirmation_periods + 1):
            if i >= len(ema_short) or i >= len(ema_long):
                trend_confirmed = False
                break
                
            if ema_short[-i-1] >= ema_long[-i-1]:
                trend_confirmed = False
                break
        
        # Entry logic for downtrend
        if current_short < current_long and trend_confirmed:
            # Determine entry price
            entry_price = candles["close"].iloc[-1]
            
            # Calculate trend strength
            trend_strength = (current_long - current_short) / current_long
            
            self.logger.info(f"Trend following short signal for {symbol} at {entry_price}, strength: {trend_strength:.4f}")
            
            return {
                "direction": "short",
                "entry_price": entry_price,
                "strength": trend_strength
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
        
        if candles is None or len(candles) < max(self.ema_short, self.ema_long) + self.exit_reversal_bars:
            return False
        
        # Calculate EMAs
        ema_short = self._calculate_ema(candles, self.ema_short)
        ema_long = self._calculate_ema(candles, self.ema_long)
        
        if ema_short is None or ema_long is None:
            return False
        
        # Check for trend reversal
        reversal_count = 0
        
        if direction == "long":
            # Check for bearish crossover (short EMA crosses below long EMA)
            for i in range(self.exit_reversal_bars):
                if i >= len(ema_short) or i >= len(ema_long):
                    break
                    
                if ema_short[-i-1] < ema_long[-i-1]:
                    reversal_count += 1
        else:  # short
            # Check for bullish crossover (short EMA crosses above long EMA)
            for i in range(self.exit_reversal_bars):
                if i >= len(ema_short) or i >= len(ema_long):
                    break
                    
                if ema_short[-i-1] > ema_long[-i-1]:
                    reversal_count += 1
        
        # Exit if we have enough reversal bars
        if reversal_count >= self.exit_reversal_bars:
            self.logger.info(f"Trend reversal exit signal for {symbol} {direction}")
            return True
        
        # Also check ADX for weakening trend
        if self.use_adx:
            adx_value = self._calculate_adx(candles)
            
            if adx_value is not None:
                current_adx = adx_value[-1]
                previous_adx = adx_value[-2] if len(adx_value) > 1 else current_adx
                
                # Exit if ADX is falling below threshold
                if previous_adx > self.adx_threshold and current_adx < self.adx_threshold:
                    self.logger.info(f"ADX weakening exit signal for {symbol} {direction}: {current_adx:.1f}")
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
    
    def _calculate_ema(self, df, period):
        """Calculate Exponential Moving Average"""
        if "close" not in df.columns or len(df) < period:
            return None
            
        try:
            return ta.EMA(df["close"].values, timeperiod=period)
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return None
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        if not all(col in df.columns for col in ["high", "low", "close"]) or len(df) < period:
            return None
            
        try:
            return ta.ADX(df["high"].values, df["low"].values, df["close"].values, timeperiod=period)
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {str(e)}")
            return None
