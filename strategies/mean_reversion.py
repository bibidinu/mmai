"""
Mean Reversion Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd
import talib as ta

class MeanReversionStrategy:
    """
    Strategy that trades reversion to the mean after price extremes
    """
    
    def __init__(self, config, technical_analyzer):
        """
        Initialize the mean reversion strategy
        
        Args:
            config (dict): Strategy configuration
            technical_analyzer: Technical analysis module
        """
        self.logger = logging.getLogger("mean_reversion_strategy")
        self.config = config
        self.technical = technical_analyzer
        
        # Configuration parameters with defaults
        self.overbought_threshold = config.get("overbought_threshold", 70)
        self.oversold_threshold = config.get("oversold_threshold", 30)
        self.preferred_timeframe = config.get("preferred_timeframe", "1h")
        self.rsi_period = config.get("rsi_period", 14)
        self.exit_middle_band = config.get("exit_middle_band", True)
        self.profit_target = config.get("profit_target", 0.02)  # 2%
        self.stop_loss_multiplier = config.get("stop_loss_multiplier", 1.5)
        
        self.logger.info("Mean reversion strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a mean reversion entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < self.rsi_period + 5:
            return None
        
        # Calculate RSI
        rsi = self._calculate_rsi(candles)
        
        if rsi is None or len(rsi) < 2:
            return None
        
        current_rsi = rsi[-1]
        previous_rsi = rsi[-2]
        
        # Calculate Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(candles)
        
        if upper is None or middle is None or lower is None:
            return None
        
        current_price = candles["close"].iloc[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]
        
        # Entry logic for oversold conditions (long)
        if current_rsi < self.oversold_threshold and previous_rsi <= current_rsi:
            # Additional confirmation: price near lower Bollinger Band
            price_to_lower_ratio = (current_price - current_lower) / current_lower
            
            if price_to_lower_ratio < 0.01:  # Within 1% of lower band
                # Determine entry price
                entry_price = current_price
                
                # Calculate stop loss using ATR
                atr = self._calculate_atr(candles)
                if atr is not None and len(atr) > 0:
                    stop_loss = entry_price - (atr[-1] * self.stop_loss_multiplier)
                else:
                    # Fallback: use a percentage of entry price
                    stop_loss = entry_price * 0.97  # 3% below entry
                
                self.logger.info(f"Mean reversion long signal for {symbol} at {entry_price}, RSI: {current_rsi:.1f}")
                
                return {
                    "direction": "long",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target_price": current_middle if self.exit_middle_band else entry_price * (1 + self.profit_target),
                    "reason": "oversold_rsi"
                }
        
        # Entry logic for overbought conditions (short)
        if current_rsi > self.overbought_threshold and previous_rsi >= current_rsi:
            # Additional confirmation: price near upper Bollinger Band
            price_to_upper_ratio = (current_upper - current_price) / current_upper
            
            if price_to_upper_ratio < 0.01:  # Within 1% of upper band
                # Determine entry price
                entry_price = current_price
                
                # Calculate stop loss using ATR
                atr = self._calculate_atr(candles)
                if atr is not None and len(atr) > 0:
                    stop_loss = entry_price + (atr[-1] * self.stop_loss_multiplier)
                else:
                    # Fallback: use a percentage of entry price
                    stop_loss = entry_price * 1.03  # 3% above entry
                
                self.logger.info(f"Mean reversion short signal for {symbol} at {entry_price}, RSI: {current_rsi:.1f}")
                
                return {
                    "direction": "short",
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "target_price": current_middle if self.exit_middle_band else entry_price * (1 - self.profit_target),
                    "reason": "overbought_rsi"
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
        
        if candles is None or len(candles) < 20:  # Need enough data for BBands
            return False
        
        # Calculate RSI
        rsi = self._calculate_rsi(candles)
        
        if rsi is None or len(rsi) < 1:
            return False
        
        current_rsi = rsi[-1]
        
        # Calculate Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(candles)
        
        if upper is None or middle is None or lower is None:
            return False
        
        current_price = candles["close"].iloc[-1]
        current_middle = middle[-1]
        
        # Get target price from position if available
        target_price = position.get("target_price", None)
        
        if self.exit_middle_band:
            # Use middle band as target if exit_middle_band is enabled
            target_price = current_middle
        
        # Exit logic for long positions
        if direction == "long":
            # Exit if RSI moves into overbought territory
            if current_rsi > self.overbought_threshold:
                self.logger.info(f"Exiting long position for {symbol} at {current_price}, RSI overbought: {current_rsi:.1f}")
                return True
            
            # Exit if price reaches middle band (mean)
            if self.exit_middle_band and current_price >= current_middle:
                self.logger.info(f"Exiting long position for {symbol} at {current_price}, reached middle band: {current_middle:.5f}")
                return True
            
            # Exit if price reaches target
            if target_price is not None and current_price >= target_price:
                self.logger.info(f"Exiting long position for {symbol} at {current_price}, reached target: {target_price:.5f}")
                return True
        
        # Exit logic for short positions
        elif direction == "short":
            # Exit if RSI moves into oversold territory
            if current_rsi < self.oversold_threshold:
                self.logger.info(f"Exiting short position for {symbol} at {current_price}, RSI oversold: {current_rsi:.1f}")
                return True
            
            # Exit if price reaches middle band (mean)
            if self.exit_middle_band and current_price <= current_middle:
                self.logger.info(f"Exiting short position for {symbol} at {current_price}, reached middle band: {current_middle:.5f}")
                return True
            
            # Exit if price reaches target
            if target_price is not None and current_price <= target_price:
                self.logger.info(f"Exiting short position for {symbol} at {current_price}, reached target: {target_price:.5f}")
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
    
    def _calculate_rsi(self, df, period=None):
        """Calculate Relative Strength Index"""
        if period is None:
            period = self.rsi_period
            
        if "close" not in df.columns or len(df) < period:
            return None
            
        try:
            return ta.RSI(df["close"].values, timeperiod=period)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return None
    
    def _calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
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
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if not all(col in df.columns for col in ["high", "low", "close"]) or len(df) < period:
            return None
            
        try:
            return ta.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=period
            )
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return None
