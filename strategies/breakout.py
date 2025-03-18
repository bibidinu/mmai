"""
Breakout Strategy Implementation
"""
import logging
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd

class BreakoutStrategy:
    """
    Strategy that detects and trades breakout patterns
    """
    
    def __init__(self, config, technical_analyzer, volume_analyzer):
        """
        Initialize the breakout strategy
        
        Args:
            config (dict): Strategy configuration
            technical_analyzer: Technical analysis module
            volume_analyzer: Volume analysis module
        """
        self.logger = logging.getLogger("breakout_strategy")
        self.config = config
        self.technical = technical_analyzer
        self.volume = volume_analyzer
        
        # Configuration parameters with defaults
        self.lookback_periods = config.get("lookback_periods", 20)
        self.resistance_threshold = config.get("resistance_threshold", 0.005)  # 0.5%
        self.confirmation_candles = config.get("confirmation_candles", 2)
        self.volume_increase_threshold = config.get("volume_increase_threshold", 1.5)  # 150%
        self.min_distance_from_resistance = config.get("min_distance_from_resistance", 0.003)  # 0.3%
        self.max_distance_from_resistance = config.get("max_distance_from_resistance", 0.02)  # 2%
        
        self.logger.info("Breakout strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a breakout entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < self.lookback_periods:
            return None
        
        # Find resistance levels
        resistance_levels = self._find_resistance_levels(candles)
        
        if not resistance_levels:
            return None
        
        # Check if price is breaking out
        current_price = candles["close"].iloc[-1]
        current_high = candles["high"].iloc[-1]
        
        # Volume confirmation
        volume_increasing = self._check_volume_confirmation(symbol, candles)
        
        # Check for breakout
        for level in resistance_levels:
            # Price must be close to resistance level (within max_distance)
            distance_pct = (current_high - level) / level
            
            if distance_pct > self.min_distance_from_resistance and distance_pct < self.max_distance_from_resistance:
                # We need confirmation (price closed above resistance)
                if current_price > level:
                    # Check for additional confirmations
                    if self._confirm_breakout(candles, level) and volume_increasing:
                        self.logger.info(f"Breakout detected for {symbol} at {current_price}, resistance: {level}")
                        
                        return {
                            "direction": "long",
                            "entry_price": current_price,
                            "resistance_level": level,
                            "strength": self._calculate_breakout_strength(candles, level)
                        }
        
        return None
    
    def should_exit(self, symbol, direction, position, market_data):
        """
        Check if we should exit an existing position
        
        Args:
            symbol (str): Symbol to check
            direction (str): Position direction
            position (dict): Position details
            market_data (dict): Market data
            
        Returns:
            bool: True if we should exit, False otherwise
        """
        # We only manage long positions
        if direction != "long":
            return False
        
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < 3:
            return False
        
        # Check if momentum is weakening
        momentum_weakening = self._check_momentum_weakening(candles)
        
        # Get entry price
        entry_price = position.get("entry_price", 0)
        current_price = candles["close"].iloc[-1]
        
        # Calculate profit percentage
        profit_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # Exit if momentum is weakening and we've made some profit
        if momentum_weakening and profit_pct > self.config.get("min_profit_to_exit", 0.01):
            self.logger.info(f"Exiting breakout trade for {symbol} at {current_price}, momentum weakening")
            return True
        
        # Exit if price retraces back below the breakout level
        breakout_level = position.get("resistance_level", 0)
        if breakout_level > 0 and current_price < breakout_level * 0.995:  # 0.5% below breakout level
            self.logger.info(f"Exiting breakout trade for {symbol} at {current_price}, failed breakout")
            return True
        
        return False
    
    def _get_candles(self, symbol, market_data):
        """Get candle data for the symbol"""
        if symbol not in market_data:
            return None
            
        # Select the preferred timeframe
        preferred_tf = self.config.get("preferred_timeframe", "1h")
        
        if preferred_tf in market_data[symbol]:
            return market_data[symbol][preferred_tf]
        
        # Fall back to any available timeframe
        for tf in market_data[symbol]:
            return market_data[symbol][tf]
            
        return None
    
    def _find_resistance_levels(self, candles):
        """
        Find potential resistance levels
        
        Args:
            candles (pd.DataFrame): OHLCV data
            
        Returns:
            list: List of resistance levels
        """
        # Use recent high points as potential resistance
        # We'll use argrelextrema to find local maxima
        high_series = candles["high"].values
        
        # Find local maxima
        max_idx = argrelextrema(high_series, np.greater_equal, order=self.config.get("order", 5))[0]
        
        if len(max_idx) == 0:
            return []
        
        # Get the high values at these maxima
        highs = [high_series[i] for i in max_idx]
        
        # Filter to keep only significant resistance levels
        # We'll group nearby levels
        resistance_levels = []
        
        # Sort highs in descending order
        sorted_highs = sorted(highs, reverse=True)
        
        for high in sorted_highs:
            # Check if this level is close to an existing level
            if not any(abs(high - level) / level < self.resistance_threshold for level in resistance_levels):
                resistance_levels.append(high)
        
        return resistance_levels
    
    def _confirm_breakout(self, candles, level):
        """
        Confirm if the breakout is valid
        
        Args:
            candles (pd.DataFrame): OHLCV data
            level (float): Resistance level
            
        Returns:
            bool: True if breakout is confirmed, False otherwise
        """
        # Check if price has closed above resistance for confirmation_candles consecutive periods
        for i in range(1, self.confirmation_candles + 1):
            idx = -i
            if idx < -len(candles):
                return False
                
            if candles["close"].iloc[idx] <= level:
                return False
        
        return True
    
    def _check_volume_confirmation(self, symbol, candles):
        """
        Check if volume is increasing during the breakout
        
        Args:
            symbol (str): Symbol to check
            candles (pd.DataFrame): OHLCV data
            
        Returns:
            bool: True if volume is increasing, False otherwise
        """
        if "volume" not in candles.columns:
            return False
            
        # Calculate average volume over recent periods
        recent_volume = candles["volume"].iloc[-1]
        avg_volume = candles["volume"].iloc[-self.lookback_periods:-1].mean()
        
        # Check if current volume is significantly higher than average
        return recent_volume > avg_volume * self.volume_increase_threshold
    
    def _check_momentum_weakening(self, candles):
        """
        Check if bullish momentum is weakening
        
        Args:
            candles (pd.DataFrame): OHLCV data
            
        Returns:
            bool: True if momentum is weakening, False otherwise
        """
        # Calculate RSI or other momentum indicators
        rsi = self.technical.calculate_rsi(candles, 14)
        
        if rsi is None:
            return False
        
        # Check for RSI divergence
        price_making_higher_high = candles["high"].iloc[-1] > candles["high"].iloc[-2]
        rsi_making_lower_high = rsi.iloc[-1] < rsi.iloc[-2]
        
        # Bearish divergence (price up, RSI down)
        if price_making_higher_high and rsi_making_lower_high:
            return True
            
        # Check for overbought conditions
        if rsi.iloc[-1] > 70:
            return True
            
        return False
    
    def _calculate_breakout_strength(self, candles, level):
        """
        Calculate the strength of the breakout
        
        Args:
            candles (pd.DataFrame): OHLCV data
            level (float): Resistance level
            
        Returns:
            float: Breakout strength score (0-1)
        """
        # Factors to consider:
        # 1. Distance of breakout (% above resistance)
        # 2. Volume increase
        # 3. Number of times the level was tested before
        # 4. Duration of the consolidation before breakout
        
        current_price = candles["close"].iloc[-1]
        
        # 1. Distance above resistance
        breakout_distance = (current_price - level) / level
        distance_score = min(breakout_distance / 0.03, 1.0)  # Normalize to max 3%
        
        # 2. Volume increase
        volume_score = 0.5  # Default if no volume data
        if "volume" in candles.columns:
            recent_volume = candles["volume"].iloc[-1]
            avg_volume = candles["volume"].iloc[-self.lookback_periods:-1].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            volume_score = min(volume_ratio / 3.0, 1.0)  # Normalize to max 300%
        
        # 3. Level test count (simplified)
        test_count = 0
        for i in range(len(candles) - 1):
            if abs(candles["high"].iloc[i] - level) / level < 0.005:  # Within 0.5%
                test_count += 1
        
        test_score = min(test_count / 5.0, 1.0)  # Normalize to max 5 tests
        
        # Combine scores with weights
        weight_distance = 0.4
        weight_volume = 0.3
        weight_tests = 0.3
        
        strength = (weight_distance * distance_score + 
                   weight_volume * volume_score + 
                   weight_tests * test_score)
        
        return strength
