"""
Breakdown Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import talib as ta

class BreakdownStrategy:
    """
    Strategy that detects and trades breakdown patterns
    """
    
    def __init__(self, config, technical_analyzer, volume_analyzer):
        """
        Initialize the breakdown strategy
        
        Args:
            config (dict): Strategy configuration
            technical_analyzer: Technical analysis module
            volume_analyzer: Volume analysis module
        """
        self.logger = logging.getLogger("breakdown_strategy")
        self.config = config
        self.technical = technical_analyzer
        self.volume = volume_analyzer
        
        # Configuration parameters with defaults
        self.lookback_periods = config.get("lookback_periods", 20)
        self.support_threshold = config.get("support_threshold", 0.005)  # 0.5%
        self.confirmation_candles = config.get("confirmation_candles", 2)
        self.volume_increase_threshold = config.get("volume_increase_threshold", 1.5)  # 150%
        self.min_distance_from_support = config.get("min_distance_from_support", 0.003)  # 0.3%
        self.max_distance_from_support = config.get("max_distance_from_support", 0.02)  # 2%
        
        self.logger.info("Breakdown strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a breakdown entry signal is present
        
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
        
        # Find support levels
        support_levels = self._find_support_levels(candles)
        
        if not support_levels:
            return None
        
        # Check if price is breaking down
        current_price = candles["close"].iloc[-1]
        current_low = candles["low"].iloc[-1]
        
        # Volume confirmation
        volume_increasing = self._check_volume_confirmation(symbol, candles)
        
        # Check for breakdown
        for level in support_levels:
            # Price must be close to support level (within max_distance)
            distance_pct = (level - current_low) / level
            
            if distance_pct > self.min_distance_from_support and distance_pct < self.max_distance_from_support:
                # We need confirmation (price closed below support)
                if current_price < level:
                    # Check for additional confirmations
                    if self._confirm_breakdown(candles, level) and volume_increasing:
                        self.logger.info(f"Breakdown detected for {symbol} at {current_price}, support: {level}")
                        
                        return {
                            "direction": "short",
                            "entry_price": current_price,
                            "support_level": level,
                            "strength": self._calculate_breakdown_strength(candles, level)
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
        # We only manage short positions
        if direction != "short":
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
        profit_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
        
        # Exit if momentum is weakening and we've made some profit
        if momentum_weakening and profit_pct > self.config.get("min_profit_to_exit", 0.01):
            self.logger.info(f"Exiting breakdown trade for {symbol} at {current_price}, momentum weakening")
            return True
        
        # Exit if price retraces back above the breakdown level
        breakdown_level = position.get("support_level", 0)
        if breakdown_level > 0 and current_price > breakdown_level * 1.005:  # 0.5% above breakdown level
            self.logger.info(f"Exiting breakdown trade for {symbol} at {current_price}, failed breakdown")
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
    
    def _find_support_levels(self, candles):
        """
        Find potential support levels
        
        Args:
            candles (pd.DataFrame): OHLCV data
            
        Returns:
            list: List of support levels
        """
        # Use recent low points as potential support
        # We'll use argrelextrema to find local minima
        low_series = candles["low"].values
        
        # Find local minima
        min_idx = argrelextrema(low_series, np.less_equal, order=self.config.get("order", 5))[0]
        
        if len(min_idx) == 0:
            return []
        
        # Get the low values at these minima
        lows = [low_series[i] for i in min_idx]
        
        # Filter to keep only significant support levels
        # We'll group nearby levels
        support_levels = []
        
        # Sort lows in ascending order
        sorted_lows = sorted(lows)
        
        for low in sorted_lows:
            # Check if this level is close to an existing level
            if not any(abs(low - level) / level < self.support_threshold for level in support_levels):
                support_levels.append(low)
        
        return support_levels
    
    def _confirm_breakdown(self, candles, level):
        """
        Confirm if the breakdown is valid
        
        Args:
            candles (pd.DataFrame): OHLCV data
            level (float): Support level
            
        Returns:
            bool: True if breakdown is confirmed, False otherwise
        """
        # Check if price has closed below support for confirmation_candles consecutive periods
        for i in range(1, self.confirmation_candles + 1):
            idx = -i
            if idx < -len(candles):
                return False
                
            if candles["close"].iloc[idx] >= level:
                return False
        
        return True
    
    def _check_volume_confirmation(self, symbol, candles):
        """
        Check if volume is increasing during the breakdown
        
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
        Check if bearish momentum is weakening
        
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
        price_making_lower_low = candles["low"].iloc[-1] < candles["low"].iloc[-2]
        rsi_making_higher_low = rsi.iloc[-1] > rsi.iloc[-2]
        
        # Bullish divergence (price down, RSI up)
        if price_making_lower_low and rsi_making_higher_low:
            return True
            
        # Check for oversold conditions
        if rsi.iloc[-1] < 30:
            return True
            
        return False
    
    def _calculate_breakdown_strength(self, candles, level):
        """
        Calculate the strength of the breakdown
        
        Args:
            candles (pd.DataFrame): OHLCV data
            level (float): Support level
            
        Returns:
            float: Breakdown strength score (0-1)
        """
        # Factors to consider:
        # 1. Distance of breakdown (% below support)
        # 2. Volume increase
        # 3. Number of times the level was tested before
        # 4. Duration of the consolidation before breakdown
        
        current_price = candles["close"].iloc[-1]
        
        # 1. Distance below support
        breakdown_distance = (level - current_price) / level
        distance_score = min(breakdown_distance / 0.03, 1.0)  # Normalize to max 3%
        
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
            if abs(candles["low"].iloc[i] - level) / level < 0.005:  # Within 0.5%
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
