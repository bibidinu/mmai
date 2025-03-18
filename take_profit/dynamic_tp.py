"""
Dynamic Take Profit Management
"""
import logging
import numpy as np
import pandas as pd

class DynamicTakeProfitManager:
    """
    Manages dynamic take profit levels based on market conditions
    """
    
    def __init__(self, config, technical_analyzer, volatility_analyzer):
        """
        Initialize the dynamic take profit manager
        
        Args:
            config (dict): Take profit configuration
            technical_analyzer: Technical analysis module
            volatility_analyzer: Volatility analysis module
        """
        self.logger = logging.getLogger("dynamic_tp")
        self.config = config
        self.technical = technical_analyzer
        self.volatility = volatility_analyzer
        
        # Configuration parameters with defaults
        self.tp_count = config.get("tp_count", 3)  # Number of TP levels
        self.min_tp_distance = config.get("min_tp_distance", 0.005)  # Minimum 0.5% between TP levels
        self.use_fixed_percentages = config.get("use_fixed_percentages", False)
        
        # Default fixed TP percentages (if enabled)
        self.fixed_tp_percentages = config.get("fixed_tp_percentages", [0.01, 0.02, 0.035])  # 1%, 2%, 3.5%
        
        # Position distribution for partial TPs
        self.position_distribution = config.get("position_distribution", [0.3, 0.3, 0.4])  # 30%, 30%, 40%
        
        # Breakeven settings
        self.move_to_be_after_tp1 = config.get("move_to_be_after_tp1", True)  # Move to breakeven after TP1
        self.breakeven_offset = config.get("breakeven_offset", 0.001)  # 0.1% above entry for breakeven
        
        self.logger.info("Dynamic TP manager initialized")
    
    def calculate_tp_levels(self, symbol, direction, entry_price):
        """
        Calculate dynamic take profit levels
        
        Args:
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            entry_price (float): Position entry price
            
        Returns:
            list: List of take profit prices
        """
        if self.use_fixed_percentages:
            # Use fixed percentage-based TPs
            return self._calculate_fixed_percentage_tps(direction, entry_price)
        else:
            # Use dynamic TPs based on market conditions
            return self._calculate_dynamic_tps(symbol, direction, entry_price)
    
    def manage_position(self, position, exchange):
        """
        Manage an existing position, including moving to breakeven
        
        Args:
            position (dict): Position details
            exchange: Exchange interface for position updates
            
        Returns:
            bool: True if position was updated, False otherwise
        """
        symbol = position.get("symbol")
        direction = position.get("direction")
        entry_price = position.get("entry_price", 0)
        
        # Check if TP1 has been hit and position hasn't been moved to breakeven yet
        if (position.get("tp1_hit", False) and 
            not position.get("moved_to_be", False) and 
            self.move_to_be_after_tp1):
            
            # Calculate breakeven price (slightly above entry for long, below for short)
            if direction == "long":
                be_price = entry_price * (1 + self.breakeven_offset)
            else:
                be_price = entry_price * (1 - self.breakeven_offset)
            
            # Update stop loss to breakeven
            success = exchange.update_stop_loss(symbol, direction, position.get("id"), be_price)
            
            if success:
                self.logger.info(f"Moved {symbol} {direction} position to breakeven at {be_price}")
                # Mark position as moved to breakeven
                return True
        
        return False
    
    def _calculate_fixed_percentage_tps(self, direction, entry_price):
        """
        Calculate take profit levels using fixed percentages
        
        Args:
            direction (str): "long" or "short"
            entry_price (float): Position entry price
            
        Returns:
            list: List of take profit prices
        """
        tp_levels = []
        
        for percentage in self.fixed_tp_percentages[:self.tp_count]:
            if direction == "long":
                tp_price = entry_price * (1 + percentage)
            else:
                tp_price = entry_price * (1 - percentage)
                
            tp_levels.append(tp_price)
        
        return tp_levels
    
    def _calculate_dynamic_tps(self, symbol, direction, entry_price):
        """
        Calculate dynamic take profit levels based on market conditions
        
        Args:
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            entry_price (float): Position entry price
            
        Returns:
            list: List of take profit prices
        """
        # Get key levels from technical analysis
        key_levels = self.technical.get_key_levels(symbol)
        
        # Get current volatility
        volatility = self.volatility.get_volatility(symbol)
        
        # Get market momentum
        momentum = self.technical.get_momentum(symbol)
        
        # Get OHLCV data for volume profile
        ohlcv = self.technical.get_ohlcv(symbol)
        
        # 1. Base TP distances on volatility (ATR-based)
        # Higher volatility = wider TP distances
        base_distances = []
        
        # Scale factor based on volatility regime
        volatility_regime = self.volatility.get_volatility_regime(symbol)
        if volatility_regime == "low":
            vol_scale = 1.0
        elif volatility_regime == "medium":
            vol_scale = 1.5
        else:  # high
            vol_scale = 2.0
        
        atr_multipliers = self.config.get("atr_multipliers", [1.5, 2.5, 4.0])
        
        for i in range(self.tp_count):
            # Base TP on ATR
            atr_multiple = atr_multipliers[i] * vol_scale
            distance = volatility * atr_multiple
            base_distances.append(distance)
        
        # 2. Adjust based on momentum
        # Strong momentum = larger TP distances
        momentum_factor = 1.0
        if momentum > 0.7:  # Strong momentum
            momentum_factor = 1.2
        elif momentum < 0.3:  # Weak momentum
            momentum_factor = 0.8
            
        adjusted_distances = [d * momentum_factor for d in base_distances]
        
        # 3. Calculate initial TP levels
        tp_levels = []
        
        for distance in adjusted_distances:
            if direction == "long":
                tp_price = entry_price * (1 + distance)
            else:
                tp_price = entry_price * (1 - distance)
                
            tp_levels.append(tp_price)
        
        # 4. Adjust TPs to align with key levels if they're close
        final_tp_levels = []
        
        for i, tp in enumerate(tp_levels):
            # Check if there's a key level nearby
            nearest_level = self._find_nearest_key_level(key_levels, tp, direction)
            
            if nearest_level:
                # Adjust TP to align with the key level
                final_tp_levels.append(nearest_level)
            else:
                # Keep original TP
                final_tp_levels.append(tp)
        
        # 5. Ensure minimum distance between TP levels
        for i in range(1, len(final_tp_levels)):
            prev_tp = final_tp_levels[i-1]
            min_next_tp = prev_tp * (1 + self.min_tp_distance) if direction == "long" else prev_tp * (1 - self.min_tp_distance)
            
            if (direction == "long" and final_tp_levels[i] <= min_next_tp) or \
               (direction == "short" and final_tp_levels[i] >= min_next_tp):
                final_tp_levels[i] = min_next_tp
        
        return final_tp_levels
    
    def _find_nearest_key_level(self, key_levels, tp_price, direction):
        """
        Find the nearest key level to a given TP price
        
        Args:
            key_levels (dict): Dictionary with 'support' and 'resistance' lists
            tp_price (float): Take profit price
            direction (str): "long" or "short"
            
        Returns:
            float or None: Nearest key level price, or None if no suitable level found
        """
        if not key_levels:
            return None
            
        relevant_levels = []
        
        # For long positions, look at resistance levels above entry
        if direction == "long":
            relevant_levels = [level for level in key_levels.get("resistance", []) if level > tp_price * 0.98]
        # For short positions, look at support levels below entry
        else:
            relevant_levels = [level for level in key_levels.get("support", []) if level < tp_price * 1.02]
        
        if not relevant_levels:
            return None
            
        # Find the nearest level
        nearest_level = min(relevant_levels, key=lambda x: abs(x - tp_price))
        
        # Only adjust if the level is within a certain threshold of the original TP
        max_adjustment = 0.015  # Maximum 1.5% adjustment
        if abs(nearest_level - tp_price) / tp_price <= max_adjustment:
            return nearest_level
            
        return None
