"""
Volatility Analysis Module
"""
import logging
import numpy as np
import pandas as pd
import talib as ta

class VolatilityAnalyzer:
    """
    Analyzes market volatility and volatility regimes
    """
    
    def __init__(self, config):
        """
        Initialize the volatility analyzer
        
        Args:
            config (dict): Volatility analysis configuration
        """
        self.logger = logging.getLogger("volatility_analyzer")
        self.config = config
        
        # Analysis parameters
        self.atr_period = config.get("atr_period", 14)
        self.std_dev_period = config.get("std_dev_period", 20)
        self.regime_lookback = config.get("regime_lookback", 100)
        
        # Volatility thresholds
        self.high_vol_threshold = config.get("high_vol_threshold", 0.8)  # 80th percentile
        self.low_vol_threshold = config.get("low_vol_threshold", 0.2)    # 20th percentile
        
        # Contraction/expansion parameters
        self.contraction_threshold = config.get("contraction_threshold", 0.5)  # 50% of recent range
        self.expansion_threshold = config.get("expansion_threshold", 2.0)       # 200% of recent range
        
        # Data storage
        self.volatility_data = {}
        self.volatility_regimes = {}
        self.recent_changes = {}
        self.global_volatility = "medium"
        
        self.logger.info("Volatility analyzer initialized")
    
    def update(self, market_data):
        """
        Update volatility metrics based on market data
        
        Args:
            market_data (dict): Market data indexed by symbol and timeframe
        """
        # Reset data structures
        self.volatility_data = {}
        self.volatility_regimes = {}
        self.recent_changes = {}
        
        # Process each symbol
        for symbol in market_data:
            self.volatility_data[symbol] = {}
            
            # Process each timeframe
            for timeframe in market_data[symbol]:
                df = market_data[symbol][timeframe]
                
                if df is None or len(df) < max(self.atr_period, self.std_dev_period, self.regime_lookback):
                    continue
                
                # Calculate volatility metrics
                volatility_data = self._calculate_volatility_metrics(df)
                self.volatility_data[symbol][timeframe] = volatility_data
                
                # Identify volatility regime
                if timeframe == self.config.get("regime_timeframe", "4h"):
                    regime = self._identify_volatility_regime(volatility_data)
                    self.volatility_regimes[symbol] = regime
                    
                    # Check for recent changes in volatility
                    recent_change = self._check_volatility_change(volatility_data)
                    self.recent_changes[symbol] = recent_change
        
        # Calculate global volatility
        self._calculate_global_volatility()
    
    def _calculate_volatility_metrics(self, df):
        """
        Calculate various volatility metrics
        
        Args:
            df (pd.DataFrame): OHLCV data
            
        Returns:
            dict: Volatility metrics
        """
        results = {}
        
        # Calculate Average True Range (ATR)
        try:
            results["atr"] = ta.ATR(
                df["high"].values,
                df["low"].values,
                df["close"].values,
                timeperiod=self.atr_period
            )
            
            # Normalize ATR as percentage of price
            results["atr_pct"] = results["atr"] / df["close"].values * 100
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            results["atr"] = np.zeros(len(df))
            results["atr_pct"] = np.zeros(len(df))
        
        # Calculate rolling standard deviation
        try:
            results["std_dev"] = df["close"].rolling(window=self.std_dev_period).std().values
            results["std_dev_pct"] = results["std_dev"] / df["close"].values * 100
        except Exception as e:
            self.logger.error(f"Error calculating standard deviation: {str(e)}")
            results["std_dev"] = np.zeros(len(df))
            results["std_dev_pct"] = np.zeros(len(df))
        
        # Calculate Bollinger Bands width
        try:
            upper, middle, lower = ta.BBANDS(
                df["close"].values,
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2
            )
            results["bb_width"] = (upper - lower) / middle * 100
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            results["bb_width"] = np.zeros(len(df))
        
        # Calculate historical volatility (close-to-close)
        try:
            log_returns = np.log(df["close"] / df["close"].shift(1))
            results["historical_vol"] = log_returns.rolling(window=self.std_dev_period).std().values * np.sqrt(252) * 100
        except Exception as e:
            self.logger.error(f"Error calculating historical volatility: {str(e)}")
            results["historical_vol"] = np.zeros(len(df))
        
        # Detect volatility contraction/expansion
        try:
            lookback = min(self.regime_lookback, len(df) - 1)
            recent_atr = results["atr"][-lookback:]
            
            results["atr_max"] = np.max(recent_atr)
            results["atr_min"] = np.min(recent_atr)
            results["atr_mean"] = np.mean(recent_atr)
            
            # Current ATR vs range
            current_atr = results["atr"][-1]
            atr_range = results["atr_max"] - results["atr_min"]
            
            if atr_range > 0:
                # Normalized position within range (0-1)
                results["range_position"] = (current_atr - results["atr_min"]) / atr_range
            else:
                results["range_position"] = 0.5
                
            # Detect contraction/expansion
            if len(recent_atr) > 5:
                atr_5d_avg = np.mean(recent_atr[-5:])
                atr_20d_avg = np.mean(recent_atr[-20:]) if len(recent_atr) >= 20 else atr_5d_avg
                
                results["contraction"] = atr_5d_avg < atr_20d_avg * self.contraction_threshold
                results["expansion"] = atr_5d_avg > atr_20d_avg * self.expansion_threshold
            else:
                results["contraction"] = False
                results["expansion"] = False
                
        except Exception as e:
            self.logger.error(f"Error calculating volatility contraction/expansion: {str(e)}")
            results["atr_max"] = 0
            results["atr_min"] = 0
            results["atr_mean"] = 0
            results["range_position"] = 0.5
            results["contraction"] = False
            results["expansion"] = False
        
        return results
    
    def _identify_volatility_regime(self, volatility_data):
        """
        Identify volatility regime based on metrics
        
        Args:
            volatility_data (dict): Volatility metrics
            
        Returns:
            dict: Volatility regime information
        """
        if not volatility_data:
            return {"regime": "medium", "confidence": 0}
            
        # Use range position to determine regime
        range_pos = volatility_data.get("range_position", 0.5)
        
        if range_pos >= self.high_vol_threshold:
            regime = "high"
        elif range_pos <= self.low_vol_threshold:
            regime = "low"
        else:
            regime = "medium"
            
        # Calculate confidence
        if regime == "high":
            confidence = (range_pos - self.high_vol_threshold) / (1 - self.high_vol_threshold)
        elif regime == "low":
            confidence = (self.low_vol_threshold - range_pos) / self.low_vol_threshold
        else:
            # Distance from middle (0.5) normalized to 0-1
            mid_distance = abs(range_pos - 0.5) / (self.high_vol_threshold - self.low_vol_threshold)
            confidence = 1 - mid_distance
            
        confidence = max(0, min(1, confidence))
        
        # Check for contraction/expansion
        pattern = None
        if volatility_data.get("contraction", False):
            pattern = "contraction"
        elif volatility_data.get("expansion", False):
            pattern = "expansion"
            
        return {
            "regime": regime,
            "confidence": confidence,
            "range_position": range_pos,
            "pattern": pattern,
            "atr": volatility_data.get("atr", [0])[-1],
            "atr_pct": volatility_data.get("atr_pct", [0])[-1],
            "historical_vol": volatility_data.get("historical_vol", [0])[-1]
        }
    
    def _check_volatility_change(self, volatility_data):
        """
        Check for recent changes in volatility
        
        Args:
            volatility_data (dict): Volatility metrics
            
        Returns:
            dict: Volatility change information
        """
        atr = volatility_data.get("atr", [])
        
        if len(atr) < 10:
            return {"change": "none", "magnitude": 0}
            
        # Compare recent ATR to previous periods
        recent_atr = np.mean(atr[-3:])
        previous_atr = np.mean(atr[-10:-3])
        
        if previous_atr == 0:
            return {"change": "none", "magnitude": 0}
            
        change_pct = (recent_atr - previous_atr) / previous_atr
        
        # Determine change direction and magnitude
        if change_pct > 0.2:  # 20% increase
            direction = "increasing"
        elif change_pct < -0.2:  # 20% decrease
            direction = "decreasing"
        else:
            direction = "stable"
            
        return {
            "change": direction,
            "magnitude": abs(change_pct),
            "pct_change": change_pct
        }
    
    def _calculate_global_volatility(self):
        """Calculate global market volatility regime"""
        if not self.volatility_regimes:
            self.global_volatility = "medium"
            return
            
        # Count regimes
        regime_counts = {"high": 0, "medium": 0, "low": 0}
        
        for symbol, regime_data in self.volatility_regimes.items():
            regime = regime_data.get("regime", "medium")
            confidence = regime_data.get("confidence", 0)
            
            # Weight by confidence
            regime_counts[regime] += confidence
            
        # Determine dominant regime
        total = sum(regime_counts.values())
        
        if total == 0:
            self.global_volatility = "medium"
        else:
            # Normalize to percentages
            high_pct = regime_counts["high"] / total
            low_pct = regime_counts["low"] / total
            
            if high_pct > 0.5:  # More than 50% high volatility
                self.global_volatility = "high"
            elif low_pct > 0.5:  # More than 50% low volatility
                self.global_volatility = "low"
            else:
                self.global_volatility = "medium"
    
    def get_volatility(self, symbol):
        """
        Get current volatility (ATR) for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Current ATR value
        """
        if symbol not in self.volatility_data:
            return 0
            
        # Use preferred timeframe
        preferred_tf = self.config.get("preferred_timeframe", "1h")
        
        if preferred_tf in self.volatility_data[symbol]:
            vol_data = self.volatility_data[symbol][preferred_tf]
            atr = vol_data.get("atr", [0])
            return atr[-1] if len(atr) > 0 else 0
            
        # If preferred timeframe not available, use any available
        for tf in self.volatility_data[symbol]:
            vol_data = self.volatility_data[symbol][tf]
            atr = vol_data.get("atr", [0])
            return atr[-1] if len(atr) > 0 else 0
            
        return 0
    
    def get_volatility_regime(self, symbol):
        """
        Get volatility regime for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            str: Volatility regime ("high", "medium", or "low")
        """
        if symbol in self.volatility_regimes:
            return self.volatility_regimes[symbol]["regime"]
            
        return "medium"
    
    def get_global_volatility_regime(self):
        """
        Get global volatility regime
        
        Returns:
            str: Global volatility regime
        """
        return self.global_volatility
    
    def is_volatility_increasing(self, symbol):
        """
        Check if volatility is increasing for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if volatility is increasing
        """
        if symbol in self.recent_changes:
            return self.recent_changes[symbol]["change"] == "increasing"
            
        return False
    
    def is_volatility_contracting(self, symbol):
        """
        Check if volatility is contracting for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if volatility is contracting
        """
        if symbol in self.volatility_regimes:
            return self.volatility_regimes[symbol]["pattern"] == "contraction"
            
        return False
    
    def get_full_volatility_data(self, symbol):
        """
        Get complete volatility data for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Complete volatility data
        """
        result = {
            "regime": self.get_volatility_regime(symbol),
            "current_atr": self.get_volatility(symbol),
            "increasing": self.is_volatility_increasing(symbol),
            "contracting": self.is_volatility_contracting(symbol),
            "global_regime": self.global_volatility
        }
        
        if symbol in self.volatility_regimes:
            result.update(self.volatility_regimes[symbol])
            
        if symbol in self.recent_changes:
            result["recent_change"] = self.recent_changes[symbol]
            
        return result
