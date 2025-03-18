"""
Volume Analysis Module
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats

class VolumeAnalyzer:
    """
    Analyzes trading volume patterns and anomalies
    """
    
    def __init__(self, config):
        """
        Initialize the volume analyzer
        
        Args:
            config (dict): Volume analysis configuration
        """
        self.logger = logging.getLogger("volume_analyzer")
        self.config = config
        
        # Analysis parameters
        self.lookback_periods = config.get("lookback_periods", 20)
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0)  # z-score threshold
        self.volume_increase_threshold = config.get("volume_increase_threshold", 1.5)  # 50% increase
        
        # Volume data storage
        self.volume_data = {}
        self.volume_anomalies = {}
        self.volume_profiles = {}
        
        self.logger.info("Volume analyzer initialized")
    
    def update(self, market_data):
        """
        Update volume data and perform analysis
        
        Args:
            market_data (dict): Market data indexed by symbol and timeframe
        """
        self.volume_data = {}
        
        # Analyze each symbol and timeframe
        for symbol in market_data:
            self.volume_data[symbol] = {}
            self.volume_anomalies[symbol] = {}
            
            for timeframe in market_data[symbol]:
                # Skip timeframes without volume data
                df = market_data[symbol][timeframe]
                
                if df is None or "volume" not in df.columns or len(df) < self.lookback_periods:
                    continue
                
                # Store volume data
                self.volume_data[symbol][timeframe] = df["volume"].copy()
                
                # Detect volume anomalies
                self._detect_volume_anomalies(symbol, timeframe, df)
                
                # Build volume profile
                if timeframe == self.config.get("profile_timeframe", "1d"):
                    self._build_volume_profile(symbol, df)
    
    def _detect_volume_anomalies(self, symbol, timeframe, df):
        """
        Detect volume anomalies
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            df (pd.DataFrame): OHLCV data
        """
        if df is None or "volume" not in df.columns or len(df) < self.lookback_periods:
            return
            
        try:
            # Calculate volume statistics
            recent_volumes = df["volume"].values[-self.lookback_periods:]
            
            # Calculate z-scores
            mean_volume = np.mean(recent_volumes[:-1])  # Exclude the most recent
            std_volume = np.std(recent_volumes[:-1])
            
            if std_volume == 0:
                return
                
            last_volume = recent_volumes[-1]
            volume_z_score = (last_volume - mean_volume) / std_volume
            
            # Check for anomalies
            is_anomaly = abs(volume_z_score) > self.anomaly_threshold
            
            # Store results
            if timeframe not in self.volume_anomalies[symbol]:
                self.volume_anomalies[symbol][timeframe] = []
                
            self.volume_anomalies[symbol][timeframe].append({
                "timestamp": df["timestamp"].iloc[-1] if "timestamp" in df.columns else None,
                "volume": last_volume,
                "z_score": volume_z_score,
                "is_anomaly": is_anomaly,
                "direction": "increase" if volume_z_score > 0 else "decrease"
            })
            
            # Keep only recent anomalies
            max_anomalies = self.config.get("max_anomalies", 10)
            if len(self.volume_anomalies[symbol][timeframe]) > max_anomalies:
                self.volume_anomalies[symbol][timeframe] = self.volume_anomalies[symbol][timeframe][-max_anomalies:]
                
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {str(e)}")
    
    def _build_volume_profile(self, symbol, df):
        """
        Build volume profile for price levels
        
        Args:
            symbol (str): Trading symbol
            df (pd.DataFrame): OHLCV data
        """
        if df is None or "volume" not in df.columns or len(df) < 50:
            return
            
        try:
            # Create price bins
            price_range = df["high"].max() - df["low"].min()
            bin_size = price_range / self.config.get("profile_bins", 20)
            
            if bin_size <= 0:
                return
                
            bins = np.arange(df["low"].min(), df["high"].max() + bin_size, bin_size)
            
            # Initialize volume profile
            profile = {bin: 0 for bin in bins}
            
            # Distribute volume across price range for each candle
            for _, row in df.iterrows():
                candle_range = row["high"] - row["low"]
                if candle_range <= 0:
                    continue
                    
                # Simple distribution: proportional to price range
                for bin in bins:
                    if bin <= row["high"] and bin >= row["low"]:
                        # Distribute volume proportionally
                        profile[bin] += row["volume"] / (candle_range / bin_size)
            
            # Convert to list of price levels and volumes
            price_levels = list(profile.keys())
            volumes = list(profile.values())
            
            # Find value areas
            total_volume = sum(volumes)
            value_area_volume = total_volume * self.config.get("value_area_percentage", 0.7)
            
            # Sort by volume
            volume_levels = list(zip(price_levels, volumes))
            volume_levels.sort(key=lambda x: x[1], reverse=True)
            
            # Collect levels until we reach the value area volume
            value_area_levels = []
            current_volume = 0
            
            for level, volume in volume_levels:
                value_area_levels.append(level)
                current_volume += volume
                
                if current_volume >= value_area_volume:
                    break
            
            # Find point of control (price level with highest volume)
            point_of_control = volume_levels[0][0] if volume_levels else None
            
            # Store volume profile
            self.volume_profiles[symbol] = {
                "price_levels": price_levels,
                "volumes": volumes,
                "point_of_control": point_of_control,
                "value_area_high": max(value_area_levels) if value_area_levels else None,
                "value_area_low": min(value_area_levels) if value_area_levels else None
            }
            
        except Exception as e:
            self.logger.error(f"Error building volume profile: {str(e)}")
    
    def has_volume_anomaly(self, symbol):
        """
        Check if a symbol has a recent volume anomaly
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if anomaly detected, False otherwise
        """
        if symbol not in self.volume_anomalies:
            return False
            
        # Check preferred timeframes
        preferred_timeframes = self.config.get("anomaly_timeframes", ["15m", "1h", "4h"])
        
        for tf in preferred_timeframes:
            if tf in self.volume_anomalies[symbol] and self.volume_anomalies[symbol][tf]:
                # Check most recent anomaly
                latest = self.volume_anomalies[symbol][tf][-1]
                if latest.get("is_anomaly", False):
                    return True
        
        return False
    
    def get_volume_increase(self, symbol):
        """
        Get recent volume increase percentage
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Volume increase percentage (1.0 = no change)
        """
        if symbol not in self.volume_data:
            return 1.0
            
        # Use hourly timeframe if available
        preferred_tf = self.config.get("preferred_timeframe", "1h")
        
        if preferred_tf in self.volume_data[symbol]:
            volumes = self.volume_data[symbol][preferred_tf]
            
            if len(volumes) >= 3:
                # Compare latest volume to average of previous periods
                latest_volume = volumes.iloc[-1]
                previous_avg = volumes.iloc[-6:-1].mean()  # Average of 5 previous periods
                
                if previous_avg > 0:
                    return latest_volume / previous_avg
        
        # If preferred timeframe not available, try other timeframes
        for tf in self.volume_data[symbol]:
            volumes = self.volume_data[symbol][tf]
            
            if len(volumes) >= 3:
                latest_volume = volumes.iloc[-1]
                previous_avg = volumes.iloc[-6:-1].mean()
                
                if previous_avg > 0:
                    return latest_volume / previous_avg
        
        return 1.0
    
    def get_volume_profile(self, symbol):
        """
        Get volume profile for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Volume profile data
        """
        if symbol in self.volume_profiles:
            return self.volume_profiles[symbol]
            
        return {
            "price_levels": [],
            "volumes": [],
            "point_of_control": None,
            "value_area_high": None,
            "value_area_low": None
        }
    
    def is_high_volume_breakout(self, symbol, price, direction):
        """
        Check if a price movement is a high-volume breakout
        
        Args:
            symbol (str): Trading symbol
            price (float): Current price
            direction (str): "up" or "down"
            
        Returns:
            bool: True if high-volume breakout detected
        """
        # Check for volume anomaly
        if not self.has_volume_anomaly(symbol):
            return False
            
        # Check volume increase
        volume_increase = self.get_volume_increase(symbol)
        if volume_increase < self.volume_increase_threshold:
            return False
            
        # Check if price is breaking out of value area
        profile = self.get_volume_profile(symbol)
        
        if direction == "up" and profile["value_area_high"] and price > profile["value_area_high"]:
            return True
            
        if direction == "down" and profile["value_area_low"] and price < profile["value_area_low"]:
            return True
            
        return False
    
    def is_climax_volume(self, symbol):
        """
        Check if a symbol has climax volume (extremely high volume)
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if climax volume detected
        """
        if symbol not in self.volume_anomalies:
            return False
            
        # Check daily timeframe if available
        if "1d" in self.volume_anomalies[symbol] and self.volume_anomalies[symbol]["1d"]:
            latest = self.volume_anomalies[symbol]["1d"][-1]
            
            # Very high z-score indicates climax volume
            if latest.get("z_score", 0) > self.config.get("climax_threshold", 3.0):
                return True
        
        return False
    
    def get_average_volume(self, symbol, timeframe="1h"):
        """
        Get average volume for a symbol
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            
        Returns:
            float: Average volume
        """
        if symbol in self.volume_data and timeframe in self.volume_data[symbol]:
            return self.volume_data[symbol][timeframe].mean()
            
        return 0
