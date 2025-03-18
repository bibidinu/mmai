"""
Technical Analysis Module with is_ranging method added
"""
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
import talib
from typing import Dict, List, Any, Optional, Tuple

class TechnicalAnalyzer:
    """
    Performs technical analysis on market data with corrected pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the technical analyzer
        
        Args:
            config (dict): Technical analysis configuration
        """
        self.logger = logging.getLogger("technical_analyzer")
        self.config = config
        
        # Support/resistance parameters
        self.sr_timeframes = config.get("sr_timeframes", ["4h", "1d"])
        self.sr_periods = config.get("support_resistance_periods", 20)
        self.sr_threshold = config.get("support_resistance_threshold", 0.01)
        
        # Pattern recognition settings
        self.pattern_timeframes = config.get("pattern_timeframes", ["1h", "4h", "1d"])
        self.extrema_order = config.get("extrema_order", 5)
        
        # Trend detection parameters
        self.trend_timeframe = config.get("trend_timeframe", "1h")
        self.trend_periods = config.get("trend_periods", 14)
        
        # Range detection parameters
        self.range_threshold = config.get("range_threshold", 0.03)  # 3% range
        self.range_timeframe = config.get("range_timeframe", "1h")
        
        # Global trend reference assets
        self.reference_assets = config.get("reference_assets", ["BTCUSDT", "ETHUSDT"])
        
        # Selected indicators
        self.momentum_timeframe = config.get("momentum_timeframe", "1h")
        self.momentum_weights = config.get("momentum_weights", {
            "rsi": 0.3,
            "macd": 0.4,
            "ma": 0.3
        })
        
        # Pattern detection using TA-Lib
        # Note: Corrected to use only available pattern recognition functions
        self.pattern_functions = {
            # Continuation patterns
            "flag": talib.CDLHIKKAKE,               # Flag pattern (using Hikkake as substitute)
            "pennant": talib.CDLSEPARATINGLINES,    # Pennant (using separating lines as substitute)
            
            # Bullish patterns
            "hammer": talib.CDLHAMMER,              # Hammer
            "inverted_hammer": talib.CDLINVERTEDHAMMER,  # Inverted Hammer
            "engulfing_bullish": talib.CDLENGULFING,  # Bullish Engulfing
            "morning_star": talib.CDLMORNINGSTAR,   # Morning Star
            "piercing": talib.CDLPIERCING,          # Piercing pattern
            "doji_star": talib.CDLDOJISTAR,         # Doji Star
            
            # Bearish patterns
            "hanging_man": talib.CDLHANGINGMAN,     # Hanging Man
            "shooting_star": talib.CDLSHOOTINGSTAR,  # Shooting Star
            "engulfing_bearish": talib.CDLENGULFING,  # Bearish Engulfing
            "evening_star": talib.CDLEVENINGSTAR,   # Evening Star
            "dark_cloud_cover": talib.CDLDARKCLOUDCOVER,  # Dark Cloud Cover
            
            # Reversal patterns
            "doji": talib.CDLDOJI,                  # Doji
            "harami": talib.CDLHARAMI,              # Harami
            "harami_cross": talib.CDLHARAMICROSS,   # Harami Cross
            
            # Complex patterns
            "three_line_strike": talib.CDL3LINESTRIKE  # Three Line Strike
        }
        
        # Market data cache
        self.market_data = {}
        self.support_resistance_levels = {}
        self.trends = {}
        self.patterns = {}
        self.ranges = {}
        
        self.logger.info("Technical analyzer initialized")
    
    def update(self, market_data: Dict[str, Dict[str, pd.DataFrame]]):
        """
        Update technical indicators with new market data
        
        Args:
            market_data (dict): Market data by symbol and timeframe
        """
        self.market_data = market_data
        
        try:
            # Update support/resistance levels
            self._update_support_resistance()
            
            # Update trends
            self._update_trends()
            
            # Update chart patterns
            self._update_patterns()
            
            # Update ranges
            self._update_ranges()
            
        except Exception as e:
            self.logger.error(f"Error updating technical analysis: {str(e)}", exc_info=True)
    
    def _update_support_resistance(self):
        """Update support and resistance levels for all symbols"""
        for symbol, timeframes in self.market_data.items():
            try:
                self.support_resistance_levels[symbol] = {}
                
                for timeframe in self.sr_timeframes:
                    if timeframe in timeframes:
                        df = timeframes[timeframe]
                        levels = self._identify_support_resistance(df)
                        self.support_resistance_levels[symbol][timeframe] = levels
                        
            except Exception as e:
                self.logger.error(f"Error updating S/R for {symbol}: {str(e)}")
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify support and resistance levels from price data
        
        Args:
            df (DataFrame): Price data with OHLC
            
        Returns:
            dict: Support and resistance levels
        """
        if df.empty or len(df) < self.sr_periods:
            return {"support": [], "resistance": []}
        
        try:
            # Use closing prices for simplicity
            prices = df['close'].values
            
            # Find local extrema
            support = []
            resistance = []
            
            # Find peaks (resistance) and troughs (support)
            # Simple algorithm: a point is a peak if it's higher than n points before and after it
            # and a trough if it's lower than n points before and after it
            order = min(self.extrema_order, len(prices) // 3)  # Adjust order if not enough data
            
            for i in range(order, len(prices) - order):
                # Check if this is a peak (resistance)
                if all(prices[i] > prices[i-j] for j in range(1, order+1)) and \
                   all(prices[i] > prices[i+j] for j in range(1, order+1)):
                    resistance.append(prices[i])
                
                # Check if this is a trough (support)
                if all(prices[i] < prices[i-j] for j in range(1, order+1)) and \
                   all(prices[i] < prices[i+j] for j in range(1, order+1)):
                    support.append(prices[i])
            
            # Cluster similar levels (within threshold)
            support = self._cluster_levels(support)
            resistance = self._cluster_levels(resistance)
            
            return {
                "support": support,
                "resistance": resistance
            }
            
        except Exception as e:
            self.logger.error(f"Error in support/resistance detection: {str(e)}")
            return {"support": [], "resistance": []}
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """
        Cluster similar price levels
        
        Args:
            levels (list): Price levels
            
        Returns:
            list: Clustered price levels
        """
        if not levels:
            return []
        
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster similar levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            if sorted_levels[i] / current_cluster[0] - 1 < self.sr_threshold:
                # Add to current cluster
                current_cluster.append(sorted_levels[i])
            else:
                # Start a new cluster
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [sorted_levels[i]]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def _update_trends(self):
        """Update trend information for all symbols"""
        for symbol, timeframes in self.market_data.items():
            try:
                if self.trend_timeframe in timeframes:
                    df = timeframes[self.trend_timeframe]
                    self.trends[symbol] = self._identify_trend(df)
            except Exception as e:
                self.logger.error(f"Error updating trends for {symbol}: {str(e)}")
    
    def _identify_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify trend from price data
        
        Args:
            df (DataFrame): Price data with OHLC
            
        Returns:
            dict: Trend information
        """
        if df.empty or len(df) < self.trend_periods:
            return {"trend": "neutral", "strength": 0.0}
        
        try:
            # Calculate simple moving averages
            short_period = min(self.trend_periods, len(df) - 1)
            long_period = min(self.trend_periods * 2, len(df) - 1)
            
            # Ensure periods are valid
            if short_period <= 0 or long_period <= 0 or short_period >= len(df) or long_period >= len(df):
                return {"trend": "neutral", "strength": 0.0}
            
            prices = df['close'].values
            
            sma_short = talib.SMA(prices, timeperiod=short_period)
            sma_long = talib.SMA(prices, timeperiod=long_period)
            
            current_price = prices[-1]
            current_short = sma_short[-1]
            current_long = sma_long[-1]
            
            # Determine trend direction
            if current_short > current_long and current_price > current_short:
                trend = "bullish"
                strength = min(1.0, (current_short / current_long - 1) * 10)
            elif current_short < current_long and current_price < current_short:
                trend = "bearish"
                strength = min(1.0, (1 - current_short / current_long) * 10)
            else:
                trend = "neutral"
                strength = 0.0
            
            # Additional refinement using ADX if available
            try:
                adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=self.trend_periods)
                adx_value = adx[-1]
                
                # ADX > 25 indicates a strong trend
                if not np.isnan(adx_value):
                    strength = min(1.0, adx_value / 50)
            except:
                # ADX calculation may fail with insufficient data
                pass
            
            return {
                "trend": trend,
                "strength": strength
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend detection: {str(e)}")
            return {"trend": "neutral", "strength": 0.0}
    
    def _update_patterns(self):
        """Update chart patterns for all symbols"""
        for symbol, timeframes in self.market_data.items():
            try:
                self.patterns[symbol] = {}
                
                for timeframe in self.pattern_timeframes:
                    if timeframe in timeframes:
                        df = timeframes[timeframe]
                        patterns = self._identify_patterns(df)
                        self.patterns[symbol][timeframe] = patterns
                        
            except Exception as e:
                self.logger.error(f"Error updating patterns for {symbol}: {str(e)}")
    
    def _update_ranges(self):
        """Update range information for all symbols"""
        for symbol, timeframes in self.market_data.items():
            try:
                if self.range_timeframe in timeframes:
                    df = timeframes[self.range_timeframe]
                    self.ranges[symbol] = self._identify_range(df)
            except Exception as e:
                self.logger.error(f"Error updating ranges for {symbol}: {str(e)}")
    
    def _identify_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify if price is in a range
        
        Args:
            df (DataFrame): Price data with OHLC
            
        Returns:
            dict: Range information
        """
        if df.empty or len(df) < 20:  # Need sufficient data
            return {"is_ranging": False, "range_width": 0.0}
        
        try:
            prices = df['close'].values
            
            # Check using ADX
            try:
                adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
                adx_value = adx[-1]
                
                # ADX < 20 typically indicates a ranging market
                weak_trend = adx_value < 20
            except:
                weak_trend = False
            
            # Check using Bollinger Bands
            try:
                upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                
                # Calculate BB width relative to middle
                bb_width = (upper[-1] - lower[-1]) / middle[-1]
                
                # Narrow Bollinger Bands indicate consolidation/ranging
                narrow_bb = bb_width < 0.05  # 5% width is relatively narrow
            except:
                narrow_bb = False
            
            # Check price movement - range width as percentage
            recent_prices = prices[-20:]
            high = max(recent_prices)
            low = min(recent_prices)
            mid_price = (high + low) / 2
            
            range_width = (high - low) / mid_price
            
            # Consider market ranging if:
            # 1. Range width is below threshold
            # 2. AND either ADX shows weak trend OR Bollinger Bands are narrow
            is_ranging = range_width < self.range_threshold and (weak_trend or narrow_bb)
            
            return {
                "is_ranging": is_ranging,
                "range_width": range_width,
                "range_high": high,
                "range_low": low,
                "adx_value": adx_value if 'adx_value' in locals() else None,
                "bb_width": bb_width if 'bb_width' in locals() else None
            }
            
        except Exception as e:
            self.logger.error(f"Error in range detection: {str(e)}")
            return {"is_ranging": False, "range_width": 0.0}
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Identify chart patterns using TA-Lib
        
        Args:
            df (DataFrame): Price data with OHLC
            
        Returns:
            dict: Chart patterns with strength
        """
        if df.empty or len(df) < 10:  # Need minimum data for patterns
            return {}
        
        try:
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            
            patterns = {}
            
            # Check each pattern
            for pattern_name, pattern_func in self.pattern_functions.items():
                try:
                    # Calculate pattern
                    result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                    
                    # Check most recent value
                    if result is not None and len(result) > 0:
                        latest = result[-1]
                        if latest != 0:  # If pattern detected
                            patterns[pattern_name] = int(latest)
                except Exception as e:
                    self.logger.error(f"Error identifying chart patterns: {str(e)}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return {}
    
    def is_ranging(self, symbol: str) -> bool:
        """
        Check if a symbol is in a ranging market
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            bool: True if market is ranging, False otherwise
        """
        if symbol not in self.ranges:
            return False
            
        return self.ranges[symbol].get("is_ranging", False)
    
    def get_range_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get range data for a symbol
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            dict: Range data including width, high, and low prices
        """
        if symbol not in self.ranges:
            return {"is_ranging": False, "range_width": 0.0}
            
        return self.ranges[symbol]
    
    def get_global_trend(self) -> Dict[str, Any]:
        """
        Get global market trend based on reference assets
        
        Returns:
            dict: Global trend information including direction and strength
        """
        try:
            # Use reference assets to determine global trend
            trend_scores = []
            
            for asset in self.reference_assets:
                if asset in self.trends:
                    asset_trend = self.trends[asset]
                    
                    # Convert trend to numeric score
                    if asset_trend["trend"] == "bullish":
                        trend_score = asset_trend["strength"]
                    elif asset_trend["trend"] == "bearish":
                        trend_score = -asset_trend["strength"]
                    else:
                        trend_score = 0.0
                        
                    trend_scores.append(trend_score)
            
            # Calculate average trend score
            if trend_scores:
                avg_score = sum(trend_scores) / len(trend_scores)
                
                # Determine global trend
                if avg_score > 0.3:
                    trend = "bullish"
                    strength = min(1.0, avg_score)
                elif avg_score < -0.3:
                    trend = "bearish"
                    strength = min(1.0, abs(avg_score))
                else:
                    trend = "neutral"
                    strength = min(1.0, abs(avg_score) * 2)  # Scale up for neutral
                    
                return {
                    "trend": trend,
                    "strength": strength,
                    "score": avg_score
                }
            else:
                # Default if no reference assets available
                return {
                    "trend": "neutral",
                    "strength": 0.0,
                    "score": 0.0
                }
                
        except Exception as e:
            self.logger.error(f"Error getting global trend: {str(e)}")
            return {
                "trend": "neutral",
                "strength": 0.0,
                "score": 0.0
            }
    
    def get_support_resistance(self, symbol: str, timeframe: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get support and resistance levels for a symbol
        
        Args:
            symbol (str): Symbol to get levels for
            timeframe (str, optional): Timeframe to get levels for
            
        Returns:
            dict: Support and resistance levels
        """
        if symbol not in self.support_resistance_levels:
            return {"support": [], "resistance": []}
        
        if timeframe:
            return self.support_resistance_levels[symbol].get(timeframe, {"support": [], "resistance": []})
        
        # Combine levels from all timeframes
        result = {"support": [], "resistance": []}
        
        for tf_data in self.support_resistance_levels[symbol].values():
            result["support"].extend(tf_data.get("support", []))
            result["resistance"].extend(tf_data.get("resistance", []))
        
        # Cluster combined levels
        result["support"] = self._cluster_levels(result["support"])
        result["resistance"] = self._cluster_levels(result["resistance"])
        
        return result
    
    def get_trend(self, symbol: str) -> Dict[str, Any]:
        """
        Get trend information for a symbol
        
        Args:
            symbol (str): Symbol to get trend for
            
        Returns:
            dict: Trend information
        """
        return self.trends.get(symbol, {"trend": "neutral", "strength": 0.0})
    
    def get_patterns(self, symbol: str, timeframe: Optional[str] = None) -> Dict[str, int]:
        """
        Get chart patterns for a symbol
        
        Args:
            symbol (str): Symbol to get patterns for
            timeframe (str, optional): Timeframe to get patterns for
            
        Returns:
            dict: Chart patterns
        """
        if symbol not in self.patterns:
            return {}
        
        if timeframe:
            return self.patterns[symbol].get(timeframe, {})
        
        # Combine patterns from all timeframes
        result = {}
        
        for tf_patterns in self.patterns[symbol].values():
            for pattern, strength in tf_patterns.items():
                if pattern in result:
                    # Keep the strongest signal
                    result[pattern] = max(result[pattern], strength, key=abs)
                else:
                    result[pattern] = strength
        
        return result
    
    def get_momentum_score(self, symbol: str) -> float:
        """
        Calculate momentum score for a symbol (-1.0 to 1.0)
        
        Args:
            symbol (str): Symbol to calculate for
            
        Returns:
            float: Momentum score
        """
        if symbol not in self.market_data or self.momentum_timeframe not in self.market_data[symbol]:
            return 0.0
        
        try:
            df = self.market_data[symbol][self.momentum_timeframe]
            
            if df.empty or len(df) < 30:  # Need sufficient data
                return 0.0
            
            prices = df['close'].values
            
            # Calculate RSI
            rsi = talib.RSI(prices, timeperiod=14)
            rsi_value = (rsi[-1] - 50) / 50  # Normalize to -1 to 1
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(prices)
            macd_value = macd_hist[-1] / prices[-1] * 100  # Normalize
            macd_value = max(min(macd_value, 1.0), -1.0)   # Clamp to -1 to 1
            
            # Calculate moving average direction
            ma = talib.SMA(prices, timeperiod=20)
            ma_direction = (prices[-1] / ma[-1] - 1) * 10  # Normalize
            ma_direction = max(min(ma_direction, 1.0), -1.0)  # Clamp to -1 to 1
            
            # Combine indicators with weights
            momentum = (
                self.momentum_weights.get("rsi", 0.3) * rsi_value +
                self.momentum_weights.get("macd", 0.4) * macd_value +
                self.momentum_weights.get("ma", 0.3) * ma_direction
            )
            
            return max(min(momentum, 1.0), -1.0)  # Ensure result is between -1 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum for {symbol}: {str(e)}")
            return 0.0
    
    def is_overbought(self, symbol: str) -> bool:
        """
        Check if a symbol is overbought
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            bool: True if overbought
        """
        if symbol not in self.market_data or self.momentum_timeframe not in self.market_data[symbol]:
            return False
        
        try:
            df = self.market_data[symbol][self.momentum_timeframe]
            
            if df.empty or len(df) < 14:  # Need sufficient data for RSI
                return False
            
            prices = df['close'].values
            
            # Calculate RSI
            rsi = talib.RSI(prices, timeperiod=14)
            
            # RSI > 70 indicates overbought
            return rsi[-1] > 70
            
        except Exception as e:
            self.logger.error(f"Error checking overbought for {symbol}: {str(e)}")
            return False
    
    def is_oversold(self, symbol: str) -> bool:
        """
        Check if a symbol is oversold
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            bool: True if oversold
        """
        if symbol not in self.market_data or self.momentum_timeframe not in self.market_data[symbol]:
            return False
        
        try:
            df = self.market_data[symbol][self.momentum_timeframe]
            
            if df.empty or len(df) < 14:  # Need sufficient data for RSI
                return False
            
            prices = df['close'].values
            
            # Calculate RSI
            rsi = talib.RSI(prices, timeperiod=14)
            
            # RSI < 30 indicates oversold
            return rsi[-1] < 30
            
        except Exception as e:
            self.logger.error(f"Error checking oversold for {symbol}: {str(e)}")
            return False
    
    def get_market_context(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market context for a symbol
        
        Args:
            symbol (str): Symbol to analyze
            
        Returns:
            dict: Market context information
        """
        context = {
            "symbol": symbol,
            "trend": self.get_trend(symbol),
            "momentum": self.get_momentum_score(symbol),
            "patterns": self.get_patterns(symbol),
            "support_resistance": self.get_support_resistance(symbol),
            "overbought": self.is_overbought(symbol),
            "oversold": self.is_oversold(symbol),
            "global_trend": self.get_global_trend(),
            "is_ranging": self.is_ranging(symbol),
            "range_data": self.get_range_data(symbol)
        }
        
        return context
