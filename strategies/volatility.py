"""
Volatility-Based Trading Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd
import talib as ta

class VolatilityStrategy:
    """
    Strategy that trades based on volatility patterns
    """
    
    def __init__(self, config, volatility_analyzer):
        """
        Initialize the volatility-based strategy
        
        Args:
            config (dict): Strategy configuration
            volatility_analyzer: Volatility analysis module
        """
        self.logger = logging.getLogger("volatility_strategy")
        self.config = config
        self.volatility = volatility_analyzer
        
        # Configuration parameters with defaults
        self.contraction_threshold = config.get("contraction_threshold", 0.5)
        self.expansion_threshold = config.get("expansion_threshold", 2.0)
        self.preferred_timeframe = config.get("preferred_timeframe", "4h")
        self.breakout_direction_periods = config.get("breakout_direction_periods", 3)
        self.profit_target = config.get("profit_target", 0.03)  # 3%
        self.stop_loss_multiplier = config.get("stop_loss_multiplier", 1.0)
        
        self.logger.info("Volatility strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a volatility-based entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get OHLCV data
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < 20:  # Need enough data for volatility analysis
            return None
        
        # Check for volatility contraction
        if not self.volatility.is_volatility_contracting(symbol):
            return None
        
        # Get volatility data
        vol_data = self.volatility.get_full_volatility_data(symbol)
        
        # We're looking for signs of volatility expansion after contraction
        if not vol_data.get("contracting", False) or vol_data.get("increasing", False) is False:
            return None
        
        # Determine breakout direction
        direction = self._determine_breakout_direction(candles)
        
        if not direction:
            self.logger.debug(f"No clear breakout direction for {symbol}")
            return None
        
        # Current price and ATR
        current_price = candles["close"].iloc[-1]
        atr = vol_data.get("current_atr", 0)
        
        if atr <= 0:
            return None
        
        # Calculate stop loss and target
        if direction == "long":
            stop_loss = current_price - (atr * self.stop_loss_multiplier)
            target_price = current_price + (atr * (self.profit_target / (atr / current_price)))
        else:  # short
            stop_loss = current_price + (atr * self.stop_loss_multiplier)
            target_price = current_price - (atr * (self.profit_target / (atr / current_price)))
        
        # Log signal
        self.logger.info(f"Volatility breakout {direction} signal for {symbol} at {current_price}, ATR: {atr:.5f}")
        
        return {
            "direction": direction,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "volatility_regime": vol_data.get("regime", "medium"),
            "atr": atr
        }
    
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
        
        # Get entry data from position
        entry_price = position.get("entry_price", current_price)
        target_price = position.get("target_price")
        position_atr = position.get("atr", 0)
        
        # Current ATR
        current_atr = self.volatility.get_volatility(symbol)
        
        # Check if volatility has contracted again
        if current_atr < position_atr * 0.7:
            self.logger.info(f"Exiting {direction} position for {symbol} due to volatility contraction")
            return True
        
        # Exit if target reached
        if direction == "long" and target_price and current_price >= target_price:
            self.logger.info(f"Target reached for {symbol} long at {current_price}, target: {target_price:.5f}")
            return True
        elif direction == "short" and target_price and current_price <= target_price:
            self.logger.info(f"Target reached for {symbol} short at {current_price}, target: {target_price:.5f}")
            return True
        
        # Check for reversal in momentum
        if self._check_momentum_reversal(candles, direction):
            self.logger.info(f"Momentum reversal detected for {symbol} {direction} at {current_price}")
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
    
    def _determine_breakout_direction(self, candles):
        """
        Determine the likely direction of a volatility breakout
        
        Args:
            candles (pd.DataFrame): OHLCV data
            
        Returns:
            str or None: "long", "short", or None if unclear
        """
        if len(candles) < self.breakout_direction_periods + 5:
            return None
        
        # Calculate indicators for direction bias
        try:
            # RSI for momentum
            rsi = ta.RSI(candles["close"].values, timeperiod=14)
            
            # ADX for trend strength
            adx = ta.ADX(candles["high"].values, candles["low"].values, candles["close"].values, timeperiod=14)
            
            # MACD for trend direction
            macd, signal, hist = ta.MACD(
                candles["close"].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            
            if rsi is None or adx is None or macd is None or signal is None or hist is None:
                return None
                
            # Recent price action
            recent_candles = candles.tail(self.breakout_direction_periods)
            price_direction = "up" if recent_candles["close"].pct_change().mean() > 0 else "down"
            
            # Current values
            current_rsi = rsi[-1]
            current_adx = adx[-1]
            current_hist = hist[-1]
            
            # Use a combination of indicators to determine direction
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI
            if current_rsi > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # ADX - only consider if trend is strong
            if current_adx > 20:
                if price_direction == "up":
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            
            # MACD Histogram
            if current_hist > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # Recent price action
            if price_direction == "up":
                bullish_signals += 1
            else:
                bearish_signals += 1
                
            # Determine bias
            if bullish_signals >= 3:  # At least 3 of 4 signals are bullish
                return "long"
            elif bearish_signals >= 3:  # At least 3 of 4 signals are bearish
                return "short"
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining breakout direction: {str(e)}")
            return None
    
    def _check_momentum_reversal(self, candles, direction):
        """
        Check if momentum has reversed against the position
        
        Args:
            candles (pd.DataFrame): OHLCV data
            direction (str): Position direction ("long" or "short")
            
        Returns:
            bool: True if momentum has reversed, False otherwise
        """
        try:
            # Calculate indicators
            rsi = ta.RSI(candles["close"].values, timeperiod=14)
            macd, signal, hist = ta.MACD(
                candles["close"].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            
            if rsi is None or hist is None or len(rsi) < 3 or len(hist) < 3:
                return False
                
            # Check for RSI divergence
            if direction == "long":
                # Bearish divergence: price making higher high, RSI making lower high
                price_higher_high = candles["high"].iloc[-1] > candles["high"].iloc[-2]
                rsi_lower_high = rsi[-1] < rsi[-2]
                
                if price_higher_high and rsi_lower_high:
                    return True
                    
                # MACD histogram turning negative
                if hist[-2] > 0 and hist[-1] < 0:
                    return True
            else:  # short
                # Bullish divergence: price making lower low, RSI making higher low
                price_lower_low = candles["low"].iloc[-1] < candles["low"].iloc[-2]
                rsi_higher_low = rsi[-1] > rsi[-2]
                
                if price_lower_low and rsi_higher_low:
                    return True
                    
                # MACD histogram turning positive
                if hist[-2] < 0 and hist[-1] > 0:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking momentum reversal: {str(e)}")
            return False
