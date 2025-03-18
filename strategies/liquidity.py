"""
Liquidity Analysis Trading Strategy Implementation
"""
import logging
import numpy as np
import pandas as pd
import time

class LiquidityStrategy:
    """
    Strategy that trades based on order book liquidity analysis
    """
    
    def __init__(self, config):
        """
        Initialize the liquidity analysis strategy
        
        Args:
            config (dict): Strategy configuration
        """
        self.logger = logging.getLogger("liquidity_strategy")
        self.config = config
        
        # Configuration parameters with defaults
        self.imbalance_threshold = config.get("imbalance_threshold", 3.0)  # Ratio of buy/sell liquidity
        self.depth_levels = config.get("depth_levels", 20)  # Number of price levels to analyze
        self.min_liquidity = config.get("min_liquidity", 100000)  # Minimum required liquidity (USD)
        self.preferred_timeframe = config.get("preferred_timeframe", "5m")  # For candle data
        self.profit_target = config.get("profit_target", 0.01)  # 1%
        self.stop_loss_multiplier = config.get("stop_loss_multiplier", 2.0)
        
        # Cache for liquidity data
        self.liquidity_cache = {}
        self.last_update_time = {}
        
        self.logger.info("Liquidity strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if a liquidity-based entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get order book data from market_data if available
        orderbook = self._get_orderbook(symbol, market_data)
        
        if not orderbook:
            return None
        
        # Get OHLCV data for price context
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < 10:
            return None
        
        # Analyze order book
        analysis = self._analyze_orderbook(symbol, orderbook, candles)
        
        if not analysis:
            return None
        
        # Look for significant imbalances
        if analysis["imbalance_ratio"] >= self.imbalance_threshold:
            # Bullish imbalance (more buy orders than sell orders)
            current_price = candles["close"].iloc[-1]
            
            # Calculate stop loss and target
            atr = self._calculate_atr(candles)
            stop_loss = current_price - (atr * self.stop_loss_multiplier)
            target_price = current_price * (1 + self.profit_target)
            
            self.logger.info(f"Bullish liquidity imbalance for {symbol} at {current_price}, ratio: {analysis['imbalance_ratio']:.2f}")
            
            return {
                "direction": "long",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "imbalance_ratio": analysis["imbalance_ratio"],
                "buy_wall_depth": analysis["buy_wall_depth"],
                "liquidity_score": analysis["liquidity_score"]
            }
            
        elif 1 / analysis["imbalance_ratio"] >= self.imbalance_threshold:
            # Bearish imbalance (more sell orders than buy orders)
            current_price = candles["close"].iloc[-1]
            
            # Calculate stop loss and target
            atr = self._calculate_atr(candles)
            stop_loss = current_price + (atr * self.stop_loss_multiplier)
            target_price = current_price * (1 - self.profit_target)
            
            self.logger.info(f"Bearish liquidity imbalance for {symbol} at {current_price}, ratio: {1/analysis['imbalance_ratio']:.2f}")
            
            return {
                "direction": "short",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "imbalance_ratio": 1 / analysis["imbalance_ratio"],
                "sell_wall_depth": analysis["sell_wall_depth"],
                "liquidity_score": analysis["liquidity_score"]
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
        
        if candles is None or len(candles) < 5:
            return False
        
        # Get current price
        current_price = candles["close"].iloc[-1]
        
        # Get order book data from market_data if available
        orderbook = self._get_orderbook(symbol, market_data)
        
        # Get target price from position
        target_price = position.get("target_price")
        entry_price = position.get("entry_price", current_price)
        
        # Exit if target reached
        if direction == "long" and target_price and current_price >= target_price:
            self.logger.info(f"Target reached for {symbol} long at {current_price}, target: {target_price:.5f}")
            return True
        elif direction == "short" and target_price and current_price <= target_price:
            self.logger.info(f"Target reached for {symbol} short at {current_price}, target: {target_price:.5f}")
            return True
        
        # Check for liquidity shift
        if orderbook:
            analysis = self._analyze_orderbook(symbol, orderbook, candles)
            
            if analysis:
                # Check if liquidity imbalance has shifted against our position
                if direction == "long" and analysis["imbalance_ratio"] < 1.0:
                    # More selling pressure than buying
                    self.logger.info(f"Liquidity shifted bearish for {symbol} long at {current_price}, ratio: {analysis['imbalance_ratio']:.2f}")
                    return True
                elif direction == "short" and analysis["imbalance_ratio"] > 1.0:
                    # More buying pressure than selling
                    self.logger.info(f"Liquidity shifted bullish for {symbol} short at {current_price}, ratio: {analysis['imbalance_ratio']:.2f}")
                    return True
        
        # Check for price action against position
        if self._check_adverse_price_action(candles, direction):
            self.logger.info(f"Adverse price action for {symbol} {direction} at {current_price}")
            return True
        
        return False
    
    def _get_orderbook(self, symbol, market_data):
        """
        Get orderbook data for the symbol
        
        Args:
            symbol (str): Symbol to get orderbook for
            market_data (dict): Market data
            
        Returns:
            dict or None: Orderbook data
        """
        try:
            # Check if we have orderbook data in the market_data
            for exchange_name, exchange_data in market_data.items():
                if isinstance(exchange_data, dict) and "orderbook_data" in exchange_data:
                    if symbol in exchange_data["orderbook_data"]:
                        return exchange_data["orderbook_data"][symbol]
            
            # If we're here, we didn't find orderbook data
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {str(e)}")
            return None
    
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
    
    def _analyze_orderbook(self, symbol, orderbook, candles=None):
        """
        Analyze orderbook to find liquidity imbalances
        
        Args:
            symbol (str): Symbol to analyze
            orderbook (dict): Orderbook data
            candles (pd.DataFrame, optional): OHLCV data
            
        Returns:
            dict or None: Analysis results
        """
        try:
            # Check if we have bids and asks
            if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
                return None
                
            bids = orderbook["bids"]
            asks = orderbook["asks"]
            
            if not bids or not asks:
                return None
                
            # Convert to lists if they're not already
            if isinstance(bids, dict):
                bids = [[float(price), float(qty)] for price, qty in bids.items()]
                
            if isinstance(asks, dict):
                asks = [[float(price), float(qty)] for price, qty in asks.items()]
                
            # Sort bids (descending) and asks (ascending)
            bids = sorted(bids, key=lambda x: -x[0])
            asks = sorted(asks, key=lambda x: x[0])
            
            # Limit depth to analyze
            bids = bids[:self.depth_levels]
            asks = asks[:self.depth_levels]
            
            # Calculate total volume on each side
            bid_volume = sum(qty for _, qty in bids)
            ask_volume = sum(qty for _, qty in asks)
            
            # Get current market price
            current_price = asks[0][0] if asks else (bids[0][0] if bids else 0)
            
            if candles is not None and len(candles) > 0:
                current_price = candles["close"].iloc[-1]
            
            # Calculate value in quote currency
            bid_value = sum(price * qty for price, qty in bids)
            ask_value = sum(price * qty for price, qty in asks)
            
            # Check minimum liquidity requirement
            if bid_value < self.min_liquidity or ask_value < self.min_liquidity:
                return None
                
            # Calculate imbalance ratio
            imbalance_ratio = bid_value / ask_value if ask_value > 0 else float('inf')
            
            # Detect buy/sell walls
            buy_walls = self._detect_liquidity_walls(bids, "buy")
            sell_walls = self._detect_liquidity_walls(asks, "sell")
            
            # Calculate overall liquidity score (0-1)
            total_liquidity = bid_value + ask_value
            liquidity_score = min(1.0, total_liquidity / (self.min_liquidity * 10))
            
            return {
                "current_price": current_price,
                "bid_value": bid_value,
                "ask_value": ask_value,
                "imbalance_ratio": imbalance_ratio,
                "buy_walls": buy_walls,
                "sell_walls": sell_walls,
                "buy_wall_depth": len(buy_walls),
                "sell_wall_depth": len(sell_walls),
                "liquidity_score": liquidity_score,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing orderbook: {str(e)}")
            return None
    
    def _detect_liquidity_walls(self, orders, side="buy"):
        """
        Detect liquidity walls in orderbook
        
        Args:
            orders (list): List of [price, quantity] pairs
            side (str): "buy" or "sell"
            
        Returns:
            list: List of detected walls with price and size
        """
        if not orders or len(orders) < 3:
            return []
            
        avg_size = sum(qty for _, qty in orders) / len(orders)
        walls = []
        
        for i, (price, qty) in enumerate(orders):
            # A wall is significantly larger than average order size
            if qty > avg_size * 2:
                walls.append({"price": price, "size": qty})
                
        # Sort walls by size (largest first)
        walls.sort(key=lambda x: -x["size"])
        
        return walls
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if not all(col in df.columns for col in ["high", "low", "close"]) or len(df) < period:
            return df["close"].iloc[-1] * 0.01  # Default to 1% of current price
            
        try:
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            
            # Calculate true range
            tr1 = abs(high[1:] - low[1:])
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            
            # Calculate ATR
            atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return df["close"].iloc[-1] * 0.01  # Default to 1% of current price
    
    def _check_adverse_price_action(self, candles, direction):
        """
        Check for price action that goes against our position
        
        Args:
            candles (pd.DataFrame): OHLCV data
            direction (str): Position direction ("long" or "short")
            
        Returns:
            bool: True if adverse price action is detected
        """
        if len(candles) < 3:
            return False
            
        # For long positions, check for bearish price action
        if direction == "long":
            # Check for bearish engulfing pattern
            if (candles["close"].iloc[-1] < candles["open"].iloc[-1] and  # Current candle is bearish
                candles["close"].iloc[-1] < candles["open"].iloc[-2] and  # Closes below prior open
                candles["open"].iloc[-1] > candles["close"].iloc[-2]):    # Opens above prior close
                return True
                
            # Check for consecutive bearish candles
            if (candles["close"].iloc[-3] < candles["open"].iloc[-3] and
                candles["close"].iloc[-2] < candles["open"].iloc[-2] and
                candles["close"].iloc[-1] < candles["open"].iloc[-1]):
                return True
        
        # For short positions, check for bullish price action
        elif direction == "short":
            # Check for bullish engulfing pattern
            if (candles["close"].iloc[-1] > candles["open"].iloc[-1] and  # Current candle is bullish
                candles["close"].iloc[-1] > candles["open"].iloc[-2] and  # Closes above prior open
                candles["open"].iloc[-1] < candles["close"].iloc[-2]):    # Opens below prior close
                return True
                
            # Check for consecutive bullish candles
            if (candles["close"].iloc[-3] > candles["open"].iloc[-3] and
                candles["close"].iloc[-2] > candles["open"].iloc[-2] and
                candles["close"].iloc[-1] > candles["open"].iloc[-1]):
                return True
        
        return False
    
    def get_liquidity_score(self, symbol):
        """
        Get the current liquidity score for a symbol
        
        Args:
            symbol (str): Symbol to check
            
        Returns:
            str: "high", "medium", or "low" liquidity
        """
        # Check if we have cached data
        if symbol in self.liquidity_cache:
            data = self.liquidity_cache[symbol]
            liquidity_score = data.get("liquidity_score", 0.5)
            
            if liquidity_score > 0.7:
                return "high"
            elif liquidity_score < 0.3:
                return "low"
            else:
                return "medium"
        
        # Default to medium if no data
        return "medium"
