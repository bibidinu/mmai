"""
ByBit Exchange Integration with improved error handling and fallback mechanisms
"""
import logging
import time
import json
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import websocket
import threading
import queue
import random

class BybitExchange:
    """
    Handles communication with ByBit exchange API with improved error handling
    """
    
    def __init__(self, credentials, config, testnet=True):
        """
        Initialize the ByBit exchange interface
        
        Args:
            credentials (dict): API credentials
            config (dict): Exchange configuration
            testnet (bool): Use testnet if True, mainnet if False
        """
        self.logger = logging.getLogger("bybit_exchange")
        self.config = config
        self.testnet = testnet
        self.simulation_mode = config.get("simulation_mode", True)
        
        # API credentials
        self.api_key = credentials.get("api_key", "")
        self.api_secret = credentials.get("api_secret", "")
        
        # Check if credentials are valid
        if not self.api_key or self.api_key == "YOUR_API_KEY" or not self.api_secret or self.api_secret == "YOUR_API_SECRET":
            self.logger.warning("Invalid or missing ByBit API credentials - operating in simulation mode only")
            self.simulation_mode = True
        
        # API endpoints - Updated for v5 API
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
            self.ws_base_url = "wss://stream-testnet.bybit.com/v5"
        else:
            self.base_url = "https://api.bybit.com"
            self.ws_base_url = "wss://stream.bybit.com/v5"
        
        # Websocket connections
        self.ws_market = None
        self.ws_private = None
        self.ws_connected = False
        self.ws_message_queue = queue.Queue()
        self.ws_connections = {}
        
        # Market data cache
        self.market_data = {}
        self.orderbook_data = {}
        self.ticker_data = {}
        
        # Position and order tracking
        self.positions = {}
        self.orders = {}
        self.balance = {}
        
        # Simulated balance (for simulation mode)
        self.simulated_balance = 10000.0  # Start with 10,000 USDT in simulation
        
        # Reconnection parameters
        self.max_retries = 5
        self.retry_delay = 5
        
        # Initialize connections
        self.session = requests.Session()
        
        # Set up retry mechanism
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('https://', retry_adapter)
        
        # Initialize connections and data
        self._connect_websockets()
        self._initialize_data()
        
        self.logger.info(f"ByBit exchange initialized (testnet: {testnet}, simulation: {self.simulation_mode})")
    
    def _initialize_data(self):
        """Initialize data by loading markets, positions, and balances"""
        try:
            # Get markets and symbols
            self._fetch_markets()
            
            # Get account balances if not in simulation mode
            if not self.simulation_mode:
                self._fetch_balances()
                # Get current positions
                self._fetch_positions()
            else:
                # Initialize simulated balance for quote currency
                quote_currency = self.config.get("quote_currency", "USDT")
                self.balance[quote_currency] = {
                    "free": self.simulated_balance,
                    "total": self.simulated_balance,
                    "locked": 0.0
                }
                self.logger.info(f"Initialized simulated balance: {self.simulated_balance} {quote_currency}")
            
        except Exception as e:
            self.logger.error(f"Error initializing exchange data: {str(e)}", exc_info=True)
            # Ensure we have at least basic data for simulation
            self._initialize_fallback_data()
    
    def _initialize_fallback_data(self):
        """Initialize fallback data when API calls fail"""
        try:
            # Set up basic ticker data for common symbols
            symbols = self.config.get("symbols_to_watch", ["BTCUSDT", "ETHUSDT"])
            for symbol in symbols:
                self.ticker_data[symbol] = {
                    "base_asset": symbol[:-4] if symbol.endswith("USDT") else symbol[:3],
                    "quote_asset": "USDT",
                    "min_qty": 0.001,
                    "max_qty": 1000.0,
                    "price_precision": 2,
                    "qty_precision": 5,
                    "last_price": self._get_fallback_price(symbol),
                    "timestamp": int(time.time() * 1000)
                }
            
            # Initialize balance with simulation amount
            quote_currency = self.config.get("quote_currency", "USDT")
            self.balance[quote_currency] = {
                "free": self.simulated_balance,
                "total": self.simulated_balance,
                "locked": 0.0
            }
            
            self.logger.info("Initialized fallback data for simulation mode")
        except Exception as e:
            self.logger.error(f"Error initializing fallback data: {str(e)}", exc_info=True)
    
    def _get_fallback_price(self, symbol):
        """Get fallback price for a symbol when API is unavailable"""
        # Default prices for common symbols
        fallback_prices = {
            "BTCUSDT": 65000.0,
            "ETHUSDT": 3500.0,
            "SOLUSDT": 150.0,
            "BNBUSDT": 550.0,
            "XRPUSDT": 0.55,
            "ADAUSDT": 0.45,
            "DOGEUSDT": 0.12,
            "MATICUSDT": 0.75,
            "LINKUSDT": 15.0,
            "AVAXUSDT": 35.0
        }
        
        return fallback_prices.get(symbol, 100.0)  # Default to 100 if symbol not in the list
    
    def _generate_signature(self, params):
        """
        Generate signature for authenticated requests
        
        Args:
            params (dict): Request parameters
            
        Returns:
            str: HMAC signature
        """
        timestamp = int(time.time() * 1000)
        params["api_key"] = self.api_key
        params["timestamp"] = timestamp
        
        # Sort parameters by key
        sorted_params = sorted(params.items())
        query_string = "&".join([f"{key}={value}" for key, value in sorted_params])
        
        # Generate signature
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(query_string, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _make_request(self, method, endpoint, params=None, auth=False, retry_count=0):
        """
        Make HTTP request to ByBit API with improved error handling
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict, optional): Request parameters
            auth (bool): Whether authentication is required
            retry_count (int): Current retry count
            
        Returns:
            dict: API response or None on error
        """
        # If in simulation mode and it's an authenticated request, simulate the response
        if self.simulation_mode and auth:
            return self._simulate_api_response(method, endpoint, params)
        
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
            
        # Add authentication if required
        if auth:
            signature = self._generate_signature(params)
            params["sign"] = signature
        
        try:
            # Make request
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=15)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params, timeout=15)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Handle response
            response.raise_for_status()
            data = response.json()
            
            # Check if the expected response structure exists
            if isinstance(data, dict):
                if "ret_code" in data:
                    if data["ret_code"] != 0:
                        error_msg = data.get("ret_msg", "Unknown error")
                        self.logger.error(f"API error: {error_msg}")
                        
                        # Fall back to simulation for auth requests after retries
                        if auth and retry_count >= self.max_retries:
                            self.logger.warning(f"Falling back to simulation for: {endpoint}")
                            return self._simulate_api_response(method, endpoint, params)
                        
                        # Retry for specific errors
                        if "rate limit" in error_msg.lower() and retry_count < self.max_retries:
                            retry_count += 1
                            retry_delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                            self.logger.warning(f"Rate limit reached, retrying in {retry_delay}s (attempt {retry_count}/{self.max_retries})")
                            time.sleep(retry_delay)
                            return self._make_request(method, endpoint, params, auth, retry_count)
                        
                        return None
                    return data.get("result", {})
                else:
                    # Handle different API response format
                    self.logger.debug(f"API response doesn't contain ret_code, returning full response")
                    return data
            else:
                self.logger.warning(f"Unexpected API response format: {data}")
                return data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {str(e)}")
            
            # Retry on connection errors
            if retry_count < self.max_retries:
                retry_count += 1
                retry_delay = self.retry_delay * (2 ** retry_count)  # Exponential backoff
                self.logger.warning(f"Connection error, retrying in {retry_delay}s (attempt {retry_count}/{self.max_retries})")
                time.sleep(retry_delay)
                return self._make_request(method, endpoint, params, auth, retry_count)
            
            # Fall back to simulation after all retries for auth requests
            if auth:
                self.logger.warning(f"API request failed, falling back to simulation for: {endpoint}")
                return self._simulate_api_response(method, endpoint, params)
            
            return None
            
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            return None
    
    def _simulate_api_response(self, method, endpoint, params):
        """
        Simulate API response for testing and when API is unavailable
        
        Args:
            method (str): HTTP method
            endpoint (str): API endpoint
            params (dict): Request parameters
            
        Returns:
            dict: Simulated API response
        """
        # Simulate different API endpoints
        if endpoint == "/v5/market/tickers":
            symbol = params.get("symbol", "BTCUSDT")
            price = self._get_fallback_price(symbol)
            return {
                "list": [
                    {
                        "symbol": symbol,
                        "lastPrice": str(price),
                        "bidPrice": str(price * 0.999),
                        "askPrice": str(price * 1.001),
                        "volume24h": str(random.uniform(1000, 10000)),
                    }
                ]
            }
        
        elif endpoint == "/v5/market/kline":
            symbol = params.get("symbol", "BTCUSDT")
            limit = int(params.get("limit", 200))
            base_price = self._get_fallback_price(symbol)
            
            # Generate random candles
            klines = []
            current_time = int(time.time())
            
            for i in range(limit):
                timestamp = current_time - ((limit - i) * 60)  # 1-minute intervals
                open_price = base_price * (1 + random.uniform(-0.02, 0.02))
                close_price = open_price * (1 + random.uniform(-0.01, 0.01))
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
                volume = random.uniform(1, 100)
                
                kline = [
                    timestamp * 1000,  # timestamp
                    str(open_price),   # open
                    str(high_price),   # high
                    str(low_price),    # low
                    str(close_price),  # close
                    str(volume)        # volume
                ]
                klines.append(kline)
            
            return {"list": klines}
        
        elif endpoint == "/v5/account/wallet-balance":
            return {
                "list": [
                    {
                        "coin": [
                            {
                                "coin": "USDT",
                                "free": str(self.simulated_balance),
                                "total": str(self.simulated_balance),
                                "locked": "0"
                            }
                        ]
                    }
                ]
            }
        
        elif endpoint == "/v5/order/create":
            # Simulate order creation
            order_id = f"simulated_order_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
            return {"orderId": order_id}
        
        elif endpoint == "/v5/order/cancel" or endpoint == "/v5/order/cancel-all":
            return {"success": True}
        
        elif endpoint == "/v5/position/list":
            return {"list": []}
        
        # Default response for unhandled endpoints
        return {"success": True}
    
    def _connect_websockets(self):
        """Connect to market and private websockets with improved error handling"""
        try:
            # Connect to market websocket
            self._connect_market_websocket()
            
            # Connect to private websocket if not in simulation mode
            if not self.simulation_mode:
                self._connect_private_websocket()
            
            # Start message handling thread
            self.ws_connected = True
            threading.Thread(target=self._handle_websocket_messages, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Websocket connection error: {str(e)}", exc_info=True)
            self.logger.warning("Operating with REST API fallback due to WebSocket connection failure")
    
    def _connect_market_websocket(self):
        """Connect to market data websocket using v5 API with improved error handling"""
        ws_url = f"{self.ws_base_url}/public/spot"
        
        def on_message(ws, message):
            try:
                self.ws_message_queue.put({"type": "market", "data": json.loads(message)})
            except Exception as e:
                self.logger.error(f"Error processing market websocket message: {str(e)}")
        
        def on_error(ws, error):
            self.logger.error(f"Market websocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"Market websocket closed: {close_msg}")
            # Reconnect after a delay with exponential backoff
            for attempt in range(1, 6):  # Try 5 times
                time.sleep(5 * attempt)
                try:
                    self.logger.info(f"Attempting to reconnect market websocket (attempt {attempt}/5)")
                    self._connect_market_websocket()
                    break
                except Exception as e:
                    self.logger.error(f"Failed to reconnect market websocket: {str(e)}")
        
        def on_open(ws):
            self.logger.info("Market websocket connected")
            # Subscribe to relevant topics
            self._subscribe_market_topics()
        
        try:
            self.ws_market = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            threading.Thread(target=self.ws_market.run_forever, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Failed to create market websocket: {str(e)}")
            raise
    
    def _connect_private_websocket(self):
        """Connect to private data websocket using v5 API with improved error handling"""
        ws_url = f"{self.ws_base_url}/private"
        
        def on_message(ws, message):
            try:
                self.ws_message_queue.put({"type": "private", "data": json.loads(message)})
            except Exception as e:
                self.logger.error(f"Error processing private websocket message: {str(e)}")
        
        def on_error(ws, error):
            self.logger.error(f"Private websocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.warning(f"Private websocket closed: {close_msg}")
            # Reconnect after a delay with exponential backoff
            for attempt in range(1, 6):  # Try 5 times
                time.sleep(5 * attempt)
                try:
                    self.logger.info(f"Attempting to reconnect private websocket (attempt {attempt}/5)")
                    self._connect_private_websocket()
                    break
                except Exception as e:
                    self.logger.error(f"Failed to reconnect private websocket: {str(e)}")
        
        def on_open(ws):
            self.logger.info("Private websocket connected")
            # Authenticate
            self._authenticate_private_websocket()
        
        try:
            self.ws_private = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            threading.Thread(target=self.ws_private.run_forever, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Failed to create private websocket: {str(e)}")
            raise
    
    def _authenticate_private_websocket(self):
        """Authenticate private websocket connection using v5 API format"""
        timestamp = int(time.time() * 1000)
        
        # Generate signature
        expires = timestamp + 10000  # 10 seconds expiry
        val = f'GET/realtime{expires}'
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(val, "utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # Send authentication message
        auth_message = {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }
        
        self.ws_private.send(json.dumps(auth_message))
        
        # Subscribe to private topics
        self._subscribe_private_topics()
    
    def _subscribe_market_topics(self):
        """Subscribe to market data topics using v5 API format"""
        try:
            # Get symbols to subscribe to
            symbols = self.config.get("symbols_to_watch", ["BTCUSDT", "ETHUSDT"])
            
            # Subscribe to kline data for each symbol and timeframe
            timeframes = self.config.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
            
            # Convert timeframes to v5 API format
            timeframe_map = {
                "1m": "1",
                "3m": "3",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "2h": "120",
                "4h": "240",
                "6h": "360",
                "12h": "720",
                "1d": "D",
                "1w": "W",
                "1M": "M"
            }
            
            args = []
            for symbol in symbols:
                for timeframe in timeframes:
                    # Subscribe to kline data
                    if timeframe in timeframe_map:
                        args.append(f"kline.{timeframe_map[timeframe]}.{symbol}")
                
                # Subscribe to order book data
                args.append(f"orderbook.40.{symbol}")
                
                # Subscribe to trades
                args.append(f"publicTrade.{symbol}")
            
            if args:
                subscribe_message = {
                    "op": "subscribe",
                    "args": args
                }
                self.ws_market.send(json.dumps(subscribe_message))
        except Exception as e:
            self.logger.error(f"Error subscribing to market topics: {str(e)}")
    
    def _subscribe_private_topics(self):
        """Subscribe to private data topics using v5 API format"""
        try:
            topics = [
                "order",
                "execution",
                "position",
                "wallet"
            ]
            
            subscribe_message = {
                "op": "subscribe",
                "args": topics
            }
            
            self.ws_private.send(json.dumps(subscribe_message))
        except Exception as e:
            self.logger.error(f"Error subscribing to private topics: {str(e)}")
    
    def _handle_websocket_messages(self):
        """Process incoming websocket messages with improved error handling"""
        while self.ws_connected:
            try:
                message = self.ws_message_queue.get(timeout=1)
                
                if message["type"] == "market":
                    self._process_market_message(message["data"])
                elif message["type"] == "private":
                    self._process_private_message(message["data"])
                
                self.ws_message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing websocket message: {str(e)}")
    
    def _process_market_message(self, message):
        """Process market data messages"""
        if "topic" not in message:
            return
            
        try:
            topic = message["topic"]
            data = message.get("data", {})
            
            # Process kline data
            if topic.startswith("kline."):
                self._process_kline_message(topic, data)
            
            # Process orderbook data
            elif topic.startswith("orderbook."):
                self._process_orderbook_message(topic, data)
            
            # Process trade data
            elif topic.startswith("publicTrade."):
                self._process_trade_message(topic, data)
        except Exception as e:
            self.logger.error(f"Error processing market message: {str(e)}")
    
    def _process_kline_message(self, topic, data):
        """Process kline (candlestick) data message"""
        try:
            # Extract symbol and timeframe from topic
            parts = topic.split(".")
            if len(parts) != 3:
                return
                
            timeframe = parts[1]
            symbol = parts[2]
            
            # Convert v5 API timeframe back to standard format
            timeframe_map = {
                "1": "1m",
                "3": "3m",
                "5": "5m",
                "15": "15m",
                "30": "30m",
                "60": "1h",
                "120": "2h",
                "240": "4h",
                "360": "6h",
                "720": "12h",
                "D": "1d",
                "W": "1w",
                "M": "1M"
            }
            
            if timeframe in timeframe_map:
                timeframe = timeframe_map[timeframe]
            
            # Ensure market data structure exists
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
                
            if timeframe not in self.market_data[symbol]:
                self.market_data[symbol][timeframe] = pd.DataFrame(columns=[
                    "timestamp", "open", "high", "low", "close", "volume"
                ])
            
            # Update kline data
            for candle in data:
                timestamp = candle["start"]
                
                # Create new candle data
                new_candle = {
                    "timestamp": timestamp,
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle["volume"])
                }
                
                # Update existing candle or add new one
                df = self.market_data[symbol][timeframe]
                idx = df.index[df["timestamp"] == timestamp].tolist()
                
                if idx:
                    # Update existing candle
                    for key, value in new_candle.items():
                        df.at[idx[0], key] = value
                else:
                    # Add new candle
                    df = pd.concat([df, pd.DataFrame([new_candle])], ignore_index=True)
                    
                    # Keep only the last N candles
                    max_candles = self.config.get("max_candles", 500)
                    if len(df) > max_candles:
                        df = df.iloc[-max_candles:]
                    
                    self.market_data[symbol][timeframe] = df
        except Exception as e:
            self.logger.error(f"Error processing kline message: {str(e)}")
    
    def _process_orderbook_message(self, topic, data):
        """Process orderbook data message"""
        try:
            # Extract symbol from topic
            parts = topic.split(".")
            if len(parts) != 3:
                return
                
            symbol = parts[2]
            
            # Update orderbook data
            self.orderbook_data[symbol] = {
                "bids": {float(price): float(qty) for price, qty in data.get("bids", [])},
                "asks": {float(price): float(qty) for price, qty in data.get("asks", [])},
                "timestamp": data.get("ts", int(time.time() * 1000))
            }
        except Exception as e:
            self.logger.error(f"Error processing orderbook message: {str(e)}")
    
    def _process_trade_message(self, topic, data):
        """Process public trade data message"""
        try:
            # Extract symbol from topic
            parts = topic.split(".")
            if len(parts) != 2:
                return
                
            symbol = parts[1]
            
            # Update ticker data
            if symbol in self.ticker_data:
                # Update last price
                self.ticker_data[symbol]["last_price"] = float(data[-1]["p"])
                self.ticker_data[symbol]["timestamp"] = data[-1]["t"]
        except Exception as e:
            self.logger.error(f"Error processing trade message: {str(e)}")
    
    def _process_private_message(self, message):
        """Process private data messages"""
        if "topic" not in message:
            return
        
        try:    
            topic = message["topic"]
            data = message.get("data", {})
            
            # Process order updates
            if topic == "order":
                self._process_order_update(data)
            
            # Process execution updates
            elif topic == "execution":
                self._process_execution_update(data)
            
            # Process position updates
            elif topic == "position":
                self._process_position_update(data)
            
            # Process wallet updates
            elif topic == "wallet":
                self._process_wallet_update(data)
        except Exception as e:
            self.logger.error(f"Error processing private message: {str(e)}")
    
    def _process_order_update(self, data):
        """Process order update"""
        order_id = data.get("orderId")
        if not order_id:
            return
            
        # Update order status
        self.orders[order_id] = data
    
    def _process_execution_update(self, data):
        """Process execution update"""
        order_id = data.get("orderId")
        if not order_id:
            return
            
        # Update order execution
        if order_id not in self.orders:
            self.orders[order_id] = {}
            
        if "executions" not in self.orders[order_id]:
            self.orders[order_id]["executions"] = []
            
        self.orders[order_id]["executions"].append(data)
    
    def _process_position_update(self, data):
        """Process position update"""
        symbol = data.get("symbol")
        if not symbol:
            return
            
        # Update position
        self.positions[symbol] = data
    
    def _process_wallet_update(self, data):
        """Process wallet update"""
        coin = data.get("coin")
        if not coin:
            return
            
        # Update balance
        self.balance[coin] = data
    
    def _fetch_markets(self):
        """Fetch available markets and symbols with improved error handling"""
        endpoint = "/v5/market/instruments-info"
        params = {"category": "spot"}
        
        # Fetch spot markets
        spot_markets = self._make_request("GET", endpoint, params)
        
        if spot_markets:
            # Process market data
            for market in spot_markets.get("list", []):
                symbol = market.get("symbol")
                if symbol:
                    self.ticker_data[symbol] = {
                        "base_asset": market.get("baseCoin"),
                        "quote_asset": market.get("quoteCoin"),
                        "min_qty": float(market.get("minOrderQty", "0")),
                        "max_qty": float(market.get("maxOrderQty", "0")),
                        "price_precision": int(market.get("priceScale", "0")),
                        "qty_precision": int(market.get("quantityScale", "0")),
                        "last_price": 0,
                        "timestamp": 0
                    }
        else:
            # If API request failed, initialize with default values for common symbols
            self._initialize_fallback_data()
        
        # Fetch futures markets if needed
        if self.config.get("trade_futures", False):
            params = {"category": "linear"}
            futures_markets = self._make_request("GET", endpoint, params)
            
            if futures_markets:
                # Process market data
                for market in futures_markets.get("list", []):
                    symbol = market.get("symbol")
                    if symbol:
                        self.ticker_data[symbol] = {
                            "base_asset": market.get("baseCoin"),
                            "quote_asset": market.get("quoteCoin"),
                            "min_qty": float(market.get("minOrderQty", "0")),
                            "max_qty": float(market.get("maxOrderQty", "0")),
                            "price_precision": int(market.get("priceScale", "0")),
                            "qty_precision": int(market.get("quantityScale", "0")),
                            "last_price": 0,
                            "timestamp": 0
                        }
    
    def _fetch_balances(self):
        """Fetch account balances with improved error handling"""
        if self.simulation_mode:
            # In simulation mode, use simulated balance
            quote_currency = self.config.get("quote_currency", "USDT")
            self.balance[quote_currency] = {
                "free": self.simulated_balance,
                "total": self.simulated_balance,
                "locked": 0.0
            }
            return
            
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": "SPOT"}
        
        balances = self._make_request("GET", endpoint, params, auth=True)
        
        if balances:
            for account in balances.get("list", []):
                for coin in account.get("coin", []):
                    asset = coin.get("coin")
                    self.balance[asset] = {
                        "free": float(coin.get("free", "0")),
                        "total": float(coin.get("total", "0")),
                        "locked": float(coin.get("locked", "0"))
                    }
    
    def _fetch_positions(self):
        """Fetch current positions with improved error handling"""
        # For spot, we don't have positions in the traditional sense
        # For futures, we need to fetch positions
        if not self.config.get("trade_futures", False):
            return
            
        if self.simulation_mode:
            return  # No positions in simulation mode by default
            
        endpoint = "/v5/position/list"
        params = {"category": "linear"}
        
        positions = self._make_request("GET", endpoint, params, auth=True)
        
        if positions:
            for position in positions.get("list", []):
                symbol = position.get("symbol")
                if symbol and float(position.get("size", "0")) > 0:
                    self.positions[symbol] = {
                        "symbol": symbol,
                        "direction": "long" if position.get("side") == "Buy" else "short",
                        "size": float(position.get("size", "0")),
                        "entry_price": float(position.get("entryPrice", "0")),
                        "mark_price": float(position.get("markPrice", "0")),
                        "unrealized_pnl": float(position.get("unrealisedPnl", "0")),
                        "leverage": float(position.get("leverage", "1"))
                    }
    
    def get_market_data(self, symbols, timeframes):
        """
        Get market data for specified symbols and timeframes
        
        Args:
            symbols (list): List of symbols
            timeframes (list): List of timeframes
            
        Returns:
            dict: Market data indexed by symbol and timeframe
        """
        result = {}
        
        # Check if we have requested data already
        missing_data = False
        for symbol in symbols:
            result[symbol] = {}
            
            if symbol not in self.market_data:
                missing_data = True
                continue
                
            for timeframe in timeframes:
                if timeframe in self.market_data[symbol]:
                    result[symbol][timeframe] = self.market_data[symbol][timeframe]
                else:
                    missing_data = True
        
        # If we're missing data, fetch it
        if missing_data:
            self._fetch_missing_data(symbols, timeframes)
            
            # Update result with fetched data
            for symbol in symbols:
                if symbol not in result:
                    result[symbol] = {}
                    
                if symbol in self.market_data:
                    for timeframe in timeframes:
                        if timeframe in self.market_data[symbol]:
                            result[symbol][timeframe] = self.market_data[symbol][timeframe]
        
        return result
    
    def _fetch_missing_data(self, symbols, timeframes):
        """
        Fetch missing market data for specified symbols and timeframes
        
        Args:
            symbols (list): List of symbols
            timeframes (list): List of timeframes
        """
        endpoint = "/v5/market/kline"
        
        for symbol in symbols:
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
                
            for timeframe in timeframes:
                if timeframe not in self.market_data[symbol]:
                    # Convert timeframe to API format
                    interval = self._convert_timeframe(timeframe)
                    
                    # Fetch kline data
                    params = {
                        "category": "spot",
                        "symbol": symbol,
                        "interval": interval,
                        "limit": 200  # Max allowed
                    }
                    
                    klines = self._make_request("GET", endpoint, params)
                    
                    if klines and "list" in klines:
                        # Process kline data
                        data = []
                        for kline in klines["list"]:
                            try:
                                data.append({
                                    "timestamp": int(kline[0]),
                                    "open": float(kline[1]),
                                    "high": float(kline[2]),
                                    "low": float(kline[3]),
                                    "close": float(kline[4]),
                                    "volume": float(kline[5])
                                })
                            except (IndexError, ValueError):
                                continue
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(data)
                        if not df.empty:
                            df.sort_values("timestamp", inplace=True)
                            self.market_data[symbol][timeframe] = df
                    else:
                        # If API request failed, create synthetic data
                        self._create_synthetic_market_data(symbol, timeframe)
    
    def _create_synthetic_market_data(self, symbol, timeframe):
        """
        Create synthetic market data when API is unavailable
        
        Args:
            symbol (str): Symbol to create data for
            timeframe (str): Timeframe to create data for
        """
        self.logger.warning(f"Creating synthetic market data for {symbol} {timeframe}")
        
        # Get base price
        base_price = self._get_fallback_price(symbol)
        
        # Create synthetic data
        data = []
        current_time = int(time.time())
        interval_seconds = self._timeframe_to_seconds(timeframe)
        
        for i in range(200):  # Generate 200 candles
            timestamp = current_time - ((200 - i) * interval_seconds)
            
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.01)  # Normal distribution with 1% standard deviation
            price = base_price * (1 + price_change * i/200)  # Gradual trend
            
            # Random candle formation
            open_price = price * (1 + np.random.normal(0, 0.005))
            close_price = price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
            
            # Random volume
            volume = abs(np.random.normal(100, 50))
            
            data.append({
                "timestamp": timestamp * 1000,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df.sort_values("timestamp", inplace=True)
        
        # Store in market_data
        self.market_data[symbol][timeframe] = df
    
    def _timeframe_to_seconds(self, timeframe):
        """
        Convert timeframe to seconds
        
        Args:
            timeframe (str): Timeframe string (e.g., "1m", "1h")
            
        Returns:
            int: Timeframe in seconds
        """
        conversions = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "12h": 43200,
            "1d": 86400,
            "1w": 604800,
            "1M": 2592000
        }
        
        return conversions.get(timeframe, 60)
    
    def _convert_timeframe(self, timeframe):
        """
        Convert timeframe to ByBit API format
        
        Args:
            timeframe (str): Timeframe in common format (e.g., "1m", "1h")
            
        Returns:
            str: Timeframe in ByBit API format
        """
        conversions = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "6h": "360",
            "12h": "720",
            "1d": "D",
            "1w": "W",
            "1M": "M"
        }
        
        return conversions.get(timeframe, "1")
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            list: List of current positions
        """
        if self.simulation_mode:
            # For simulation mode, return simulated positions
            return [pos for pos in self.positions.values()]
            
        # For spot trading, we don't have positions in the traditional sense
        # For futures, return the positions
        if not self.config.get("trade_futures", False):
            return []
            
        # Refresh positions from API
        self._fetch_positions()
            
        # Convert positions dict to list
        return [pos for pos in self.positions.values()]
    
    def get_balance(self):
        """
        Get account balance
        
        Returns:
            float: Account balance
        """
        # Return balance of the quote currency
        quote_currency = self.config.get("quote_currency", "USDT")
        
        if self.simulation_mode:
            return self.simulated_balance
            
        # Refresh balance from API if not in simulation mode
        if not self.simulation_mode:
            self._fetch_balances()
            
        if quote_currency in self.balance:
            return self.balance[quote_currency].get("total", 0)
            
        return 0
    
    def get_market_price(self, symbol):
        """
        Get current market price for a symbol
        
        Args:
            symbol (str): Symbol to get price for
            
        Returns:
            float: Current market price
        """
        if symbol in self.ticker_data and self.ticker_data[symbol].get("last_price", 0) > 0:
            return self.ticker_data[symbol].get("last_price", 0)
            
        # If not in cache or price is 0, fetch from API
        endpoint = "/v5/market/tickers"
        params = {"category": "spot", "symbol": symbol}
        
        tickers = self._make_request("GET", endpoint, params)
        
        if tickers and "list" in tickers and len(tickers["list"]) > 0:
            price = float(tickers["list"][0].get("lastPrice", "0"))
            
            # Update ticker data
            if symbol not in self.ticker_data:
                self.ticker_data[symbol] = {}
                
            self.ticker_data[symbol]["last_price"] = price
            
            return price
        
        # If API request failed, return fallback price
        return self._get_fallback_price(symbol)
    
    def open_position(self, symbol, direction, position_size, tp_levels, stop_loss):
        """
        Open a new position
        
        Args:
            symbol (str): Symbol to trade
            direction (str): "long" or "short"
            position_size (float): Position size in quote currency
            tp_levels (list): List of take profit prices
            stop_loss (float): Stop loss price
            
        Returns:
            dict: Trade result with status and details
        """
        # Calculate quantity
        price = self.get_market_price(symbol)
        if price <= 0:
            self.logger.error(f"Invalid price for {symbol}: {price}")
            return {"success": False, "error": "Invalid price"}
            
        quantity = position_size / price
        
        # Round quantity to appropriate precision
        if symbol in self.ticker_data:
            qty_precision = self.ticker_data[symbol].get("qty_precision", 4)
            quantity = round(quantity, qty_precision)
            
        # Check if quantity meets minimum
        if symbol in self.ticker_data:
            min_qty = self.ticker_data[symbol].get("min_qty", 0)
            if quantity < min_qty:
                self.logger.warning(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
                quantity = min_qty
        
        # Update simulated balance in simulation mode
        if self.simulation_mode:
            quote_currency = self.config.get("quote_currency", "USDT")
            if quote_currency in self.balance:
                if self.balance[quote_currency]["free"] < position_size:
                    self.logger.error(f"Insufficient balance for position: {position_size} {quote_currency}")
                    return {"success": False, "error": "Insufficient balance"}
                
                # Deduct position size from balance
                self.balance[quote_currency]["free"] -= position_size
                self.balance[quote_currency]["locked"] += position_size
                self.simulated_balance = self.balance[quote_currency]["total"]
                
                # Create simulated position
                position_id = f"{symbol}_{direction}_{int(time.time())}"
                
                self.positions[position_id] = {
                    "id": position_id,
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profits": tp_levels.copy() if tp_levels else [],
                    "order_id": f"sim_order_{int(time.time())}",
                    "entry_time": int(time.time() * 1000),
                    "tp1_hit": False,
                    "moved_to_be": False,
                    "position_size": position_size
                }
                
                self.logger.info(f"Opened simulated {direction} position for {symbol}: {quantity} @ {price}")
                
                return {
                    "success": True,
                    "trade_id": position_id,
                    "order_id": self.positions[position_id]["order_id"],
                    "symbol": symbol,
                    "direction": direction,
                    "quantity": quantity,
                    "entry_price": price
                }
        
        # Prepare order parameters
        side = "Buy" if direction == "long" else "Sell"
        
        # For futures, we need to determine order type
        if self.config.get("trade_futures", False):
            order_type = "Market"
            
            # Create order
            endpoint = "/v5/order/create"
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(quantity),
                "timeInForce": "GTC",
                "reduceOnly": False,
                "closeOnTrigger": False
            }
            
            # Add stop loss
            if stop_loss > 0:
                if direction == "long":
                    params["stopLoss"] = str(stop_loss)
                else:
                    params["stopLoss"] = str(stop_loss)
            
            # Add take profit levels (just the first one for now)
            if tp_levels and len(tp_levels) > 0:
                if direction == "long":
                    params["takeProfit"] = str(tp_levels[0])
                else:
                    params["takeProfit"] = str(tp_levels[0])
        else:
            # For spot, just create a simple market order
            order_type = "Market"
            
            # Create order
            endpoint = "/v5/order/create"
            params = {
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(quantity),
                "timeInForce": "GTC"
            }
        
        # Send order
        result = self._make_request("POST", endpoint, params, auth=True)
        
        if not result:
            return {"success": False, "error": "Order creation failed"}
            
        order_id = result.get("orderId")
        
        if not order_id:
            return {"success": False, "error": "No order ID returned"}
            
        # For spot, create separate orders for stop loss and take profits
        if not self.config.get("trade_futures", False):
            # Create stop loss order
            if stop_loss > 0:
                sl_side = "Sell" if direction == "long" else "Buy"
                sl_params = {
                    "category": "spot",
                    "symbol": symbol,
                    "side": sl_side,
                    "orderType": "StopMarket" if self.config.get("use_market_stops", True) else "StopLimit",
                    "qty": str(quantity),
                    "stopPrice": str(stop_loss),
                    "timeInForce": "GTC"
                }
                
                if sl_params["orderType"] == "StopLimit":
                    # For stop limit, we need a price
                    sl_params["price"] = str(stop_loss * 0.99 if direction == "long" else stop_loss * 1.01)
                
                self._make_request("POST", endpoint, sl_params, auth=True)
            
            # Create take profit orders
            for i, tp_price in enumerate(tp_levels):
                # Calculate TP quantity (partial)
                if i < len(tp_levels) - 1:
                    tp_qty = quantity * self.config.get("tp_percentages", [0.3, 0.3, 0.4])[i]
                else:
                    tp_qty = quantity * self.config.get("tp_percentages", [0.3, 0.3, 0.4])[i]
                
                tp_side = "Sell" if direction == "long" else "Buy"
                tp_params = {
                    "category": "spot",
                    "symbol": symbol,
                    "side": tp_side,
                    "orderType": "LimitMaker",
                    "qty": str(tp_qty),
                    "price": str(tp_price),
                    "timeInForce": "GTC"
                }
                
                self._make_request("POST", endpoint, tp_params, auth=True)
        
        # Store position details
        position_id = f"{symbol}_{direction}_{int(time.time())}"
        
        self.positions[position_id] = {
            "id": position_id,
            "symbol": symbol,
            "direction": direction,
            "entry_price": price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profits": tp_levels.copy() if tp_levels else [],
            "order_id": order_id,
            "entry_time": int(time.time() * 1000),
            "tp1_hit": False,
            "moved_to_be": False
        }
        
        return {
            "success": True,
            "trade_id": position_id,
            "order_id": order_id,
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "entry_price": price
        }
    
    def close_position(self, symbol, position_id):
        """
        Close an existing position
        
        Args:
            symbol (str): Symbol to trade
            position_id (str): Position ID to close
            
        Returns:
            dict: Trade result with status and details
        """
        # Check if position exists
        if position_id not in self.positions:
            self.logger.error(f"Position {position_id} not found")
            return {"success": False, "error": "Position not found"}
            
        position = self.positions[position_id]
        
        # Get current price
        exit_price = self.get_market_price(symbol)
        
        # Handle simulated positions
        if self.simulation_mode:
            # Calculate P&L
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            
            if position["direction"] == "long":
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            # Update simulated balance
            quote_currency = self.config.get("quote_currency", "USDT")
            if quote_currency in self.balance:
                position_size = position.get("position_size", quantity * entry_price)
                self.balance[quote_currency]["free"] += position_size + pnl
                self.balance[quote_currency]["locked"] -= position_size
                self.balance[quote_currency]["total"] = self.balance[quote_currency]["free"] + self.balance[quote_currency]["locked"]
                self.simulated_balance = self.balance[quote_currency]["total"]
            
            # Remove position from tracked positions
            del self.positions[position_id]
            
            self.logger.info(f"Closed simulated position {position_id}: PnL = {pnl}")
            
            return {
                "success": True,
                "trade_id": position_id,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": "manual_close"
            }
        
        # For futures
        if self.config.get("trade_futures", False):
            # Create order to close position
            endpoint = "/v5/order/create"
            params = {
                "category": "linear",
                "symbol": symbol,
                "side": "Sell" if position["direction"] == "long" else "Buy",
                "orderType": "Market",
                "qty": str(position["quantity"]),
                "timeInForce": "GTC",
                "reduceOnly": True
            }
            
            result = self._make_request("POST", endpoint, params, auth=True)
            
            if not result:
                return {"success": False, "error": "Order creation failed"}
                
            order_id = result.get("orderId")
            
            if not order_id:
                return {"success": False, "error": "No order ID returned"}
        else:
            # For spot, create a simple market order
            endpoint = "/v5/order/create"
            params = {
                "category": "spot",
                "symbol": symbol,
                "side": "Sell" if position["direction"] == "long" else "Buy",
                "orderType": "Market",
                "qty": str(position["quantity"]),
                "timeInForce": "GTC"
            }
            
            result = self._make_request("POST", endpoint, params, auth=True)
            
            if not result:
                return {"success": False, "error": "Order creation failed"}
                
            order_id = result.get("orderId")
            
            if not order_id:
                return {"success": False, "error": "No order ID returned"}
            
            # Cancel any remaining TP or SL orders
            self._cancel_all_orders(symbol)
        
        # Calculate P&L
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        
        if position["direction"] == "long":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Remove position from tracked positions
        del self.positions[position_id]
        
        return {
            "success": True,
            "trade_id": position_id,
            "exit_price": exit_price,
            "pnl": pnl,
            "exit_reason": "manual_close"
        }
    
    def update_stop_loss(self, symbol, direction, position_id, new_stop_loss):
        """
        Update stop loss for an existing position
        
        Args:
            symbol (str): Symbol to trade
            direction (str): "long" or "short"
            position_id (str): Position ID to update
            new_stop_loss (float): New stop loss price
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if position exists
        if position_id not in self.positions:
            self.logger.error(f"Position {position_id} not found")
            return False
            
        position = self.positions[position_id]
        
        # Handle simulated positions
        if self.simulation_mode:
            # Update position stop loss
            self.positions[position_id]["stop_loss"] = new_stop_loss
            self.positions[position_id]["moved_to_be"] = True
            self.logger.info(f"Updated simulated stop loss for {position_id} to {new_stop_loss}")
            return True
        
        # For futures
        if self.config.get("trade_futures", False):
            endpoint = "/v5/position/trading-stop"
            params = {
                "category": "linear",
                "symbol": symbol,
                "stopLoss": str(new_stop_loss)
            }
            
            result = self._make_request("POST", endpoint, params, auth=True)
            
            if not result:
                return False
        else:
            # For spot, cancel existing stop loss and create a new one
            # First, cancel all stop loss orders
            self._cancel_stop_loss_orders(symbol, direction)
            
            # Then create a new stop loss order
            endpoint = "/v5/order/create"
            params = {
                "category": "spot",
                "symbol": symbol,
                "side": "Sell" if direction == "long" else "Buy",
                "orderType": "StopMarket" if self.config.get("use_market_stops", True) else "StopLimit",
                "qty": str(position["quantity"]),
                "stopPrice": str(new_stop_loss),
                "timeInForce": "GTC"
            }
            
            if params["orderType"] == "StopLimit":
                # For stop limit, we need a price
                params["price"] = str(new_stop_loss * 0.99 if direction == "long" else new_stop_loss * 1.01)
            
            result = self._make_request("POST", endpoint, params, auth=True)
            
            if not result:
                return False
        
        # Update position
        self.positions[position_id]["stop_loss"] = new_stop_loss
        self.positions[position_id]["moved_to_be"] = True
        
        return True
    
    def _cancel_all_orders(self, symbol):
        """
        Cancel all orders for a symbol
        
        Args:
            symbol (str): Symbol to cancel orders for
            
        Returns:
            bool: True if successful, False otherwise
        """
        endpoint = "/v5/order/cancel-all"
        params = {
            "category": "spot" if not self.config.get("trade_futures", False) else "linear",
            "symbol": symbol
        }
        
        result = self._make_request("POST", endpoint, params, auth=True)
        
        return result is not None
    
    def _cancel_stop_loss_orders(self, symbol, direction):
        """
        Cancel stop loss orders for a symbol
        
        Args:
            symbol (str): Symbol to cancel orders for
            direction (str): "long" or "short"
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get open orders
        endpoint = "/v5/order/realtime"
        params = {
            "category": "spot",
            "symbol": symbol
        }
        
        orders = self._make_request("GET", endpoint, params, auth=True)
        
        if not orders or "list" not in orders:
            return False
        
        # Find stop loss orders
        sl_side = "Sell" if direction == "long" else "Buy"
        sl_orders = []
        
        for order in orders["list"]:
            if order.get("side") == sl_side and order.get("orderType") in ["StopMarket", "StopLimit"]:
                sl_orders.append(order.get("orderId"))
        
        # Cancel stop loss orders
        for order_id in sl_orders:
            cancel_params = {
                "category": "spot",
                "symbol": symbol,
                "orderId": order_id
            }
            
            self._make_request("POST", "/v5/order/cancel", cancel_params, auth=True)
        
        return True
    
    def close(self):
        """Close connections and perform cleanup"""
        self.ws_connected = False
        
        if self.ws_market:
            self.ws_market.close()
            
        if self.ws_private:
            self.ws_private.close()
            
        self.session.close()
        
        self.logger.info("ByBit exchange connection closed")
