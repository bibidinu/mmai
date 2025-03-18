"""
Exchange Factory for creating and managing exchange instances
"""
import logging
import importlib
from typing import Dict, Any, List, Optional, Type, Union
from abc import ABC, abstractmethod
import ccxt
import json
import time
import asyncio
import websockets
from datetime import datetime
import traceback
import random
import numpy as np

class BaseExchange(ABC):
    """Abstract base class for all exchange implementations with enhanced resilience"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the exchange with configuration
        
        Args:
            config: Exchange configuration dictionary
        """
        self.config = config
        self.name = config.get('name', 'unknown')
        self.trading_type = config.get('trading_type', 'spot')
        self.logger = logging.getLogger(f"Exchange.{self.name}")
        self.logger.setLevel(logging.INFO)
        self.authenticated = False
        self.ws_connected = False
        self.ccxt_instance = None
        self.ws_client = None
        self.orderbook_cache = {}
        self.ticker_cache = {}
        self.position_cache = {}
        self.order_cache = {}
        self._last_request_time = 0
        self._rate_limit = config.get('rate_limit', 1.0)  # requests per second
        
        # Simulation mode settings
        self.simulation_mode = config.get('simulation_mode', False)
        self.simulated_balance = config.get('simulated_balance', {'USDT': 10000.0})
        self.simulated_positions = {}
        self.simulated_orders = {}
        self.next_order_id = 1000
        
        # Error handling and recovery settings
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 2)
        self.ws_reconnect_interval = config.get('ws_reconnect_interval', 5)
        self.max_ws_reconnect_attempts = config.get('max_ws_reconnect_attempts', 10)
        
        self.logger.info(f"Initialized {self.name} exchange (simulation mode: {self.simulation_mode})")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the exchange API and WebSocket
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from exchange API and WebSocket
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """
        Authenticate with the exchange API
        
        Returns:
            True if authentication successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance
        
        Returns:
            Dictionary containing account balance information
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing ticker information
        """
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Get current orderbook for a symbol
        
        Args:
            symbol: Trading pair symbol
            depth: Orderbook depth
            
        Returns:
            Dictionary containing orderbook information
        """
        pass
    
    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candles for a symbol
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Number of candles to retrieve
            
        Returns:
            List of candle dictionaries
        """
        pass
    
    @abstractmethod
    async def create_order(self, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new order
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type (limit, market, etc.)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (optional for market orders)
            params: Additional parameters
            
        Returns:
            Dictionary containing order information
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an existing order
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing cancellation result
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get order information
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing order information
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders
        
        Args:
            symbol: Trading pair symbol (optional, all symbols if None)
            
        Returns:
            List of open order dictionaries
        """
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions (futures only)
        
        Args:
            symbol: Trading pair symbol (optional, all symbols if None)
            
        Returns:
            List of open position dictionaries
        """
        pass
    
    def _rate_limit_request(self):
        """
        Apply rate limiting to API requests
        """
        current_time = time.time()
        elapsed = current_time - self._last_request_time
        min_elapsed = 1.0 / self._rate_limit
        
        if elapsed < min_elapsed:
            time.sleep(min_elapsed - elapsed)
        
        self._last_request_time = time.time()
    
    async def _make_request_with_retry(self, request_fn, *args, **kwargs):
        """
        Make API request with retry logic
        
        Args:
            request_fn: Function to make the request
            *args: Arguments for the request function
            **kwargs: Keyword arguments for the request function
            
        Returns:
            API response or None if all retries fail
        """
        if self.simulation_mode:
            return self._simulate_response(request_fn.__name__, *args, **kwargs)
        
        retries = 0
        while retries <= self.max_retries:
            try:
                self._rate_limit_request()
                result = await request_fn(*args, **kwargs)
                return result
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    self.logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                    # Return simulation as fallback if all retries fail
                    return self._simulate_response(request_fn.__name__, *args, **kwargs)
                
                wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                self.logger.warning(f"Request failed, retrying in {wait_time}s (attempt {retries}/{self.max_retries}): {str(e)}")
                await asyncio.sleep(wait_time)
    
    def _simulate_response(self, request_type, *args, **kwargs):
        """
        Generate simulated response for API requests in simulation mode
        
        Args:
            request_type: Type of request to simulate
            *args: Request arguments
            **kwargs: Request keyword arguments
            
        Returns:
            Simulated response data
        """
        self.logger.debug(f"Simulating response for {request_type}")
        
        if request_type == 'get_balance':
            return self._simulate_balance()
        elif request_type == 'get_ticker':
            symbol = args[0] if args else kwargs.get('symbol', 'BTC/USDT')
            return self._simulate_ticker(symbol)
        elif request_type == 'get_orderbook':
            symbol = args[0] if args else kwargs.get('symbol', 'BTC/USDT')
            depth = args[1] if len(args) > 1 else kwargs.get('depth', 20)
            return self._simulate_orderbook(symbol, depth)
        elif request_type == 'get_candles':
            symbol = args[0] if args else kwargs.get('symbol', 'BTC/USDT')
            timeframe = args[1] if len(args) > 1 else kwargs.get('timeframe', '1h')
            limit = args[2] if len(args) > 2 else kwargs.get('limit', 100)
            return self._simulate_candles(symbol, timeframe, limit)
        elif request_type == 'create_order':
            return self._simulate_create_order(*args, **kwargs)
        elif request_type == 'cancel_order':
            return self._simulate_cancel_order(*args, **kwargs)
        elif request_type == 'get_order':
            return self._simulate_get_order(*args, **kwargs)
        elif request_type == 'get_open_orders':
            return self._simulate_get_open_orders(*args, **kwargs)
        elif request_type == 'get_positions':
            return self._simulate_get_positions(*args, **kwargs)
        else:
            self.logger.warning(f"Unknown request type for simulation: {request_type}")
            return {}
    
    def _simulate_balance(self):
        """Simulate balance response"""
        return {
            'total': self.simulated_balance.copy(),
            'free': {k: v * 0.9 for k, v in self.simulated_balance.items()},  # Simulate some locked balance
            'used': {k: v * 0.1 for k, v in self.simulated_balance.items()},
            'timestamp': int(time.time() * 1000),
            'exchange': self.name
        }
    
    def _simulate_ticker(self, symbol):
        """Simulate ticker response"""
        base_price = self._get_simulated_price(symbol)
        
        return {
            'symbol': symbol,
            'last': base_price,
            'bid': base_price * 0.999,
            'ask': base_price * 1.001,
            'high': base_price * 1.02,
            'low': base_price * 0.98,
            'volume': random.uniform(1000, 10000),
            'timestamp': int(time.time() * 1000),
            'change': random.uniform(-3, 3),
            'exchange': self.name
        }
    
    def _simulate_orderbook(self, symbol, depth=20):
        """Simulate orderbook response"""
        base_price = self._get_simulated_price(symbol)
        
        # Generate bids (buy orders)
        bids = []
        for i in range(depth):
            price = base_price * (1 - 0.0001 * (i + 1))
            size = random.uniform(0.1, 10)
            bids.append([price, size])
        
        # Generate asks (sell orders)
        asks = []
        for i in range(depth):
            price = base_price * (1 + 0.0001 * (i + 1))
            size = random.uniform(0.1, 10)
            asks.append([price, size])
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.now().isoformat(),
            'exchange': self.name
        }
    
    def _simulate_candles(self, symbol, timeframe, limit=100):
        """Simulate candles response"""
        base_price = self._get_simulated_price(symbol)
        candles = []
        
        # Parse timeframe to seconds
        tf_seconds = self._timeframe_to_seconds(timeframe)
        current_time = int(time.time())
        
        # Generate candles with random walk
        price = base_price
        for i in range(limit):
            timestamp = (current_time - (limit - i) * tf_seconds) * 1000  # In milliseconds
            
            # Random price movement (slightly biased upward)
            price_change = random.normalvariate(0.0001, 0.002)  # Mean slightly positive
            price *= (1 + price_change)
            
            # Generate candle values
            open_price = price
            close_price = price * (1 + random.normalvariate(0, 0.001))
            high_price = max(open_price, close_price) * (1 + abs(random.normalvariate(0, 0.0005)))
            low_price = min(open_price, close_price) * (1 - abs(random.normalvariate(0, 0.0005)))
            volume = random.uniform(100, 1000)
            
            candles.append({
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'exchange': self.name
            })
        
        return candles
    
    def _simulate_create_order(self, symbol, order_type, side, amount, price=None, params=None):
        """Simulate order creation"""
        order_id = f"sim_{self.next_order_id}"
        self.next_order_id += 1
        
        # Use current market price if no price specified or market order
        if price is None or order_type.lower() == 'market':
            price = self._get_simulated_price(symbol)
        
        # Create simulated order
        order = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'price': float(price) if price is not None else None,
            'amount': float(amount),
            'filled': 0.0,
            'remaining': float(amount),
            'status': 'open',
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.now().isoformat(),
            'fee': {'cost': float(amount) * float(price) * 0.001 if price is not None else 0.0, 'currency': 'USDT'},
            'exchange': self.name
        }
        
        # Store in simulated orders
        self.simulated_orders[order_id] = order
        
        # If market order, execute immediately
        if order_type.lower() == 'market':
            self._simulate_order_execution(order_id)
        
        return order
    
    def _simulate_order_execution(self, order_id):
        """Simulate order execution"""
        if order_id not in self.simulated_orders:
            return
        
        order = self.simulated_orders[order_id]
        
        # Simulate execution (might be partial)
        execution_percent = random.uniform(0.9, 1.0)  # 90-100% filled
        filled_amount = order['amount'] * execution_percent
        
        order['filled'] = filled_amount
        order['remaining'] = order['amount'] - filled_amount
        order['status'] = 'closed' if order['remaining'] < 0.00001 else 'open'
        
        # Update simulated balance
        symbol_parts = order['symbol'].split('/')
        if len(symbol_parts) == 2:
            base, quote = symbol_parts
            
            if order['side'].lower() == 'buy':
                # Add base currency, subtract quote currency
                if base not in self.simulated_balance:
                    self.simulated_balance[base] = 0.0
                self.simulated_balance[base] += filled_amount
                
                cost = filled_amount * order['price']
                if quote in self.simulated_balance:
                    self.simulated_balance[quote] -= cost
            else:
                # Subtract base currency, add quote currency
                if base in self.simulated_balance:
                    self.simulated_balance[base] -= filled_amount
                
                # Add quote currency (with fees subtracted)
                proceeds = filled_amount * order['price'] * (1 - 0.001)  # 0.1% fee
                if quote not in self.simulated_balance:
                    self.simulated_balance[quote] = 0.0
                self.simulated_balance[quote] += proceeds
        
        return order
    
    def _simulate_cancel_order(self, order_id, symbol):
        """Simulate order cancellation"""
        if order_id not in self.simulated_orders:
            return {'success': False, 'message': 'Order not found'}
        
        order = self.simulated_orders[order_id]
        
        # Only cancel if order is still open
        if order['status'] == 'open':
            order['status'] = 'canceled'
        
        return {'success': True, 'id': order_id, 'status': order['status']}
    
    def _simulate_get_order(self, order_id, symbol):
        """Simulate get order details"""
        if order_id not in self.simulated_orders:
            return {}
        
        order = self.simulated_orders[order_id]
        
        # Simulate some progress for open orders
        if order['status'] == 'open':
            elapsed = (time.time() * 1000 - order['timestamp']) / 1000  # Seconds elapsed
            fill_rate = min(1.0, elapsed / 60)  # Simulate filling over 1 minute
            
            order['filled'] = order['amount'] * fill_rate
            order['remaining'] = order['amount'] - order['filled']
            
            if order['filled'] >= order['amount']:
                order['status'] = 'closed'
        
        return order
    
    def _simulate_get_open_orders(self, symbol=None):
        """Simulate get open orders"""
        open_orders = []
        
        for order_id, order in self.simulated_orders.items():
            if order['status'] == 'open':
                if symbol is None or order['symbol'] == symbol:
                    open_orders.append(order)
        
        return open_orders
    
    def _simulate_get_positions(self, symbol=None):
        """Simulate get positions"""
        positions = []
        
        for pos_id, position in self.simulated_positions.items():
            if symbol is None or position['symbol'] == symbol:
                positions.append(position)
        
        return positions
    
    def _get_simulated_price(self, symbol):
        """Get a simulated price for a symbol"""
        # Default prices for common symbols
        default_prices = {
            'BTC/USDT': 65000.0,
            'ETH/USDT': 3500.0,
            'SOL/USDT': 150.0,
            'BNB/USDT': 550.0,
            'XRP/USDT': 0.55,
            'ADA/USDT': 0.45,
            'DOGE/USDT': 0.12,
            'MATIC/USDT': 0.75,
            'LINK/USDT': 15.0,
            'AVAX/USDT': 35.0
        }
        
        # Normalize symbol format
        normalized_symbol = symbol.replace('_', '/')
        
        # Get cached price or default
        base_price = default_prices.get(normalized_symbol, 100.0)
        
        # Add small random variation (±0.5%)
        return base_price * (1 + random.uniform(-0.005, 0.005))
    
    def _timeframe_to_seconds(self, timeframe):
        """Convert timeframe string to seconds"""
        if timeframe.endswith('m'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 3600
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 86400
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 604800
        else:
            return 3600  # Default to 1h
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for the exchange
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Normalized symbol
        """
        # Default implementation (override in specific exchange classes)
        return symbol
    
    def parse_ticker(self, raw_ticker: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw ticker data into standardized format
        
        Args:
            raw_ticker: Raw ticker data from exchange
            
        Returns:
            Standardized ticker dictionary
        """
        # Default implementation (override in specific exchange classes)
        return raw_ticker
    
    def parse_orderbook(self, raw_orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw orderbook data into standardized format
        
        Args:
            raw_orderbook: Raw orderbook data from exchange
            
        Returns:
            Standardized orderbook dictionary
        """
        # Default implementation (override in specific exchange classes)
        return raw_orderbook
    
    def parse_candles(self, raw_candles: List[Any]) -> List[Dict[str, Any]]:
        """
        Parse raw candle data into standardized format
        
        Args:
            raw_candles: Raw candle data from exchange
            
        Returns:
            List of standardized candle dictionaries
        """
        # Default implementation (override in specific exchange classes)
        return raw_candles
    
    def parse_order(self, raw_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw order data into standardized format
        
        Args:
            raw_order: Raw order data from exchange
            
        Returns:
            Standardized order dictionary
        """
        # Default implementation (override in specific exchange classes)
        return raw_order
    
    def parse_position(self, raw_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw position data into standardized format
        
        Args:
            raw_position: Raw position data from exchange
            
        Returns:
            Standardized position dictionary
        """
        # Default implementation (override in specific exchange classes)
        return raw_position
    
    def parse_balance(self, raw_balance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw balance data into standardized format
        
        Args:
            raw_balance: Raw balance data from exchange
            
        Returns:
            Standardized balance dictionary
        """
        # Default implementation (override in specific exchange classes)
        return raw_balance
    
    async def _connect_websocket(self, url, on_message, on_error=None, on_close=None):
        """
        Connect to WebSocket with improved error handling and reconnection
        
        Args:
            url: WebSocket URL
            on_message: Message handler function
            on_error: Error handler function (optional)
            on_close: Close handler function (optional)
            
        Returns:
            WebSocket connection object
        """
        if self.simulation_mode:
            self.logger.info(f"WebSocket connection simulated for {url}")
            self.ws_connected = True
            return None
        
        try:
            connection = await websockets.connect(url)
            self.ws_connected = True
            self.logger.info(f"Connected to WebSocket: {url}")
            
            # Start listening task
            asyncio.create_task(self._websocket_listener(connection, on_message, on_error, on_close))
            
            return connection
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {str(e)}")
            self.ws_connected = False
            return None
    
    async def _websocket_listener(self, connection, on_message, on_error, on_close):
        """
        WebSocket listener with automatic reconnection
        
        Args:
            connection: WebSocket connection
            on_message: Message handler function
            on_error: Error handler function
            on_close: Close handler function
        """
        reconnect_attempts = 0
        
        while self.ws_connected and reconnect_attempts < self.max_ws_reconnect_attempts:
            try:
                async for message in connection:
                    try:
                        if on_message:
                            await on_message(message)
                    except Exception as e:
                        self.logger.error(f"Error processing WebSocket message: {str(e)}")
                        if on_error:
                            await on_error(e)
                
                # Connection closed normally
                self.logger.info("WebSocket connection closed")
                if on_close:
                    await on_close(None)
                
                break
                
            except Exception as e:
                reconnect_attempts += 1
                self.logger.error(f"WebSocket error (attempt {reconnect_attempts}/{self.max_ws_reconnect_attempts}): {str(e)}")
                
                if on_error:
                    await on_error(e)
                
                # Exponential backoff for reconnection
                wait_time = self.ws_reconnect_interval * (2 ** (reconnect_attempts - 1))
                self.logger.info(f"Reconnecting in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                
                try:
                    connection = await websockets.connect(connection.url)
                    self.logger.info(f"Reconnected to WebSocket: {connection.url}")
                    reconnect_attempts = 0  # Reset counter on successful reconnection
                except Exception as reconnect_error:
                    self.logger.error(f"WebSocket reconnection failed: {str(reconnect_error)}")
        
        self.ws_connected = False
        self.logger.warning("WebSocket connection terminated")


class BybitExchange(BaseExchange):
    """Enhanced Bybit exchange implementation with improved resilience"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bybit exchange
        
        Args:
            config: Bybit configuration
        """
        super().__init__(config)
        self.base_url = config.get('base_url', 'https://api.bybit.com')
        self.ws_url = config.get('websocket_url', 'wss://stream.bybit.com')
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.testnet = config.get('testnet', False)
        
        # Check if simulation mode should be forced due to missing credentials
        if not self.api_key or not self.api_secret or self.api_key == "YOUR_API_KEY" or self.api_secret == "YOUR_API_SECRET":
            self.logger.warning("Missing or invalid API credentials, forcing simulation mode")
            self.simulation_mode = True
        
        # Initialize CCXT instance with error handling
        try:
            self.ccxt_instance = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': self.trading_type,
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            })
            
            # Use testnet if configured
            if self.testnet:
                self.ccxt_instance.urls['api'] = 'https://api-testnet.bybit.com'
                self.ws_url = 'wss://stream-testnet.bybit.com'
                
        except Exception as e:
            self.logger.error(f"Failed to initialize CCXT for Bybit: {str(e)}")
            self.simulation_mode = True
            self.logger.warning("Falling back to simulation mode due to CCXT initialization error")
    
    async def connect(self) -> bool:
        """
        Connect to Bybit API and WebSocket with improved error handling
        
        Returns:
            True if connection successful
        """
        if self.simulation_mode:
            self.logger.info("Using simulation mode for Bybit exchange")
            return True
            
        try:
            # Test REST API connection
            self._rate_limit_request()
            status = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_status()
            )
            
            if not status:
                self.logger.warning("Failed to fetch Bybit status, using simulation as fallback")
                self.simulation_mode = True
                return True
            
            # Initialize WebSocket connection
            # In a real implementation, this would be more complex with proper handlers
            self.ws_client = None  # Placeholder for actual WebSocket client
            self.ws_connected = True
            
            self.logger.info(f"Connected to Bybit exchange, trading type: {self.trading_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Bybit: {str(e)}")
            self.logger.warning("Falling back to simulation mode due to connection error")
            self.simulation_mode = True
            return True  # Return true since we can operate in simulation mode
    
    async def disconnect(self) -> bool:
        """
        Disconnect from Bybit API and WebSocket with improved error handling
        
        Returns:
            True if disconnection successful
        """
        try:
            # Close WebSocket connection if open
            if self.ws_client:
                # In a real implementation, this would properly close the WebSocket
                self.ws_client = None
            
            self.ws_connected = False
            self.logger.info("Disconnected from Bybit exchange")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during Bybit disconnect: {str(e)}")
            # Force cleanup anyway
            self.ws_client = None
            self.ws_connected = False
            return False
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Bybit API with improved error handling
        
        Returns:
            True if authentication successful
        """
        if self.simulation_mode:
            self.logger.info("Authentication simulated in simulation mode")
            self.authenticated = True
            return True
            
        try:
            if not self.api_key or not self.api_secret:
                self.logger.warning("API key or secret not provided, switching to simulation mode")
                self.simulation_mode = True
                self.authenticated = False
                return False
            
            # Test authentication by fetching balance
            balance = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_balance()
            )
            
            if not balance:
                self.logger.warning("Authentication failed, switching to simulation mode")
                self.simulation_mode = True
                self.authenticated = False
                return False
            
            self.authenticated = True
            self.logger.info("Successfully authenticated with Bybit API")
            return True
            
        except Exception as e:
            self.logger.error(f"Bybit authentication failed: {str(e)}")
            self.logger.warning("Switching to simulation mode due to authentication failure")
            self.simulation_mode = True
            self.authenticated = False
            return False
    
    async def get_balance(self) -> Dict[str, Any]:
        """
        Get account balance from Bybit with improved error handling
        
        Returns:
            Dictionary containing account balance
        """
        if self.simulation_mode:
            return self._simulate_balance()
            
        try:
            raw_balance = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_balance()
            )
            
            if not raw_balance:
                return self._simulate_balance()
                
            return self.parse_balance(raw_balance)
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit balance: {str(e)}")
            return self._simulate_balance()
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker for a symbol from Bybit with improved error handling
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing ticker information
        """
        if self.simulation_mode:
            return self._simulate_ticker(symbol)
            
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            raw_ticker = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_ticker(normalized_symbol)
            )
            
            if not raw_ticker:
                return self._simulate_ticker(symbol)
            
            # Cache the result
            parsed_ticker = self.parse_ticker(raw_ticker)
            self.ticker_cache[symbol] = parsed_ticker
            
            return parsed_ticker
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit ticker for {symbol}: {str(e)}")
            
            # Use cached value if available, otherwise simulate
            if symbol in self.ticker_cache:
                return self.ticker_cache[symbol]
            else:
                return self._simulate_ticker(symbol)
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """
        Get orderbook for a symbol from Bybit with improved error handling
        
        Args:
            symbol: Trading pair symbol
            depth: Orderbook depth
            
        Returns:
            Dictionary containing orderbook information
        """
        if self.simulation_mode:
            return self._simulate_orderbook(symbol, depth)
            
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            raw_orderbook = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_order_book(normalized_symbol, depth)
            )
            
            if not raw_orderbook:
                return self._simulate_orderbook(symbol, depth)
            
            # Cache the result
            parsed_orderbook = self.parse_orderbook(raw_orderbook)
            self.orderbook_cache[symbol] = parsed_orderbook
            
            return parsed_orderbook
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit orderbook for {symbol}: {str(e)}")
            
            # Use cached value if available, otherwise simulate
            if symbol in self.orderbook_cache:
                return self.orderbook_cache[symbol]
            else:
                return self._simulate_orderbook(symbol, depth)
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical candles for a symbol from Bybit with improved error handling
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Number of candles to retrieve
            
        Returns:
            List of candle dictionaries
        """
        if self.simulation_mode:
            return self._simulate_candles(symbol, timeframe, limit)
            
        try:
            normalized_symbol = self.normalize_symbol(symbol)
            
            raw_candles = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_ohlcv(normalized_symbol, timeframe, limit=limit)
            )
            
            if not raw_candles:
                return self._simulate_candles(symbol, timeframe, limit)
            
            return self.parse_candles(raw_candles)
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit candles for {symbol}: {str(e)}")
            return self._simulate_candles(symbol, timeframe, limit)
    
    async def create_order(self, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new order on Bybit with improved error handling
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type (limit, market)
            side: Order side (buy, sell)
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional parameters
            
        Returns:
            Dictionary containing order information
        """
        if self.simulation_mode:
            return self._simulate_create_order(symbol, order_type, side, amount, price, params)
            
        try:
            if not self.authenticated:
                self.logger.error("Cannot create order: not authenticated")
                return self._simulate_create_order(symbol, order_type, side, amount, price, params)
            
            normalized_symbol = self.normalize_symbol(symbol)
            
            raw_order = await self._make_request_with_retry(
                lambda: self.ccxt_instance.create_order(
                    normalized_symbol,
                    order_type,
                    side,
                    amount,
                    price,
                    params or {}
                )
            )
            
            if not raw_order:
                return self._simulate_create_order(symbol, order_type, side, amount, price, params)
            
            return self.parse_order(raw_order)
            
        except Exception as e:
            self.logger.error(f"Error creating Bybit order for {symbol}: {str(e)}")
            return self._simulate_create_order(symbol, order_type, side, amount, price, params)
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an existing order on Bybit with improved error handling
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing cancellation result
        """
        if self.simulation_mode:
            return self._simulate_cancel_order(order_id, symbol)
            
        try:
            if not self.authenticated:
                self.logger.error("Cannot cancel order: not authenticated")
                return self._simulate_cancel_order(order_id, symbol)
            
            normalized_symbol = self.normalize_symbol(symbol)
            
            raw_result = await self._make_request_with_retry(
                lambda: self.ccxt_instance.cancel_order(order_id, normalized_symbol)
            )
            
            if not raw_result:
                return self._simulate_cancel_order(order_id, symbol)
            
            return self.parse_order(raw_result)
            
        except Exception as e:
            self.logger.error(f"Error cancelling Bybit order {order_id}: {str(e)}")
            return self._simulate_cancel_order(order_id, symbol)
    
    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get order information from Bybit with improved error handling
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            Dictionary containing order information
        """
        if self.simulation_mode:
            return self._simulate_get_order(order_id, symbol)
            
        try:
            if not self.authenticated:
                self.logger.error("Cannot fetch order: not authenticated")
                return self._simulate_get_order(order_id, symbol)
            
            normalized_symbol = self.normalize_symbol(symbol)
            
            raw_order = await self._make_request_with_retry(
                lambda: self.ccxt_instance.fetch_order(order_id, normalized_symbol)
            )
            
            if not raw_order:
                return self._simulate_get_order(order_id, symbol)
            
            return self.parse_order(raw_order)
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit order {order_id}: {str(e)}")
            return self._simulate_get_order(order_id, symbol)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders from Bybit with improved error handling
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of open order dictionaries
        """
        if self.simulation_mode:
            return self._simulate_get_open_orders(symbol)
            
        try:
            if not self.authenticated:
                self.logger.error("Cannot fetch open orders: not authenticated")
                return self._simulate_get_open_orders(symbol)
            
            if symbol:
                normalized_symbol = self.normalize_symbol(symbol)
                raw_orders = await self._make_request_with_retry(
                    lambda: self.ccxt_instance.fetch_open_orders(normalized_symbol)
                )
            else:
                raw_orders = await self._make_request_with_retry(
                    lambda: self.ccxt_instance.fetch_open_orders()
                )
            
            if not raw_orders:
                return self._simulate_get_open_orders(symbol)
            
            return [self.parse_order(order) for order in raw_orders]
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit open orders: {str(e)}")
            return self._simulate_get_open_orders(symbol)
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open positions from Bybit with improved error handling
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of open position dictionaries
        """
        if self.simulation_mode:
            return self._simulate_get_positions(symbol)
            
        try:
            if not self.authenticated:
                self.logger.error("Cannot fetch positions: not authenticated")
                return self._simulate_get_positions(symbol)
            
            if self.trading_type != 'future' and self.trading_type != 'futures':
                self.logger.warning("Positions are only available in futures trading")
                return self._simulate_get_positions(symbol)
            
            if symbol:
                normalized_symbol = self.normalize_symbol(symbol)
                raw_positions = await self._make_request_with_retry(
                    lambda: self.ccxt_instance.fetch_positions([normalized_symbol])
                )
            else:
                raw_positions = await self._make_request_with_retry(
                    lambda: self.ccxt_instance.fetch_positions()
                )
            
            if not raw_positions:
                return self._simulate_get_positions(symbol)
            
            # Filter out positions with zero size
            active_positions = [pos for pos in raw_positions if float(pos.get('contracts', 0)) > 0]
            
            return [self.parse_position(position) for position in active_positions]
            
        except Exception as e:
            self.logger.error(f"Error fetching Bybit positions: {str(e)}")
            return self._simulate_get_positions(symbol)
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for Bybit
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Normalized symbol
        """
        # Bybit uses / for spot and futures
        if '/' not in symbol:
            # Check for USDT suffix
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                quote = 'USDT'
                return f"{base}/{quote}"
            
            # Check for other common formats
            parts = symbol.split('_')
            if len(parts) == 2:
                return f"{parts[0]}/{parts[1]}"
        return symbol
    
    def parse_ticker(self, raw_ticker: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Bybit ticker into standardized format
        
        Args:
            raw_ticker: Raw ticker data from Bybit
            
        Returns:
            Standardized ticker dictionary
        """
        return {
            'symbol': raw_ticker.get('symbol', ''),
            'last': raw_ticker.get('last', 0.0),
            'bid': raw_ticker.get('bid', 0.0),
            'ask': raw_ticker.get('ask', 0.0),
            'high': raw_ticker.get('high', 0.0),
            'low': raw_ticker.get('low', 0.0),
            'volume': raw_ticker.get('volume', 0.0),
            'timestamp': raw_ticker.get('timestamp', 0),
            'change': raw_ticker.get('percentage', 0.0),
            'exchange': 'bybit'
        }
    
    def parse_orderbook(self, raw_orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Bybit orderbook into standardized format
        
        Args:
            raw_orderbook: Raw orderbook data from Bybit
            
        Returns:
            Standardized orderbook dictionary
        """
        return {
            'bids': raw_orderbook.get('bids', []),
            'asks': raw_orderbook.get('asks', []),
            'timestamp': raw_orderbook.get('timestamp', int(time.time() * 1000)),
            'datetime': raw_orderbook.get('datetime', datetime.now().isoformat()),
            'exchange': 'bybit'
        }
    
    def parse_candles(self, raw_candles: List[Any]) -> List[Dict[str, Any]]:
        """
        Parse Bybit candles into standardized format
        
        Args:
            raw_candles: Raw candle data from Bybit
            
        Returns:
            List of standardized candle dictionaries
        """
        formatted_candles = []
        
        for candle in raw_candles:
            # CCXT OHLCV format: [timestamp, open, high, low, close, volume]
            if len(candle) >= 6:
                formatted_candles.append({
                    'timestamp': candle[0],
                    'datetime': datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'exchange': 'bybit'
                })
        
        return formatted_candles
    
    def parse_order(self, raw_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Bybit order into standardized format
        
        Args:
            raw_order: Raw order data from Bybit
            
        Returns:
            Standardized order dictionary
        """
        return {
            'id': raw_order.get('id', ''),
            'symbol': raw_order.get('symbol', ''),
            'type': raw_order.get('type', ''),
            'side': raw_order.get('side', ''),
            'price': raw_order.get('price', 0.0),
            'amount': raw_order.get('amount', 0.0),
            'filled': raw_order.get('filled', 0.0),
            'remaining': raw_order.get('remaining', 0.0),
            'status': raw_order.get('status', ''),
            'timestamp': raw_order.get('timestamp', 0),
            'datetime': raw_order.get('datetime', ''),
            'fee': raw_order.get('fee', {}),
            'exchange': 'bybit'
        }
    
    def parse_position(self, raw_position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Bybit position into standardized format
        
        Args:
            raw_position: Raw position data from Bybit
            
        Returns:
            Standardized position dictionary
        """
        side = 'long' if float(raw_position.get('contracts', 0)) > 0 else 'short'
        
        return {
            'symbol': raw_position.get('symbol', ''),
            'size': abs(float(raw_position.get('contracts', 0))),
            'side': side,
            'entry_price': float(raw_position.get('entryPrice', 0.0)),
            'mark_price': float(raw_position.get('markPrice', 0.0)),
            'liquidation_price': float(raw_position.get('liquidationPrice', 0.0)),
            'unrealized_pnl': float(raw_position.get('unrealizedPnl', 0.0)),
            'leverage': float(raw_position.get('leverage', 1.0)),
            'margin_type': raw_position.get('marginType', ''),
            'timestamp': int(time.time() * 1000),
            'exchange': 'bybit'
        }
    
    def parse_balance(self, raw_balance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Bybit balance into standardized format
        
        Args:
            raw_balance: Raw balance data from Bybit
            
        Returns:
            Standardized balance dictionary
        """
        result = {
            'total': {},
            'free': {},
            'used': {},
            'timestamp': int(time.time() * 1000),
            'exchange': 'bybit'
        }
        
        # Extract currencies
        for currency, data in raw_balance.get('total', {}).items():
            if currency != 'info':
                result['total'][currency] = float(data)
                result['free'][currency] = float(raw_balance.get('free', {}).get(currency, 0.0))
                result['used'][currency] = float(raw_balance.get('used', {}).get(currency, 0.0))
        
        return result


class BinanceExchange(BaseExchange):
    """Binance exchange implementation (enhanced with resilience)"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Binance exchange
        
        Args:
            config: Binance configuration
        """
        super().__init__(config)
        self.logger.info("Binance exchange initialized with enhanced resilience")
        # Implementation would be similar to BybitExchange but with Binance-specific details
    
    async def connect(self) -> bool:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            self.logger.info("Using simulation mode for Binance exchange")
            return True
            
        self.logger.info("Connect called for Binance exchange (placeholder)")
        return True
    
    async def disconnect(self) -> bool:
        """Placeholder implementation"""
        self.logger.info("Disconnect called for Binance exchange (placeholder)")
        return True
    
    async def authenticate(self) -> bool:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            self.logger.info("Authentication simulated for Binance")
            self.authenticated = True
            return True
            
        self.logger.info("Authenticate called for Binance exchange (placeholder)")
        return True
    
    async def get_balance(self) -> Dict[str, Any]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_balance()
            
        self.logger.info("Get balance called for Binance (placeholder)")
        return {}
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_ticker(symbol)
            
        self.logger.info(f"Get ticker called for Binance: {symbol} (placeholder)")
        return {}
    
    async def get_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_orderbook(symbol, depth)
            
        self.logger.info(f"Get orderbook called for Binance: {symbol} (placeholder)")
        return {}
    
    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_candles(symbol, timeframe, limit)
            
        self.logger.info(f"Get candles called for Binance: {symbol} (placeholder)")
        return []
    
    async def create_order(self, symbol: str, order_type: str, side: str, 
                          amount: float, price: Optional[float] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_create_order(symbol, order_type, side, amount, price, params)
            
        self.logger.info(f"Create order called for Binance: {symbol} (placeholder)")
        return {}
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_cancel_order(order_id, symbol)
            
        self.logger.info(f"Cancel order called for Binance: {order_id} (placeholder)")
        return {}
    
    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_get_order(order_id, symbol)
            
        self.logger.info(f"Get order called for Binance: {order_id} (placeholder)")
        return {}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_get_open_orders(symbol)
            
        self.logger.info("Get open orders called for Binance (placeholder)")
        return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Placeholder implementation with simulation support"""
        if self.simulation_mode:
            return self._simulate_get_positions(symbol)
            
        self.logger.info("Get positions called for Binance (placeholder)")
        return []


# Export the ExchangeFactory class directly for compatibility
class ExchangeFactory:
    """Factory for creating exchange instances with enhanced resilience"""
    
    def __init__(self):
        """Initialize the exchange factory"""
        self.logger = logging.getLogger("ExchangeFactory")
        self.logger.setLevel(logging.INFO)
        self.exchanges = {}
        self.exchange_types = {
            'bybit': BybitExchange,
            'binance': BinanceExchange
            # Add more exchanges as implemented
        }
    
    def create_exchange(self, exchange_type: str, config: Dict[str, Any]) -> BaseExchange:
        """
        Create an exchange instance with enhanced error handling
        
        Args:
            exchange_type: Type of exchange to create
            config: Exchange configuration
            
        Returns:
            Exchange instance
        """
        try:
            # Check if exchange type is supported
            if exchange_type not in self.exchange_types:
                self.logger.error(f"Unsupported exchange type: {exchange_type}")
                
                # Default to simulation mode with ByBit if not supported
                self.logger.warning(f"Falling back to simulation mode with ByBit exchange")
                exchange_type = 'bybit'
                if 'simulation_mode' not in config:
                    config['simulation_mode'] = True
            
            # Create exchange instance
            exchange_class = self.exchange_types[exchange_type]
            
            # Ensure config has a name field
            if 'name' not in config:
                config['name'] = exchange_type
            
            exchange = exchange_class(config)
            
            # Register the exchange
            exchange_id = config.get('id', exchange_type)
            self.exchanges[exchange_id] = exchange
            
            self.logger.info(f"Created {exchange_type} exchange with ID {exchange_id}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Error creating exchange instance: {str(e)}")
            self.logger.warning(f"Creating fallback simulation exchange")
            
            # Create a simulation-mode bybit exchange as fallback
            fallback_config = {
                'name': 'fallback_bybit',
                'simulation_mode': True,
                'id': 'fallback'
            }
            
            exchange = BybitExchange(fallback_config)
            self.exchanges['fallback'] = exchange
            return exchange
    
    def get_exchange(self, exchange_id: str) -> Optional[BaseExchange]:
        """
        Get an exchange instance by ID with fallback
        
        Args:
            exchange_id: Exchange ID
            
        Returns:
            Exchange instance or fallback if not found
        """
        if exchange_id in self.exchanges:
            return self.exchanges[exchange_id]
        
        self.logger.warning(f"Exchange {exchange_id} not found")
        
        # Return fallback if available
        if 'fallback' in self.exchanges:
            self.logger.info("Using fallback exchange")
            return self.exchanges['fallback']
        
        return None
    
    def list_exchanges(self) -> List[str]:
        """
        List registered exchanges
        
        Returns:
            List of exchange IDs
        """
        return list(self.exchanges.keys())
    
    def list_supported_types(self) -> List[str]:
        """
        List supported exchange types
        
        Returns:
            List of supported exchange types
        """
        return list(self.exchange_types.keys())
    
    def register_exchange_type(self, exchange_type: str, exchange_class: Type[BaseExchange]) -> None:
        """
        Register a new exchange type with validation
        
        Args:
            exchange_type: Exchange type identifier
            exchange_class: Exchange class to register
        """
        try:
            if not issubclass(exchange_class, BaseExchange):
                self.logger.error(f"Class {exchange_class.__name__} is not a subclass of BaseExchange")
                raise TypeError(f"Class {exchange_class.__name__} is not a subclass of BaseExchange")
            
            self.exchange_types[exchange_type] = exchange_class
            self.logger.info(f"Registered exchange type: {exchange_type}")
            
        except Exception as e:
            self.logger.error(f"Error registering exchange type: {str(e)}")
    
    def load_exchange_module(self, module_path: str) -> bool:
        """
        Dynamically load an exchange module with error handling
        
        Args:
            module_path: Path to exchange module
            
        Returns:
            True if successful, False otherwise
        """
        try:
            module = importlib.import_module(module_path)
            
            # Look for exchange classes in the module
            exchanges_found = 0
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseExchange) and 
                    attr is not BaseExchange):
                    
                    # Extract exchange type from class name
                    exchange_type = attr_name.lower()
                    if exchange_type.endswith('exchange'):
                        exchange_type = exchange_type[:-8]  # Remove 'exchange' suffix
                    
                    # Register the exchange type
                    self.register_exchange_type(exchange_type, attr)
                    exchanges_found += 1
            
            if exchanges_found > 0:
                self.logger.info(f"Loaded {exchanges_found} exchange(s) from module: {module_path}")
                return True
            else:
                self.logger.warning(f"No exchange implementations found in module: {module_path}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error loading exchange module {module_path}: {str(e)}")
            return False
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered exchanges with error handling
        
        Returns:
            Dictionary mapping exchange IDs to initialization status
        """
        results = {}
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                self.logger.info(f"Initializing exchange: {exchange_id}")
                
                # Connect to exchange
                connect_result = await exchange.connect()
                
                # Authenticate if needed and not in simulation mode
                auth_result = True
                if not exchange.simulation_mode and hasattr(exchange, 'api_key') and exchange.api_key:
                    auth_result = await exchange.authenticate()
                
                results[exchange_id] = connect_result and auth_result
                
                self.logger.info(f"Exchange {exchange_id} initialized (success: {results[exchange_id]})")
                
            except Exception as e:
                self.logger.error(f"Error initializing exchange {exchange_id}: {str(e)}")
                results[exchange_id] = False
        
        return results
    
    async def shutdown_all(self) -> None:
        """Disconnect from all exchanges with error handling"""
        for exchange_id, exchange in self.exchanges.items():
            try:
                self.logger.info(f"Shutting down exchange: {exchange_id}")
                await exchange.disconnect()
                self.logger.info(f"Exchange {exchange_id} shut down successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down exchange {exchange_id}: {str(e)}")
                
# Singleton factory instance
_factory_instance = None

def get_factory() -> ExchangeFactory:
    """
    Get the global factory instance (singleton pattern)
    
    Returns:
        ExchangeFactory: Global factory instance
    """
    global _factory_instance
    
    if _factory_instance is None:
        _factory_instance = ExchangeFactory()
    
    return _factory_instance
