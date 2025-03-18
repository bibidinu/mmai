#!/usr/bin/env python3
"""
Dashboard Server Module for Crypto Trading Bot
"""
import os
import sys
import json
import time
import threading
import subprocess
import logging
import socket
from pathlib import Path
import http.server
import socketserver
import json
from urllib.parse import urlparse, parse_qs

class DashboardServer:
    def __init__(self, trading_bot, host="localhost", port=3000, frontend_dir=None):
        """
        Initialize the dashboard server
        
        Args:
            trading_bot: Reference to the CryptoTradingBot instance
            host (str): Host to bind the server to
            port (int): Port to bind the server to
            frontend_dir (str): Directory containing the frontend code
        """
        self.trading_bot = trading_bot
        self.host = host
        self.port = port
        self.server_running = False
        self.server_thread = None
        self.logger = logging.getLogger('dashboard_server')
        
        # Determine frontend directory
        if frontend_dir is None:
            # Default to frontend directory in the same directory as main.py
            root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.frontend_dir = root_dir / "frontend"
        else:
            self.frontend_dir = Path(frontend_dir)
            
        self.api_server = None
        self.api_port = port + 1  # Use port+1 for the API server
        self.api_thread = None
        
        # Ensure frontend directory exists
        if not self.frontend_dir.exists():
            self.logger.warning(f"Frontend directory {self.frontend_dir} does not exist")
        
    def is_port_in_use(self, port):
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
            
    def start_dashboard(self, blocking=False):
        """
        Start the dashboard server
        
        Args:
            blocking (bool): If True, start the server in the current thread
        """
        if self.server_running:
            self.logger.info("Dashboard server is already running")
            return
            
        # Check if ports are already in use
        if self.is_port_in_use(self.port):
            self.logger.warning(f"Port {self.port} is already in use, dashboard may already be running")
            self.server_running = True
            return
        
        if self.is_port_in_use(self.api_port):
            self.logger.warning(f"Port {self.api_port} is already in use, API server may already be running")
            
        # Start the API server
        self.start_api_server()
        
        # Start the dashboard server
        if self._check_node_installed():
            if blocking:
                self._start_next_server()
            else:
                self.server_thread = threading.Thread(target=self._start_next_server)
                self.server_thread.daemon = True
                self.server_thread.start()
                
                # Wait a moment for the server to start
                time.sleep(1)
                
                if self.is_port_in_use(self.port):
                    self.logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
                    self.server_running = True
                else:
                    self.logger.error("Failed to start dashboard server")
        else:
            self.logger.warning("Node.js is not installed, dashboard will not be available")
            
    def _check_node_installed(self):
        """Check if Node.js is installed"""
        try:
            subprocess.run(["node", "--version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def _start_next_server(self):
        """Start the Next.js server"""
        os.chdir(self.frontend_dir)
        
        # Check if dependencies are installed
        if not (self.frontend_dir / "node_modules").exists():
            self.logger.info("Installing dependencies...")
            try:
                subprocess.run(["npm", "install"], check=True)
            except subprocess.SubprocessError as e:
                self.logger.error(f"Failed to install dependencies: {e}")
                return
        
        # Start the Next.js server
        try:
            self.logger.info(f"Starting Next.js server at http://{self.host}:{self.port}")
            # Configure Next.js to use our API server
            env = os.environ.copy()
            env["NEXT_PUBLIC_API_URL"] = f"http://{self.host}:{self.api_port}"
            
            # Start the development server
            subprocess.run(["npm", "run", "dev", "--", "-p", str(self.port)], 
                          env=env, check=True)
        except subprocess.SubprocessError as e:
            self.logger.error(f"Failed to start Next.js server: {e}")
            
    def start_api_server(self):
        """Start the API server to provide data to the dashboard"""
        api_handler = self.create_api_handler()
        
        handler = http.server.SimpleHTTPRequestHandler
        handler.protocol_version = "HTTP/1.0"
        
        class CustomHandler(handler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                
            def do_GET(self):
                if self.path.startswith('/api/'):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    # Parse the URL
                    parsed_url = urlparse(self.path)
                    endpoint = parsed_url.path.split('/api/')[1]
                    query_params = parse_qs(parsed_url.query)
                    
                    # Convert query params from lists to single values
                    params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
                    
                    # Call the API handler
                    result = api_handler(endpoint, params)
                    self.wfile.write(json.dumps(result).encode())
                else:
                    self.send_error(404)
        
        socketserver.TCPServer.allow_reuse_address = True
        self.api_server = socketserver.TCPServer(("", self.api_port), CustomHandler)
        
        self.api_thread = threading.Thread(target=self.api_server.serve_forever)
        self.api_thread.daemon = True
        self.api_thread.start()
        
        self.logger.info(f"API server started at http://{self.host}:{self.api_port}")
        
    def create_api_handler(self):
        """Create a handler function for API requests"""
        bot = self.trading_bot
        
        def api_handler(endpoint, params):
            """Handle API requests"""
            environment = params.get('environment', 'virtual')
            
            try:
                if endpoint == 'dashboard':
                    # Return dashboard data
                    return {
                        'overview': self.get_overview_data(environment),
                        'positions': self.get_positions_data(environment),
                        'performance': self.get_performance_data(environment),
                        'strategies': self.get_strategies_data(environment),
                        'alerts': self.get_alerts_data(environment)
                    }
                elif endpoint == 'positions':
                    # Return positions data
                    symbol = params.get('symbol')
                    return self.get_positions_data(environment, symbol)
                elif endpoint == 'candles':
                    # Return candle data
                    symbol = params.get('symbol', 'BTCUSDT')
                    timeframe = params.get('timeframe', '1h')
                    return self.get_candles_data(symbol, timeframe, environment)
                else:
                    return {'error': 'Invalid endpoint'}
            except Exception as e:
                self.logger.error(f"API error: {str(e)}")
                return {'error': str(e)}
                
        return api_handler
    
    def get_overview_data(self, environment):
        """Get overview data for the dashboard"""
        try:
            # In a real implementation, this would fetch data from the bot
            # For now, we'll return mock data similar to the frontend
            return {
                'lastUpdate': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'status': 'running',
                'balance': 10000,
                'dailyPnL': 120,
                'monthlyPnL': 1450,
                'monthlyPnLPercentage': 14.5,
                'activePositions': 3,
                'winRate': 65,
                'drawdown': -8.2,
                'sharpeRatio': 2.3,
                'bestStrategy': 'Breakout',
                'activeSince': '2023-01-15'
            }
        except Exception as e:
            self.logger.error(f"Error getting overview data: {str(e)}")
            return {}
            
    def get_positions_data(self, environment, symbol=None):
        """Get positions data for the dashboard"""
        try:
            # In a real implementation, this would fetch positions from the bot
            # For now, we'll just return an empty list
            positions = []
            if self.trading_bot and hasattr(self.trading_bot, 'exchange'):
                # Get actual positions from the bot if available
                bot_positions = self.trading_bot.exchange.get_positions()
                for pos in bot_positions:
                    positions.append({
                        'id': pos.get('id', ''),
                        'symbol': pos.get('symbol', ''),
                        'direction': pos.get('direction', ''),
                        'entryPrice': pos.get('entry_price', 0),
                        'currentPrice': pos.get('current_price', 0),
                        'pnl': pos.get('unrealized_pnl', 0),
                        'pnlPercentage': pos.get('pnl_percentage', 0),
                        'strategy': pos.get('strategy', ''),
                        'entryTime': pos.get('entry_time', ''),
                        'tp1': pos.get('tp1', 0),
                        'tp2': pos.get('tp2', 0),
                        'tp3': pos.get('tp3', 0),
                        'stopLoss': pos.get('stop_loss', 0),
                        'tp1Hit': pos.get('tp1_hit', False),
                        'tp2Hit': pos.get('tp2_hit', False),
                        'tp3Hit': pos.get('tp3_hit', False)
                    })
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions data: {str(e)}")
            return []
            
    def get_performance_data(self, environment):
        """Get performance data for the dashboard"""
        try:
            # In a real implementation, this would fetch performance data from the bot
            # For now, we'll return mock data similar to the frontend
            return {
                'totalTrades': 142,
                'winRate': 65.3,
                'wins': 92,
                'losses': 50,
                'profitFactor': 1.87,
                'averageTrade': 25.4,
                'averageTradePercentage': 0.76,
                'maxDrawdown': 12.4,
                'monthlyReturn': 14.5,
                'sharpeRatio': 2.3,
                'avgTradeDuration': '3h 42m',
                'dailyPerformance': self.generate_daily_performance()
            }
        except Exception as e:
            self.logger.error(f"Error getting performance data: {str(e)}")
            return {}
            
    def generate_daily_performance(self):
        """Generate daily performance data for the chart"""
        daily_performance = []
        balance = 10000 - 1500  # Starting balance 30 days ago
        
        for i in range(30, -1, -1):
            date = time.time() - (i * 24 * 60 * 60)  # i days ago
            date_str = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(date))
            
            # Generate some random daily changes
            daily_change = (0.5 - (0.5 * (i/30))) * 200  # Trending upward
            balance += daily_change
            
            daily_performance.append({
                'date': date_str,
                'balance': balance,
                'pnl': daily_change
            })
            
        return daily_performance
            
    def get_strategies_data(self, environment):
        """Get strategy performance data for the dashboard"""
        try:
            # In a real implementation, this would fetch strategy data from the bot
            # For now, we'll return mock data similar to the frontend
            return [
                {
                    'name': 'Breakout',
                    'winRate': 72.5,
                    'totalTrades': 40,
                    'profit': 580,
                    'roi': 5.8,
                    'usageCount': 40
                },
                {
                    'name': 'Trend Following',
                    'winRate': 68.2,
                    'totalTrades': 35,
                    'profit': 420,
                    'roi': 4.2,
                    'usageCount': 35
                },
                {
                    'name': 'Mean Reversion',
                    'winRate': 60.0,
                    'totalTrades': 25,
                    'profit': 320,
                    'roi': 3.2,
                    'usageCount': 25
                },
                {
                    'name': 'Range',
                    'winRate': 63.6,
                    'totalTrades': 22,
                    'profit': 280,
                    'roi': 2.8,
                    'usageCount': 22
                },
                {
                    'name': 'Volatility',
                    'winRate': 55.0,
                    'totalTrades': 20,
                    'profit': 150,
                    'roi': 1.5,
                    'usageCount': 20
                }
            ]
        except Exception as e:
            self.logger.error(f"Error getting strategies data: {str(e)}")
            return []
            
    def get_alerts_data(self, environment):
        """Get alerts data for the dashboard"""
        try:
            # In a real implementation, this would fetch alerts from the bot
            # For now, we'll return mock data similar to the frontend
            return [
                {
                    'type': 'success',
                    'title': 'Trade Closed',
                    'message': 'BTCUSDT long position closed with profit of $120.50 (2.3%)',
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 1800)),
                    'actions': ['View Details']
                },
                {
                    'type': 'info',
                    'title': 'New Position Opened',
                    'message': 'ETHUSDT short position opened at $1,845.20 using Trend Following strategy',
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 3600)),
                    'actions': ['View Position']
                },
                {
                    'type': 'warning',
                    'title': 'High Volatility Detected',
                    'message': 'Market volatility for BTC has increased. Position sizing adjusted accordingly.',
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() - 7200))
                }
            ]
        except Exception as e:
            self.logger.error(f"Error getting alerts data: {str(e)}")
            return []
    
    def get_candles_data(self, symbol, timeframe, environment):
        """Get candle data for the specified symbol and timeframe"""
        try:
            # In a real implementation, this would fetch candle data from the bot
            # For now, we'll return mock data
            return self.generate_candle_data(symbol, timeframe, 200)
        except Exception as e:
            self.logger.error(f"Error getting candle data: {str(e)}")
            return []
            
    def generate_candle_data(self, symbol, timeframe, count):
        """Generate realistic candle data with trends, volatility and volume"""
        # Set base price based on symbol
        base_prices = {
            'BTCUSDT': 26500,
            'ETHUSDT': 1850,
            'SOLUSDT': 32,
            'BNBUSDT': 245,
            'ADAUSDT': 0.35
        }
        base_price = base_prices.get(symbol, 100)
        
        # Calculate interval in seconds
        intervals = {
            '5m': 5 * 60,
            '15m': 15 * 60,
            '1h': 60 * 60,
            '4h': 4 * 60 * 60,
            '1d': 24 * 60 * 60
        }
        interval = intervals.get(timeframe, 60 * 60)
        
        # Set volatility based on symbol
        volatilities = {
            'BTCUSDT': 0.015,
            'ETHUSDT': 0.02,
            'SOLUSDT': 0.035,
            'BNBUSDT': 0.025,
            'ADAUSDT': 0.03
        }
        volatility = volatilities.get(symbol, 0.02)
        
        # Generate candles with random walk
        candles = []
        current_price = base_price
        trend = 0
        trend_strength = 0
        trend_duration = 0
        
        # End time is now
        end_time = int(time.time())
        
        for i in range(count):
            # Update trend every ~20 candles
            if i % 20 == 0 or trend_duration <= 0:
                import random
                trend = 1 if random.random() > 0.5 else -1
                trend_strength = random.random() * 0.01
                trend_duration = int(random.random() * 40) + 10
            trend_duration -= 1
            
            # Calculate timestamp (going backward from now)
            timestamp = end_time - ((count - i) * interval)
            
            # Random price change with trend bias
            import random
            price_change = (random.random() * 2 - 1) * volatility * current_price
            trend_change = trend * trend_strength * current_price
            current_price += price_change + trend_change
            
            # Ensure price doesn't go negative
            current_price = max(current_price, base_price * 0.1)
            
            # Generate candle
            open_price = current_price
            close_price = current_price * (1 + (random.random() * 0.01 - 0.005))
            high_price = max(open_price, close_price) * (1 + random.random() * 0.005)
            low_price = min(open_price, close_price) * (1 - random.random() * 0.005)
            
            # Volume increases with volatility
            price_swing = abs(high_price - low_price) / current_price
            base_volume = current_price * 100
            volume = base_volume * (1 + price_swing * 10)
            
            candles.append({
                'timestamp': timestamp * 1000,  # Convert to milliseconds
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            # Update current price to close price for next candle
            current_price = close_price
            
        return candles
    
    def stop(self):
        """Stop the dashboard server"""
        if self.api_server:
            self.api_server.shutdown()
            self.api_server.server_close()
            self.logger.info("API server stopped")
            
        self.server_running = False
        self.logger.info("Dashboard server stopped")
