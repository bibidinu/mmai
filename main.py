#!/usr/bin/env python3
"""
Adaptive Multi-Strategy Crypto Trading Bot - Main Entry Point
"""
import os
import json
import yaml
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

# Import ConfigManager
from config.config_manager import ConfigManager

# Core system imports
from exchange.exchange_factory import ExchangeFactory  # We'll use the factory pattern
from exchange.bybit import BybitExchange
from strategies.strategy_selector import StrategySelector
from take_profit.dynamic_tp import DynamicTakeProfitManager
from risk_management.position_sizing import RiskManager
from market_analysis.technical import TechnicalAnalyzer
from market_analysis.sentiment import SentimentAnalyzer
from market_analysis.volume import VolumeAnalyzer
from market_analysis.correlation import CorrelationMatrix
from market_analysis.volatility import VolatilityAnalyzer
from learning.self_optimization import LearningManager
from notification.telegram import TelegramNotifier
from notification.discord import DiscordNotifier
from utils.database import DatabaseManager
from utils.logger import setup_logger

class CryptoTradingBot:
    def __init__(self, config_path, mode="virtual", exchange_name="bybit"):
        """
        Initialize the trading bot with configuration
        
        Args:
            config_path (str): Path to configuration directory
            mode (str): Trading mode - 'virtual' or 'mainnet'
            exchange_name (str): Name of exchange to use
        """
        self.mode = mode
        self.exchange_name = exchange_name
        
        # Initialize logging first
        log_dir = os.path.join(os.path.dirname(config_path), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = setup_logger(f"bot_{mode}", log_dir=log_dir)
        self.logger.info(f"Initializing trading bot in {mode} mode using {exchange_name} exchange")
        
        try:
            # Initialize ConfigManager
            self.config_manager = ConfigManager(config_path, env=mode)
            
            # Load configuration using ConfigManager
            self.config = self.config_manager.get_config('env')
            self.credentials = self.config_manager.get_config('credentials')
            
            if not self.config:
                self.logger.error("Failed to load configuration. Check your config files.")
                raise ValueError("Configuration loading failed")
            
            # Create data directory if it doesn't exist
            data_path = self.config.get("database", {}).get("path", "./data/trading.db")
            data_dir = os.path.dirname(data_path)
            if data_dir and not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                self.logger.info(f"Created data directory: {data_dir}")
            
            # Initialize components with proper error handling
            self._initialize_components()
            
            self.logger.info("Trading bot initialization complete")
            
        except Exception as e:
            self.logger.error(f"Critical error during initialization: {str(e)}", exc_info=True)
            print(f"Critical error during initialization: {str(e)}")
            sys.exit(1)
    
    def _initialize_components(self):
        """Initialize all bot components with proper error handling"""
        try:
            # Initialize database
            self.db = self._initialize_database()
            
            # Initialize exchange
            self.exchange = self._initialize_exchange()
            
            # Initialize market analyzers
            self.technical = self._initialize_technical_analyzer()
            self.sentiment = self._initialize_sentiment_analyzer()
            self.volume = self._initialize_volume_analyzer()
            self.correlation = self._initialize_correlation_analyzer() 
            self.volatility = self._initialize_volatility_analyzer()
            
            # Initialize strategy components
            self.strategy_selector = self._initialize_strategy_selector()
            
            # Initialize trade management components
            self.risk_manager = self._initialize_risk_manager()
            self.tp_manager = self._initialize_tp_manager()
            
            # Initialize learning system
            self.learning_manager = self._initialize_learning_manager()
            
            # Initialize notification systems
            self.telegram = self._initialize_telegram()
            self.discord = self._initialize_discord()
            
            # Connect components that need references to each other
            if self.telegram:
                self.telegram.set_exchange(self.exchange)
            
        except Exception as e:
            self.logger.error(f"Error during component initialization: {str(e)}", exc_info=True)
            raise
    
    def _initialize_database(self):
        """Initialize database with error handling"""
        try:
            db_config = self.config.get("database", {})
            db = DatabaseManager(db_config)
            self.logger.info(f"Database initialized: {db_config.get('type', 'sqlite')}")
            return db
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
            self.logger.warning("Using in-memory database as fallback")
            # Return a simple in-memory database as fallback
            return DatabaseManager({"type": "sqlite", "path": ":memory:"})
    
    def _initialize_exchange(self):
        """Initialize exchange with error handling"""
        try:
            # For now we're directly initializing ByBit exchange
            # In the future we can implement an ExchangeFactory
            exchange_config = self.config.get("exchange", {})
            exchange_credentials = self.credentials.get(self.exchange_name, {})
            
            # Add simulation_mode option for testing without valid credentials
            exchange_config["simulation_mode"] = self.mode == "virtual"
            
            exchange = BybitExchange(
                exchange_credentials, 
                exchange_config,
                testnet=(self.mode != "mainnet")
            )
            
            self.logger.info(f"Exchange initialized: {self.exchange_name} (testnet: {self.mode != 'mainnet'})")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {str(e)}", exc_info=True)
            
            # Create a minimal exchange config for fallback
            minimal_config = {
                "simulation_mode": True,
                "symbols_to_watch": self.config.get("trading", {}).get("symbols", ["BTCUSDT"]),
                "quote_currency": "USDT"
            }
            
            # Initialize with minimal credentials and simulation mode
            self.logger.warning("Using simulation-only exchange as fallback")
            return BybitExchange(
                {"api_key": "", "api_secret": ""}, 
                minimal_config,
                testnet=True
            )
    
    def _initialize_technical_analyzer(self):
        """Initialize technical analyzer with error handling"""
        try:
            technical_config = self.config.get("technical_analysis", {})
            analyzer = TechnicalAnalyzer(technical_config)
            self.logger.info("Technical analyzer initialized")
            return analyzer
        except Exception as e:
            self.logger.error(f"Technical analyzer initialization failed: {str(e)}", exc_info=True)
            # Return a minimal technical analyzer as fallback
            self.logger.warning("Using minimal technical analyzer as fallback")
            return TechnicalAnalyzer({})
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analyzer with error handling"""
        try:
            sentiment_config = self.config.get("sentiment_analysis", {})
            analyzer = SentimentAnalyzer(sentiment_config)
            self.logger.info("Sentiment analyzer initialized")
            return analyzer
        except Exception as e:
            self.logger.error(f"Sentiment analyzer initialization failed: {str(e)}", exc_info=True)
            # Return a minimal sentiment analyzer as fallback
            self.logger.warning("Using minimal sentiment analyzer as fallback")
            return SentimentAnalyzer({})
    
    def _initialize_volume_analyzer(self):
        """Initialize volume analyzer with error handling"""
        try:
            volume_config = self.config.get("volume_analysis", {})
            analyzer = VolumeAnalyzer(volume_config)
            self.logger.info("Volume analyzer initialized")
            return analyzer
        except Exception as e:
            self.logger.error(f"Volume analyzer initialization failed: {str(e)}", exc_info=True)
            # Return a minimal volume analyzer as fallback
            self.logger.warning("Using minimal volume analyzer as fallback")
            return VolumeAnalyzer({})
    
    def _initialize_correlation_analyzer(self):
        """Initialize correlation analyzer with error handling"""
        try:
            correlation_config = self.config.get("correlation_analysis", {})
            analyzer = CorrelationMatrix(correlation_config)
            self.logger.info("Correlation analyzer initialized")
            return analyzer
        except Exception as e:
            self.logger.error(f"Correlation analyzer initialization failed: {str(e)}", exc_info=True)
            # Return a minimal correlation analyzer as fallback
            self.logger.warning("Using minimal correlation analyzer as fallback")
            return CorrelationMatrix({})
    
    def _initialize_volatility_analyzer(self):
        """Initialize volatility analyzer with error handling"""
        try:
            volatility_config = self.config.get("volatility_analysis", {})
            analyzer = VolatilityAnalyzer(volatility_config)
            self.logger.info("Volatility analyzer initialized")
            return analyzer
        except Exception as e:
            self.logger.error(f"Volatility analyzer initialization failed: {str(e)}", exc_info=True)
            # Return a minimal volatility analyzer as fallback
            self.logger.warning("Using minimal volatility analyzer as fallback")
            return VolatilityAnalyzer({})
    
    def _initialize_strategy_selector(self):
        """Initialize strategy selector with error handling"""
        try:
            strategy_config = self.config.get("strategies", {})
            selector = StrategySelector(
                strategy_config,
                self.technical,
                self.sentiment,
                self.volume,
                self.correlation,
                self.volatility
            )
            self.logger.info("Strategy selector initialized")
            return selector
        except Exception as e:
            self.logger.error(f"Strategy selector initialization failed: {str(e)}", exc_info=True)
            # Return a minimal strategy selector as fallback
            self.logger.warning("Using minimal strategy selector as fallback")
            return StrategySelector(
                {},
                self.technical,
                self.sentiment,
                self.volume,
                self.correlation,
                self.volatility
            )
    
    def _initialize_risk_manager(self):
        """Initialize risk manager with error handling"""
        try:
            risk_config = self.config.get("risk_management", {})
            manager = RiskManager(risk_config)
            self.logger.info("Risk manager initialized")
            return manager
        except Exception as e:
            self.logger.error(f"Risk manager initialization failed: {str(e)}", exc_info=True)
            # Return a minimal risk manager as fallback
            self.logger.warning("Using minimal risk manager as fallback")
            return RiskManager({})
    
    def _initialize_tp_manager(self):
        """Initialize take profit manager with error handling"""
        try:
            tp_config = self.config.get("take_profit", {})
            manager = DynamicTakeProfitManager(
                tp_config,
                self.technical,
                self.volatility
            )
            self.logger.info("Take profit manager initialized")
            return manager
        except Exception as e:
            self.logger.error(f"Take profit manager initialization failed: {str(e)}", exc_info=True)
            # Return a minimal TP manager as fallback
            self.logger.warning("Using minimal take profit manager as fallback")
            return DynamicTakeProfitManager(
                {},
                self.technical,
                self.volatility
            )
    
    def _initialize_learning_manager(self):
        """Initialize learning manager with error handling"""
        try:
            learning_config = self.config.get("learning", {})
            manager = LearningManager(
                learning_config,
                self.db
            )
            self.logger.info("Learning manager initialized")
            return manager
        except Exception as e:
            self.logger.error(f"Learning manager initialization failed: {str(e)}", exc_info=True)
            # Return a minimal learning manager as fallback
            self.logger.warning("Using minimal learning manager as fallback")
            return LearningManager({}, self.db)
    
    def _initialize_telegram(self):
        """Initialize Telegram notifier with error handling"""
        try:
            telegram_config = self.config.get("notifications", {}).get("telegram", {})
            telegram_credentials = self.credentials.get("telegram", {})
            
            if not telegram_config.get("enabled", False):
                self.logger.info("Telegram notifications disabled in config")
                return None
                
            notifier = TelegramNotifier(
                telegram_credentials,
                telegram_config
            )
            
            if notifier.enabled:
                self.logger.info("Telegram notifier initialized")
            else:
                self.logger.info("Telegram notifier initialized in logging-only mode")
                
            return notifier
            
        except Exception as e:
            self.logger.error(f"Telegram notifier initialization failed: {str(e)}", exc_info=True)
            self.logger.warning("Telegram notifications will not be available")
            return None
    
    def _initialize_discord(self):
        """Initialize Discord notifier with error handling"""
        try:
            discord_config = self.config.get("notifications", {}).get("discord", {})
            discord_credentials = self.credentials.get("discord", {})
            
            if not discord_config.get("enabled", False):
                self.logger.info("Discord notifications disabled in config")
                return None
                
            notifier = DiscordNotifier(
                discord_credentials,
                discord_config
            )
            
            self.logger.info("Discord notifier initialized")
            return notifier
            
        except Exception as e:
            self.logger.error(f"Discord notifier initialization failed: {str(e)}", exc_info=True)
            self.logger.warning("Discord notifications will not be available")
            return None
    
    def start(self):
        """Start the trading bot with improved error handling"""
        self.logger.info("Starting trading bot")
        self.running = True
        
        # Send startup notifications with error handling
        self._send_startup_notification()
        
        try:
            self._run_trading_loop()
        except KeyboardInterrupt:
            self.logger.info("Bot stopped by user")
        except Exception as e:
            self.logger.error(f"Critical error in trading loop: {str(e)}", exc_info=True)
            if self.telegram:
                self.telegram.send_error(e, "Trading loop")
            if self.discord:
                self.discord.send_message(f"⚠️ Bot critical error: {str(e)}")
        finally:
            self.stop()
    
    def _send_startup_notification(self):
        """Send startup notifications with error handling"""
        startup_message = f"🚀 Trading bot started in {self.mode} mode"
        
        try:
            if self.telegram:
                self.telegram.send_message(startup_message)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram startup notification: {str(e)}")
        
        try:
            if self.discord:
                self.discord.send_message(startup_message)
        except Exception as e:
            self.logger.error(f"Failed to send Discord startup notification: {str(e)}")
    
    def _run_trading_loop(self):
        """Main trading loop with improved error handling and recovery"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                # Get market data with error handling
                market_data = self._get_market_data()
                
                # Update market analyzers with error handling
                self._update_analyzers(market_data)
                
                # Get current positions with error handling
                positions = self._get_positions()
                
                # Select best strategy based on current market conditions
                selected_strategy, action, params = self._select_strategy(market_data, positions)
                
                # Execute trade if action is suggested
                if action:
                    self._execute_trade(selected_strategy, action, params)
                
                # Manage existing positions (move to breakeven, etc.)
                self._manage_positions(positions)
                
                # Run self-optimization if scheduled
                self._run_optimization_if_needed()
                
                # Reset error counter on successful cycle
                consecutive_errors = 0
                
                # Wait for next cycle
                time.sleep(self.config.get("trading", {}).get("update_interval", 10))
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Error in trading cycle: {str(e)}", exc_info=True)
                
                # Send error notification (limit frequency to avoid spam)
                if consecutive_errors <= 3:
                    if self.telegram:
                        self.telegram.send_error(e, "Trading cycle")
                
                # Increase wait time if we're having consecutive errors
                if consecutive_errors > max_consecutive_errors:
                    error_wait_time = min(60, 5 * consecutive_errors)  # Cap at 60 seconds
                    self.logger.warning(f"Too many consecutive errors, waiting {error_wait_time}s before retrying")
                    time.sleep(error_wait_time)
                else:
                    time.sleep(5)  # Short wait before retrying
    
    def _get_market_data(self):
        """Get market data with error handling"""
        try:
            return self.exchange.get_market_data(
                self.config.get("trading", {}).get("symbols", ["BTCUSDT"]),
                self.config.get("trading", {}).get("timeframes", ["1h"])
            )
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            # Return empty data structure as fallback
            return {}
    
    def _update_analyzers(self, market_data):
        """Update market analyzers with error handling"""
        components = [
            ("Technical Analyzer", lambda: self.technical.update(market_data)),
            ("Sentiment Analyzer", lambda: self.sentiment.update()),
            ("Volume Analyzer", lambda: self.volume.update(market_data)),
            ("Correlation Matrix", lambda: self.correlation.update(market_data)),
            ("Volatility Analyzer", lambda: self.volatility.update(market_data))
        ]
        
        for name, update_fn in components:
            try:
                update_fn()
            except Exception as e:
                self.logger.error(f"Error updating {name}: {str(e)}")
    
    def _get_positions(self):
        """Get current positions with error handling"""
        try:
            return self.exchange.get_positions()
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def _select_strategy(self, market_data, positions):
        """Select strategy with error handling"""
        try:
            return self.strategy_selector.select_strategy(market_data, positions)
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {str(e)}")
            # Return no action as fallback
            return None, None, {}
    
    def _execute_trade(self, selected_strategy, action, params):
        """Execute trade with error handling"""
        try:
            # Check if we have necessary parameters
            symbol = params.get("symbol")
            direction = params.get("direction")
            
            if not symbol or not direction:
                self.logger.warning(f"Missing parameters for trade execution: {params}")
                return
                
            # Determine position size
            position_size = self._calculate_position_size(symbol, direction)
            
            # Calculate take profit levels
            tp_levels = self._calculate_tp_levels(symbol, direction, params)
            
            # Calculate stop loss
            stop_loss = self._calculate_stop_loss(symbol, direction, params)
            
            # Execute trade based on action
            if action == "enter":
                self._execute_entry(selected_strategy, symbol, direction, position_size, tp_levels, stop_loss, params)
            elif action == "exit":
                self._execute_exit(selected_strategy, symbol, direction, params)
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
    
    def _calculate_position_size(self, symbol, direction):
        """Calculate position size with error handling"""
        try:
            return self.risk_manager.calculate_position_size(
                symbol, 
                direction,
                self.exchange.get_balance(),
                self.volatility.get_volatility(symbol)
            )
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            # Return minimal position size as fallback
            return self.config.get("risk_management", {}).get("min_position_size", 0.001)
    
    def _calculate_tp_levels(self, symbol, direction, params):
        """Calculate take profit levels with error handling"""
        try:
            entry_price = params.get("entry_price", self.exchange.get_market_price(symbol))
            return self.tp_manager.calculate_tp_levels(symbol, direction, entry_price)
        except Exception as e:
            self.logger.error(f"Error calculating TP levels: {str(e)}")
            # Calculate simple percentage-based TPs as fallback
            current_price = self.exchange.get_market_price(symbol)
            tp_percentages = self.config.get("take_profit", {}).get("fixed_tp_percentages", [0.01, 0.02, 0.035])
            
            if direction == "long":
                return [current_price * (1 + pct) for pct in tp_percentages]
            else:
                return [current_price * (1 - pct) for pct in tp_percentages]
    
    def _calculate_stop_loss(self, symbol, direction, params):
        """Calculate stop loss with error handling"""
        try:
            entry_price = params.get("entry_price", self.exchange.get_market_price(symbol))
            return self.risk_manager.calculate_stop_loss(
                symbol,
                direction,
                entry_price,
                self.volatility.get_volatility(symbol)
            )
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            # Calculate simple percentage-based SL as fallback
            current_price = self.exchange.get_market_price(symbol)
            sl_percentage = self.config.get("risk_management", {}).get("default_sl_atr_multiple", 1.5) * 0.01
            
            if direction == "long":
                return current_price * (1 - sl_percentage)
            else:
                return current_price * (1 + sl_percentage)
    
    def _execute_entry(self, selected_strategy, symbol, direction, position_size, tp_levels, stop_loss, params):
        """Execute entry trade with error handling"""
        try:
            # Execute trade
            trade_result = self.exchange.open_position(
                symbol,
                direction,
                position_size,
                tp_levels,
                stop_loss
            )
            
            # Send notifications
            if trade_result and trade_result.get("success"):
                self._send_entry_notification(selected_strategy, symbol, direction, trade_result, tp_levels, stop_loss, params)
                
                # Store trade data for learning
                self._store_trade_entry(trade_result.get("trade_id"), symbol, direction, selected_strategy, 
                                       params.get('entry_price', trade_result.get("entry_price")),
                                       position_size, tp_levels, stop_loss)
                                       
        except Exception as e:
            self.logger.error(f"Error executing entry: {str(e)}")
    
    def _send_entry_notification(self, strategy, symbol, direction, trade_result, tp_levels, stop_loss, params):
        """Send entry notification with error handling"""
        try:
            trade_id = trade_result.get("trade_id")
            message = (
                f"🔵 New trade opened:\n"
                f"Symbol: {symbol}\n"
                f"Direction: {direction}\n"
                f"Strategy: {strategy}\n"
                f"Entry: {params.get('entry_price', trade_result.get('entry_price'))}\n"
                f"TP1: {tp_levels[0] if tp_levels and len(tp_levels) > 0 else 'N/A'}\n"
                f"TP2: {tp_levels[1] if tp_levels and len(tp_levels) > 1 else 'N/A'}\n"
                f"TP3: {tp_levels[2] if tp_levels and len(tp_levels) > 2 else 'N/A'}\n"
                f"SL: {stop_loss}"
            )
            
            if self.telegram:
                self.telegram.send_message(message, level="trade_entry")
            if self.discord:
                self.discord.send_message(message)
                
        except Exception as e:
            self.logger.error(f"Error sending entry notification: {str(e)}")
    
    def _store_trade_entry(self, trade_id, symbol, direction, strategy, entry_price, position_size, tp_levels, stop_loss):
        """Store trade entry data for learning with error handling"""
        try:
            self.learning_manager.store_trade_entry(
                trade_id,
                symbol,
                direction,
                strategy,
                entry_price,
                position_size,
                tp_levels,
                stop_loss,
                self.technical.get_market_context(symbol),
                self.sentiment.get_sentiment(symbol),
                self.volume.get_volume_profile(symbol),
                self.correlation.get_correlations(symbol),
                self.volatility.get_volatility_regime(symbol)
            )
        except Exception as e:
            self.logger.error(f"Error storing trade entry data: {str(e)}")
    
    def _execute_exit(self, selected_strategy, symbol, direction, params):
        """Execute exit trade with error handling"""
        try:
            # Execute trade
            trade_result = self.exchange.close_position(
                symbol,
                params.get("position_id")
            )
            
            # Send notifications
            if trade_result and trade_result.get("success"):
                self._send_exit_notification(selected_strategy, symbol, direction, trade_result, params)
                
                # Store trade exit data for learning
                self._store_trade_exit(params.get("position_id"), trade_result)
                
        except Exception as e:
            self.logger.error(f"Error executing exit: {str(e)}")
    
    def _send_exit_notification(self, strategy, symbol, direction, trade_result, params):
        """Send exit notification with error handling"""
        try:
            message = (
                f"🟢 Position closed:\n"
                f"Symbol: {symbol}\n"
                f"Direction: {direction}\n"
                f"Strategy: {strategy}\n"
                f"Exit price: {trade_result.get('exit_price')}\n"
                f"Profit/Loss: {trade_result.get('pnl')}"
            )
            
            if self.telegram:
                self.telegram.send_message(message, level="trade_exit")
            if self.discord:
                self.discord.send_message(message)
                
        except Exception as e:
            self.logger.error(f"Error sending exit notification: {str(e)}")
    
    def _store_trade_exit(self, position_id, trade_result):
        """Store trade exit data for learning with error handling"""
        try:
            self.learning_manager.store_trade_exit(
                position_id,
                trade_result.get('exit_price'),
                trade_result.get('pnl'),
                trade_result.get('exit_reason')
            )
        except Exception as e:
            self.logger.error(f"Error storing trade exit data: {str(e)}")
    
    def _manage_positions(self, positions):
        """Manage existing positions with error handling"""
        for position in positions:
            try:
                self.tp_manager.manage_position(position, self.exchange)
            except Exception as e:
                position_id = position.get("id", "unknown")
                self.logger.error(f"Error managing position {position_id}: {str(e)}")
    
    def _run_optimization_if_needed(self):
        """Run self-optimization if scheduled with error handling"""
        try:
            if self.learning_manager.should_optimize():
                self.logger.info("Running scheduled optimization")
                
                # Send notification
                if self.telegram:
                    self.telegram.send_message("🧠 Running model optimization...", level="info")
                
                # Run optimization
                self.learning_manager.optimize()
                
                # Update strategy selector model
                self.strategy_selector.update_model(self.learning_manager.get_model())
                
                # Send notification
                if self.telegram:
                    self.telegram.send_message("✅ Model optimization complete", level="info")
                    
        except Exception as e:
            self.logger.error(f"Error during model optimization: {str(e)}")
    
    def stop(self):
        """Stop the trading bot with improved error handling"""
        self.logger.info("Stopping trading bot")
        self.running = False
        
        # Send shutdown notifications
        self._send_shutdown_notification()
        
        # Close connections safely
        self._safe_close()
    
    def _send_shutdown_notification(self):
        """Send shutdown notification with error handling"""
        shutdown_message = f"🛑 Trading bot stopped in {self.mode} mode"
        
        try:
            if self.telegram:
                self.telegram.send_message(shutdown_message)
        except Exception as e:
            self.logger.error(f"Failed to send Telegram shutdown notification: {str(e)}")
        
        try:
            if self.discord:
                self.discord.send_message(shutdown_message)
        except Exception as e:
            self.logger.error(f"Failed to send Discord shutdown notification: {str(e)}")
    
    def _safe_close(self):
        """Safely close all connections with error handling"""
        components = [
            ("Exchange", lambda: self.exchange.close() if hasattr(self.exchange, 'close') else None),
            ("Database", lambda: self.db.close() if hasattr(self.db, 'close') else None),
            ("Telegram", lambda: self.telegram.close() if self.telegram and hasattr(self.telegram, 'close') else None),
            ("Discord", lambda: self.discord.close() if self.discord and hasattr(self.discord, 'close') else None)
        ]
        
        for name, close_fn in components:
            try:
                close_fn()
                self.logger.info(f"Closed {name} connection")
            except Exception as e:
                self.logger.error(f"Error closing {name} connection: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Multi-Strategy Crypto Trading Bot")
    parser.add_argument("--config", type=str, default="./config", help="Path to config directory")
    parser.add_argument("--mode", type=str, default="virtual", choices=["virtual", "mainnet"], 
                        help="Trading mode: virtual (paper trading) or mainnet (real trading)")
    parser.add_argument("--exchange", type=str, default="bybit", choices=["bybit", "binance"],
                        help="Exchange to use for trading")
    
    args = parser.parse_args()
    
    # Configure logging to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)
    
    try:
        bot = CryptoTradingBot(args.config, args.mode, args.exchange)
        bot.start()
    except Exception as e:
        logging.critical(f"Bot startup failed: {str(e)}", exc_info=True)
        print(f"ERROR: Bot startup failed: {str(e)}")
        sys.exit(1)
