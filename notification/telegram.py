"""
Telegram Notification Module with improved error handling and fallback mechanisms
"""
import logging
import time
import json
import threading
import requests
from datetime import datetime
import traceback

class TelegramNotifier:
    """
    Handles notifications through Telegram with improved error handling
    """
    
    def __init__(self, credentials, config):
        """
        Initialize the Telegram notifier
        
        Args:
            credentials (dict): API credentials
            config (dict): Notification configuration
        """
        self.logger = logging.getLogger("telegram_notifier")
        self.config = config
        
        # Telegram Bot API
        self.token = credentials.get("bot_token", "")
        self.chat_ids = config.get("chat_ids", [])
        self.admin_chat_id = config.get("admin_chat_id", None)
        
        # Check if token is valid
        self.enabled = True
        if not self.token or self.token == "YOUR_TELEGRAM_BOT_TOKEN":
            self.logger.warning("Invalid or missing Telegram bot token - notifications will be logged only")
            self.enabled = False
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 20)  # Max messages per minute
        self.last_messages = []
        self.message_lock = threading.Lock()
        
        # Message settings
        self.notification_levels = config.get("notification_levels", {
            "trade_entry": True,
            "trade_exit": True,
            "take_profit": True,
            "stop_loss": True,
            "error": True,
            "warning": True,
            "info": False,
            "debug": False
        })
        
        # Command handling
        self.commands = {
            "/status": self._handle_status_command,
            "/balance": self._handle_balance_command,
            "/positions": self._handle_positions_command,
            "/performance": self._handle_performance_command,
            "/help": self._handle_help_command
        }
        
        # Start command polling thread if enabled
        self.polling_enabled = config.get("enable_commands", True) and self.enabled
        self.polling_interval = config.get("polling_interval", 10)  # seconds
        self.polling_thread = None
        self.running = False
        
        # Exchange API reference (to be set later)
        self.exchange = None
        
        # Session for API requests
        self.session = requests.Session()
        
        # Set up retry mechanism
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=5,
            pool_maxsize=5
        )
        self.session.mount('https://', retry_adapter)
        
        # Start command polling if enabled
        if self.polling_enabled:
            self._start_command_polling()
        
        if self.enabled:
            self.logger.info("Telegram notifier initialized and enabled")
        else:
            self.logger.info("Telegram notifier initialized in logging-only mode")
    
    def set_exchange(self, exchange):
        """
        Set exchange API reference for command handling
        
        Args:
            exchange: Exchange API object
        """
        self.exchange = exchange
    
    def _start_command_polling(self):
        """Start polling for commands"""
        self.running = True
        self.polling_thread = threading.Thread(target=self._command_polling_loop, daemon=True)
        self.polling_thread.start()
        self.logger.info("Telegram command polling started")
    
    def _stop_command_polling(self):
        """Stop polling for commands"""
        self.running = False
        if self.polling_thread:
            self.polling_thread.join(timeout=2)
            self.polling_thread = None
        self.logger.info("Telegram command polling stopped")
    
    def _command_polling_loop(self):
        """Main command polling loop with improved error handling"""
        last_update_id = 0
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                # Get updates from Telegram
                updates = self._get_updates(last_update_id)
                
                if updates and "result" in updates and updates["result"]:
                    consecutive_errors = 0  # Reset error counter on success
                    
                    for update in updates["result"]:
                        # Process update
                        last_update_id = max(last_update_id, update["update_id"] + 1)
                        
                        # Handle message if present
                        if "message" in update and "text" in update["message"]:
                            self._process_command(update["message"])
                            
            except Exception as e:
                self.logger.error(f"Error in command polling: {str(e)}")
                consecutive_errors += 1
                
                # If we have too many consecutive errors, increase polling interval
                if consecutive_errors > max_consecutive_errors:
                    self.logger.warning(f"Too many consecutive errors, increasing polling interval")
                    time.sleep(60)  # Sleep for a minute before retrying
                    consecutive_errors = 0
                
            # Sleep before next poll
            time.sleep(self.polling_interval)
    
    def _get_updates(self, offset=0):
        """
        Get updates from Telegram API with improved error handling
        
        Args:
            offset (int): Offset for updates
            
        Returns:
            dict: API response or None on error
        """
        if not self.enabled:
            return None
            
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {
                "offset": offset,
                "timeout": 30
            }
            
            response = self.session.get(url, params=params, timeout=31)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting Telegram updates: {str(e)}")
            return None
        except (ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"Error parsing Telegram response: {str(e)}")
            return None
    
    def _process_command(self, message):
        """
        Process a command message with improved error handling
        
        Args:
            message (dict): Message data
        """
        try:
            text = message.get("text", "").strip()
            chat_id = message.get("chat", {}).get("id")
            user_id = message.get("from", {}).get("id")
            username = message.get("from", {}).get("username", "Unknown")
            
            # Validate user access
            if not self._validate_user_access(chat_id, user_id):
                self.logger.warning(f"Unauthorized command attempt from {username} (ID: {user_id})")
                return
                
            # Check if message is a command
            if text.startswith("/"):
                command = text.split()[0].lower()
                
                # Handle command
                if command in self.commands:
                    self.logger.info(f"Processing command {command} from {username}")
                    
                    # Get args (everything after the command)
                    args = text[len(command):].strip()
                    
                    # Call command handler
                    handler = self.commands[command]
                    response = handler(chat_id, args)
                    
                    # Send response
                    if response:
                        self.send_message(response, chat_id=chat_id)
                else:
                    self.send_message(f"Unknown command: {command}\nType /help for available commands.", chat_id=chat_id)
                    
        except Exception as e:
            self.logger.error(f"Error processing command: {str(e)}")
    
    def _validate_user_access(self, chat_id, user_id):
        """
        Validate if a user has access to commands
        
        Args:
            chat_id (int): Chat ID
            user_id (int): User ID
            
        Returns:
            bool: True if user has access
        """
        # Check if chat_id is in allowed list
        return chat_id in self.chat_ids or chat_id == self.admin_chat_id
    
    def _handle_status_command(self, chat_id, args):
        """
        Handle status command
        
        Args:
            chat_id (int): Chat ID
            args (str): Command arguments
            
        Returns:
            str: Command response
        """
        if not self.exchange:
            return "Exchange API not available"
            
        status_message = "🤖 *Bot Status*\n\n"
        
        try:
            # Get current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_message += f"📅 *Time*: {current_time}\n"
            
            # Get account balance
            balance = self.exchange.get_balance()
            status_message += f"💰 *Balance*: {balance:.2f} USDT\n\n"
            
            # Get open positions
            positions = self.exchange.get_positions()
            
            if positions:
                status_message += f"🔴 *Open Positions*: {len(positions)}\n"
                
                for pos in positions[:5]:  # Show only up to 5 positions
                    symbol = pos.get("symbol")
                    direction = pos.get("direction")
                    entry_price = pos.get("entry_price", 0)
                    current_price = self.exchange.get_market_price(symbol)
                    
                    pnl_pct = ((current_price / entry_price) - 1) * 100
                    if direction == "short":
                        pnl_pct = -pnl_pct
                        
                    status_message += f"• {symbol} {direction.upper()}: {pnl_pct:.2f}%\n"
                    
                if len(positions) > 5:
                    status_message += f"...and {len(positions) - 5} more positions\n"
            else:
                status_message += "🔴 *Open Positions*: None\n"
                
            return status_message
            
        except Exception as e:
            self.logger.error(f"Error handling status command: {str(e)}")
            return f"Error getting status: {str(e)}"
    
    def _handle_balance_command(self, chat_id, args):
        """
        Handle balance command
        
        Args:
            chat_id (int): Chat ID
            args (str): Command arguments
            
        Returns:
            str: Command response
        """
        if not self.exchange:
            return "Exchange API not available"
            
        try:
            # Get account balance
            balance = self.exchange.get_balance()
            
            # Format response
            response = "💰 *Account Balance*\n\n"
            response += f"Total: {balance:.2f} USDT"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling balance command: {str(e)}")
            return f"Error getting balance: {str(e)}"
    
    def _handle_positions_command(self, chat_id, args):
        """
        Handle positions command
        
        Args:
            chat_id (int): Chat ID
            args (str): Command arguments
            
        Returns:
            str: Command response
        """
        if not self.exchange:
            return "Exchange API not available"
            
        try:
            # Get open positions
            positions = self.exchange.get_positions()
            
            if not positions:
                return "No open positions"
                
            # Format response
            response = "🔴 *Open Positions*\n\n"
            
            for pos in positions:
                symbol = pos.get("symbol")
                direction = pos.get("direction")
                entry_price = pos.get("entry_price", 0)
                quantity = pos.get("quantity", 0)
                current_price = self.exchange.get_market_price(symbol)
                
                pnl_pct = ((current_price / entry_price) - 1) * 100
                if direction == "short":
                    pnl_pct = -pnl_pct
                    
                pnl = (current_price - entry_price) * quantity
                if direction == "short":
                    pnl = -pnl
                
                response += f"*{symbol}* ({direction.upper()})\n"
                response += f"Entry: {entry_price:.5f}\n"
                response += f"Current: {current_price:.5f}\n"
                response += f"PnL: {pnl:.2f} USDT ({pnl_pct:.2f}%)\n"
                response += f"Size: {quantity:.5f}\n\n"
                
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling positions command: {str(e)}")
            return f"Error getting positions: {str(e)}"
    
    def _handle_performance_command(self, chat_id, args):
        """
        Handle performance command
        
        Args:
            chat_id (int): Chat ID
            args (str): Command arguments
            
        Returns:
            str: Command response
        """
        # This would require access to the database
        return "Performance statistics not available"
    
    def _handle_help_command(self, chat_id, args):
        """
        Handle help command
        
        Args:
            chat_id (int): Chat ID
            args (str): Command arguments
            
        Returns:
            str: Command response
        """
        response = "🤖 *Available Commands*\n\n"
        response += "/status - Get bot status and overview\n"
        response += "/balance - Get account balance\n"
        response += "/positions - List open positions\n"
        response += "/performance - Get performance statistics\n"
        response += "/help - Show this help message\n"
        
        return response
    
    def send_message(self, message, level="info", chat_id=None):
        """
        Send a message to Telegram with improved error handling
        
        Args:
            message (str): Message text
            level (str): Notification level
            chat_id (int, optional): Specific chat ID to send to
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Always log the message
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"TELEGRAM: {message}")
        
        # If Telegram is disabled, just log and return
        if not self.enabled:
            return False
            
        # Check notification level
        if level not in self.notification_levels or not self.notification_levels[level]:
            return False
            
        # Check rate limiting
        if not self._check_rate_limit():
            self.logger.warning("Rate limit exceeded, message not sent")
            return False
            
        # Add level emoji
        level_emoji = {
            "trade_entry": "🔵",
            "trade_exit": "🟢",
            "take_profit": "✅",
            "stop_loss": "🛑",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "debug": "🔍"
        }
        
        if level in level_emoji:
            message = f"{level_emoji[level]} {message}"
            
        # Determine chat IDs to send to
        targets = []
        
        if chat_id:
            # Send only to specific chat
            targets = [chat_id]
        else:
            # Send to all configured chats
            targets = self.chat_ids
            
            # Also send to admin for error/warning
            if level in ["error", "warning"] and self.admin_chat_id and self.admin_chat_id not in targets:
                targets.append(self.admin_chat_id)
        
        # Send to all targets
        success = True
        
        for target in targets:
            if not self._send_telegram_message(target, message):
                success = False
                
        return success
    
    def _send_telegram_message(self, chat_id, message):
        """
        Send a message to Telegram API with improved error handling
        
        Args:
            chat_id (int): Chat ID
            message (str): Message text
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown"
            }
            
            # Handle long messages - Telegram has a 4096 char limit
            if len(message) > 4000:
                message = message[:3997] + "..."
                data["text"] = message
            
            response = self.session.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            # Add to rate limit tracking
            with self.message_lock:
                self.last_messages.append(time.time())
                
            return True
            
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            self.logger.error(f"Error sending Telegram message: {error_msg}")
            
            # If we got a JSON response with error details, log it
            if hasattr(e, "response") and e.response:
                try:
                    error_json = e.response.json()
                    if "description" in error_json:
                        self.logger.error(f"Telegram API error: {error_json['description']}")
                except:
                    pass
                    
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending Telegram message: {str(e)}")
            return False
    
    def send_error(self, error, context=""):
        """
        Send an error message with traceback to Telegram and log it
        
        Args:
            error (Exception): Error object
            context (str): Error context
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            error_message = f"❌ *Error: {type(error).__name__}*\n"
            
            if context:
                error_message += f"*Context:* {context}\n"
                
            error_message += f"*Message:* {str(error)}\n\n"
            
            # Add traceback (shortened)
            tb = traceback.format_exc()
            tb_lines = tb.split("\n")
            
            if len(tb_lines) > 10:
                tb_short = "\n".join(tb_lines[:3] + ["..."] + tb_lines[-5:])
            else:
                tb_short = tb
                
            # Always log the full traceback
            self.logger.error(f"Error in {context}: {str(error)}\n{tb}")
            
            # For Telegram, use shortened traceback
            error_message += f"```\n{tb_short}\n```"
            
            return self.send_message(error_message, level="error")
            
        except Exception as e:
            self.logger.error(f"Error sending error message: {str(e)}")
            return False
    
    def _check_rate_limit(self):
        """
        Check if we're within rate limits
        
        Returns:
            bool: True if within limits, False otherwise
        """
        current_time = time.time()
        
        # Remove old messages (older than 1 minute)
        with self.message_lock:
            self.last_messages = [t for t in self.last_messages if current_time - t < 60]
            
            # Check if we're at the limit
            return len(self.last_messages) < self.rate_limit
    
    def close(self):
        """Clean up resources"""
        self._stop_command_polling()
        self.session.close()
