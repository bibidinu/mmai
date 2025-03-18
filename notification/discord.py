"""
Discord Notification Module with improved error handling
"""
import logging
import time
import json
import threading
import requests
from datetime import datetime
import traceback

class DiscordNotifier:
    """
    Handles notifications through Discord webhooks with improved error handling
    """
    
    def __init__(self, credentials, config):
        """
        Initialize the Discord notifier
        
        Args:
            credentials (dict): API credentials
            config (dict): Notification configuration
        """
        self.logger = logging.getLogger("discord_notifier")
        self.config = config
        
        # Discord webhook
        self.webhook_url = credentials.get("webhook_url", "")
        self.role_id = credentials.get("role_id", "")
        
        # Check if webhook is valid
        self.enabled = True
        if not self.webhook_url or self.webhook_url == "YOUR_DISCORD_WEBHOOK_URL":
            self.logger.warning("Invalid or missing Discord webhook URL - notifications will be logged only")
            self.enabled = False
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 5)  # Max messages per minute
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
        
        # Channel routing
        self.channel_routing = config.get("channel_routing", {
            "trade_entry": "trades",
            "trade_exit": "trades",
            "take_profit": "trades",
            "stop_loss": "trades",
            "error": "errors",
            "warning": "alerts",
            "info": "general",
            "debug": "general"
        })
        
        # Set up HTTP session with retry
        self.session = requests.Session()
        retry_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=5,
            pool_maxsize=5
        )
        self.session.mount('https://', retry_adapter)
        
        if self.enabled:
            self.logger.info("Discord notifier initialized and enabled")
        else:
            self.logger.info("Discord notifier initialized in logging-only mode")
    
    def send_message(self, message, level="info", username="Trading Bot"):
        """
        Send a message to Discord with improved error handling
        
        Args:
            message (str): Message text
            level (str): Notification level
            username (str): Username to show in Discord
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Always log the message
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, f"DISCORD: {message}")
        
        # If Discord is disabled, just log and return
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
        
        if level in level_emoji and not message.startswith(level_emoji[level]):
            message = f"{level_emoji[level]} {message}"
        
        # Add role mention if configured
        if self.role_id and level in ["error", "warning"]:
            message = f"<@&{self.role_id}> {message}"
        
        # Send to Discord
        return self._send_discord_message(message, username)
    
    def _send_discord_message(self, message, username="Trading Bot"):
        """
        Send a message to Discord webhook with improved error handling
        
        Args:
            message (str): Message text
            username (str): Username to show in Discord
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        try:
            # Prepare data
            payload = {
                "content": message,
                "username": username
            }
            
            # Handle long messages - Discord has a 2000 char limit
            if len(message) > 1950:
                # Split into multiple messages
                parts = []
                remaining = message
                
                while len(remaining) > 1950:
                    part = remaining[:1950]
                    # Try to split at a line break
                    last_newline = part.rfind('\n')
                    if last_newline > 1500:  # Only split at newline if it's not too close to the start
                        part = part[:last_newline]
                        remaining = remaining[last_newline:]
                    else:
                        remaining = remaining[1950:]
                    
                    parts.append(part)
                
                # Add the final part
                if remaining:
                    parts.append(remaining)
                
                # Send each part
                success = True
                for i, part in enumerate(parts):
                    part_payload = {
                        "content": f"Part {i+1}/{len(parts)}: {part}",
                        "username": username
                    }
                    
                    if not self._send_webhook_request(part_payload):
                        success = False
                    
                    # Avoid rate limiting by waiting between messages
                    if i < len(parts) - 1:
                        time.sleep(1)
                        
                return success
            else:
                # Send single message
                return self._send_webhook_request(payload)
                
        except Exception as e:
            self.logger.error(f"Unexpected error sending Discord message: {str(e)}")
            return False
    
    def _send_webhook_request(self, payload):
        """
        Send actual HTTP request to Discord webhook
        
        Args:
            payload (dict): Request payload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = self.session.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            # Add to rate limit tracking
            with self.message_lock:
                self.last_messages.append(time.time())
                
            return True
            
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            self.logger.error(f"Error sending Discord message: {error_msg}")
            
            # If we got a JSON response with error details, log it
            if hasattr(e, "response") and e.response:
                try:
                    error_json = e.response.json()
                    self.logger.error(f"Discord API error: {error_json}")
                except:
                    pass
                    
            return False
    
    def send_error(self, error, context=""):
        """
        Send an error message with traceback to Discord and log it
        
        Args:
            error (Exception): Error object
            context (str): Error context
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            error_message = f"❌ **Error: {type(error).__name__}**\n"
            
            if context:
                error_message += f"**Context:** {context}\n"
                
            error_message += f"**Message:** {str(error)}\n\n"
            
            # Add traceback (shortened)
            tb = traceback.format_exc()
            tb_lines = tb.split("\n")
            
            if len(tb_lines) > 10:
                tb_short = "\n".join(tb_lines[:3] + ["..."] + tb_lines[-5:])
            else:
                tb_short = tb
                
            # Always log the full traceback
            self.logger.error(f"Error in {context}: {str(error)}\n{tb}")
            
            # For Discord, use shortened traceback
            error_message += f"```\n{tb_short}\n```"
            
            return self.send_message(error_message, level="error", username="Error Reporter")
            
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
        self.session.close()
        self.logger.info("Discord notifier closed")
