"""
Risk Management and Position Sizing
"""
import logging
import numpy as np
import pandas as pd

class RiskManager:
    """
    Handles risk management and position sizing
    """
    
    def __init__(self, config):
        """
        Initialize the risk manager
        
        Args:
            config (dict): Risk management configuration
        """
        self.logger = logging.getLogger("risk_manager")
        self.config = config
        
        # Base risk parameters
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.01)  # 1% of account per trade
        self.min_risk_per_trade = config.get("min_risk_per_trade", 0.001)  # 0.1% of account per trade
        self.max_portfolio_risk = config.get("max_portfolio_risk", 0.05)  # 5% of account at risk
        self.max_correlated_risk = config.get("max_correlated_risk", 0.03)  # 3% in correlated assets
        
        # Position sizing parameters
        self.min_position_size = config.get("min_position_size", 0.001)  # Minimum position size (% of account)
        self.max_position_size = config.get("max_position_size", 0.2)  # Maximum position size (% of account)
        self.position_sizing_method = config.get("position_sizing_method", "volatility")  # volatility, fixed, or kelly
        
        # Leverage settings
        self.use_leverage = config.get("use_leverage", False)
        self.base_leverage = config.get("base_leverage", 1.0)  # Base leverage (1x = no leverage)
        self.max_leverage = config.get("max_leverage", 3.0)  # Maximum allowed leverage
        
        # Stop loss settings
        self.default_sl_atr_multiple = config.get("default_sl_atr_multiple", 1.5)  # Default stop loss = 1.5 * ATR
        self.min_sl_distance = config.get("min_sl_distance", 0.005)  # Minimum 0.5% stop loss distance
        
        # Circuit breaker settings
        self.daily_loss_limit = config.get("daily_loss_limit", -0.05)  # -5% daily loss limit
        self.weekly_loss_limit = config.get("weekly_loss_limit", -0.1)  # -10% weekly loss limit
        self.circuit_breaker_activated = False
        
        self.logger.info("Risk manager initialized")
    
    def calculate_position_size(self, symbol, direction, account_balance, volatility):
        """
        Calculate optimal position size based on risk parameters and market conditions
        
        Args:
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            account_balance (float): Current account balance
            volatility (float): Current volatility (ATR)
            
        Returns:
            float: Position size in quote currency
        """
        # Check if circuit breaker is activated
        if self.circuit_breaker_activated:
            self.logger.warning("Circuit breaker activated, no new positions allowed")
            return 0
        
        # Calculate position size based on selected method
        if self.position_sizing_method == "fixed":
            position_size = self._fixed_position_sizing(account_balance)
        elif self.position_sizing_method == "kelly":
            position_size = self._kelly_position_sizing(symbol, direction, account_balance)
        else:  # Default to volatility-based
            position_size = self._volatility_position_sizing(symbol, direction, account_balance, volatility)
        
        # Ensure position size is within allowed range
        min_size = account_balance * self.min_position_size
        max_size = account_balance * self.max_position_size
        
        position_size = max(min_size, min(position_size, max_size))
        
        # Apply leverage if enabled
        if self.use_leverage:
            leverage = self._calculate_optimal_leverage(symbol, volatility)
            position_size *= leverage
            
        self.logger.info(f"Calculated position size for {symbol} {direction}: {position_size:.2f} ({(position_size/account_balance)*100:.2f}% of account)")
        
        return position_size
    
    def calculate_stop_loss(self, symbol, direction, entry_price, volatility):
        """
        Calculate stop loss price based on volatility and market conditions
        
        Args:
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            entry_price (float): Position entry price
            volatility (float): Current volatility (ATR)
            
        Returns:
            float: Stop loss price
        """
        # Calculate ATR-based stop loss distance
        sl_distance = volatility * self.default_sl_atr_multiple
        
        # Ensure minimum distance
        min_distance = entry_price * self.min_sl_distance
        sl_distance = max(sl_distance, min_distance)
        
        # Calculate stop loss price
        if direction == "long":
            sl_price = entry_price - sl_distance
        else:
            sl_price = entry_price + sl_distance
            
        self.logger.debug(f"Calculated stop loss for {symbol} {direction}: {sl_price:.5f}")
        
        return sl_price
    
    def check_risk_limits(self, new_trade_risk, current_portfolio_risk, correlation_matrix=None):
        """
        Check if a new trade would exceed risk limits
        
        Args:
            new_trade_risk (float): Risk amount of the new trade
            current_portfolio_risk (float): Current portfolio risk
            correlation_matrix (pd.DataFrame, optional): Correlation matrix for assets
            
        Returns:
            bool: True if trade is allowed, False if it would exceed limits
        """
        # Check if circuit breaker is activated
        if self.circuit_breaker_activated:
            return False
        
        # Check portfolio risk limit
        if current_portfolio_risk + new_trade_risk > self.max_portfolio_risk:
            self.logger.warning(f"Trade rejected: Would exceed max portfolio risk of {self.max_portfolio_risk*100}%")
            return False
        
        # Check correlated risk if correlation matrix is provided
        if correlation_matrix is not None:
            # Implementation depends on the format of correlation_matrix
            # This is a simplified check
            return True
        
        return True
    
    def update_circuit_breaker(self, daily_pnl, weekly_pnl):
        """
        Update circuit breaker status based on P&L
        
        Args:
            daily_pnl (float): Daily P&L as a percentage
            weekly_pnl (float): Weekly P&L as a percentage
            
        Returns:
            bool: True if circuit breaker is activated, False otherwise
        """
        if daily_pnl <= self.daily_loss_limit or weekly_pnl <= self.weekly_loss_limit:
            self.circuit_breaker_activated = True
            self.logger.warning(f"Circuit breaker activated: Daily P&L: {daily_pnl:.2%}, Weekly P&L: {weekly_pnl:.2%}")
        else:
            self.circuit_breaker_activated = False
            
        return self.circuit_breaker_activated
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker status"""
        self.circuit_breaker_activated = False
        self.logger.info("Circuit breaker reset")
    
    def _fixed_position_sizing(self, account_balance):
        """
        Calculate position size using fixed percentage of account
        
        Args:
            account_balance (float): Current account balance
            
        Returns:
            float: Position size in quote currency
        """
        risk_percentage = self.config.get("fixed_risk_percentage", 0.01)  # Default 1%
        return account_balance * risk_percentage
    
    def _kelly_position_sizing(self, symbol, direction, account_balance):
        """
        Calculate position size using Kelly criterion
        
        Args:
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            account_balance (float): Current account balance
            
        Returns:
            float: Position size in quote currency
        """
        # Kelly formula: K = W - (1-W)/R
        # Where W = win rate, R = win/loss ratio
        
        # Get historical performance for strategy/symbol
        win_rate = self.config.get("default_win_rate", 0.55)
        win_loss_ratio = self.config.get("default_win_loss_ratio", 1.5)
        
        # Apply Kelly formula with a fraction
        kelly_percentage = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply Kelly fraction (usually 0.1 to 0.5 of Kelly)
        kelly_fraction = self.config.get("kelly_fraction", 0.3)
        position_size = account_balance * kelly_percentage * kelly_fraction
        
        # Kelly can sometimes give negative or large values, so cap it
        return max(0, min(position_size, account_balance * self.max_position_size))
    
    def _volatility_position_sizing(self, symbol, direction, account_balance, volatility):
        """
        Calculate position size based on volatility
        
        Args:
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            account_balance (float): Current account balance
            volatility (float): Current volatility (ATR)
            
        Returns:
            float: Position size in quote currency
        """
        # Use ATR to determine stop loss distance
        sl_distance_pct = volatility / self.config.get("avg_price", 1.0)
        
        # Risk amount in account currency
        risk_amount = account_balance * self.max_risk_per_trade
        
        # Adjust risk based on volatility regime
        volatility_multiple = 1.0
        
        if sl_distance_pct > 0.03:  # High volatility
            volatility_multiple = 0.7  # Reduce position size
        elif sl_distance_pct < 0.01:  # Low volatility
            volatility_multiple = 1.3  # Increase position size
            
        risk_amount *= volatility_multiple
        
        # Calculate position size based on risk and stop loss
        if sl_distance_pct > 0:
            position_size = risk_amount / sl_distance_pct
        else:
            position_size = 0
            
        return position_size
    
    def _calculate_optimal_leverage(self, symbol, volatility):
        """
        Calculate optimal leverage based on market volatility
        
        Args:
            symbol (str): Trading symbol
            volatility (float): Current volatility (ATR)
            
        Returns:
            float: Optimal leverage multiple
        """
        # Base leverage from config
        leverage = self.base_leverage
        
        # Adjust based on volatility
        vol_factor = self.config.get("avg_volatility", 0.01) / max(volatility, 0.0001)
        leverage = min(leverage * vol_factor, self.max_leverage)
        
        # Never go below 1x
        leverage = max(1.0, leverage)
        
        self.logger.debug(f"Calculated leverage for {symbol}: {leverage:.2f}x")
        
        return leverage
