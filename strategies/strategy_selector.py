"""
Reinforcement Learning based Strategy Selector
"""
import numpy as np
import logging
from collections import defaultdict

# Strategy imports
from strategies.breakout import BreakoutStrategy
from strategies.breakdown import BreakdownStrategy
from strategies.sentiment_volume import SentimentVolumeStrategy
from strategies.range_trading import RangeStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.volatility import VolatilityStrategy
from strategies.liquidity import LiquidityStrategy
from strategies.arbitrage import ArbitrageStrategy
from strategies.on_chain import OnChainStrategy

class StrategySelector:
    """
    RL-based strategy selector that chooses the optimal strategy based on current market conditions
    """
    
    def __init__(self, config, technical, sentiment, volume, correlation, volatility):
        """
        Initialize the strategy selector
        
        Args:
            config (dict): Strategy configuration
            technical: Technical analysis module
            sentiment: Sentiment analysis module
            volume: Volume analysis module
            correlation: Correlation matrix module
            volatility: Volatility analysis module
        """
        self.logger = logging.getLogger("strategy_selector")
        self.config = config
        
        # Initialize market analyzers
        self.technical = technical
        self.sentiment = sentiment
        self.volume = volume
        self.correlation = correlation
        self.volatility = volatility
        
        # Initialize all strategies
        self.strategies = {
            "breakout": BreakoutStrategy(config.get("breakout", {}), technical, volume),
            "breakdown": BreakdownStrategy(config.get("breakdown", {}), technical, volume),
            "sentiment_volume": SentimentVolumeStrategy(config.get("sentiment_volume", {}), sentiment, volume),
            "range": RangeStrategy(config.get("range", {}), technical),
            "trend_following": TrendFollowingStrategy(config.get("trend_following", {}), technical),
            "mean_reversion": MeanReversionStrategy(config.get("mean_reversion", {}), technical),
            "volatility": VolatilityStrategy(config.get("volatility", {}), volatility),
            "liquidity": LiquidityStrategy(config.get("liquidity", {})),
            "arbitrage": ArbitrageStrategy(config.get("arbitrage", {})),
            "on_chain": OnChainStrategy(config.get("on_chain", {}))
        }
        
        # Initialize RL model
        self.initialize_model()
        
        # Track strategy performance
        self.strategy_performance = defaultdict(lambda: {
            "wins": 0,
            "losses": 0,
            "profit": 0.0,
            "usage_count": 0
        })
        
        self.logger.info("Strategy selector initialized with %d strategies", len(self.strategies))
    
    def initialize_model(self):
        """Initialize or load the reinforcement learning model"""
        # This is a placeholder for the RL model
        # In a real implementation, this would load a trained model
        # or initialize a new one if none exists
        
        # For simplicity, we'll use a Q-table approach initially
        self.q_table = defaultdict(lambda: defaultdict(float))
    
    def get_state_representation(self, market_data):
        """
        Create a state representation for the RL model based on market conditions
        
        Args:
            market_data (dict): Current market data
            
        Returns:
            tuple: State representation as a tuple for the Q-table
        """
        states = []
        
        # Track global market state
        global_trend = self.technical.get_global_trend()
        volatility_regime = self.volatility.get_global_volatility_regime()
        
        # Convert dictionaries to hashable types (tuples)
        if isinstance(global_trend, dict):
            global_trend = tuple(sorted(global_trend.items()))
        elif global_trend is None:
            global_trend = "neutral"
            
        if isinstance(volatility_regime, dict):
            volatility_regime = tuple(sorted(volatility_regime.items()))
        elif volatility_regime is None:
            volatility_regime = "medium"
        
        for symbol in market_data.keys():
            if symbol not in self.config["symbols_to_trade"]:
                continue
                
            # Determine market regime
            trend = self.technical.get_trend(symbol)
            in_range = self.technical.is_ranging(symbol)
            volatility = self.volatility.get_volatility_regime(symbol)
            sentiment_score = self.sentiment.get_sentiment_score(symbol)
            volume_anomaly = self.volume.has_volume_anomaly(symbol)
            liquidity = self.strategies["liquidity"].get_liquidity_score(symbol)
            
            # Ensure all values are hashable
            if isinstance(trend, dict):
                trend = tuple(sorted(trend.items()))
            if trend is None:
                trend = "neutral"
                
            if isinstance(in_range, dict):
                in_range = tuple(sorted(in_range.items()))
            if in_range is None:
                in_range = 0
                
            if isinstance(volatility, dict):
                volatility = tuple(sorted(volatility.items()))
            if volatility is None:
                volatility = "medium"
                
            if isinstance(sentiment_score, dict):
                sentiment_score = tuple(sorted(sentiment_score.items()))
            if sentiment_score is None:
                sentiment_score = 0
                
            if isinstance(volume_anomaly, dict):
                volume_anomaly = tuple(sorted(volume_anomaly.items()))
            if volume_anomaly is None:
                volume_anomaly = 0
                
            if isinstance(liquidity, dict):
                liquidity = tuple(sorted(liquidity.items()))
            if liquidity is None:
                liquidity = "medium"
            
            # Create a discrete state representation
            symbol_state = (
                symbol,
                trend,              # "uptrend", "downtrend", "neutral"
                1 if in_range else 0,
                volatility,         # "low", "medium", "high"
                1 if sentiment_score > 0.7 else (0 if sentiment_score < -0.7 else 0.5),
                1 if volume_anomaly else 0,
                liquidity           # "high", "medium", "low"
            )
            
            states.append(symbol_state)
        
        # Return tuple of global state and symbol states
        return (global_trend, volatility_regime, tuple(states))
    
    def select_strategy(self, market_data, current_positions):
        """
        Select the optimal strategy based on current market conditions
        
        Args:
            market_data (dict): Current market data
            current_positions (list): List of current open positions
            
        Returns:
            tuple: (selected_strategy_name, action, params)
                - selected_strategy_name (str): Name of the selected strategy
                - action (str): "enter", "exit", or None
                - params (dict): Parameters for the action
        """
        self.logger.debug("Selecting optimal strategy")
        
        # First, check if we need to exit any positions
        for position in current_positions:
            symbol = position["symbol"]
            direction = position["direction"]
            
            # Check if any strategy suggests exiting
            for strategy_name, strategy in self.strategies.items():
                exit_signal = strategy.should_exit(symbol, direction, position, market_data)
                
                if exit_signal:
                    self.logger.info(f"Exit signal from {strategy_name} for {symbol} {direction}")
                    return strategy_name, "exit", {
                        "symbol": symbol,
                        "direction": direction,
                        "position_id": position["id"],
                        "exit_reason": f"{strategy_name} exit signal"
                    }
        
        # If no exits, look for entry opportunities
        state = self.get_state_representation(market_data)
        
        # Use either exploitation or exploration
        if np.random.random() < self.config.get("exploration_rate", 0.1):
            # Exploration: try a random strategy
            strategy_name = np.random.choice(list(self.strategies.keys()))
            self.logger.debug(f"Exploration: selected {strategy_name}")
        else:
            # Exploitation: use Q-table to select best strategy
            if state in self.q_table:
                strategy_name = max(self.q_table[state], key=self.q_table[state].get)
                self.logger.debug(f"Exploitation: selected {strategy_name} from Q-table")
            else:
                # If state not in Q-table, select strategy based on market conditions
                strategy_name = self._select_strategy_heuristic(state, market_data)
                self.logger.debug(f"Heuristic: selected {strategy_name}")
        
        # Get selected strategy
        strategy = self.strategies[strategy_name]
        
        # Check for entry signals from the selected strategy
        for symbol in market_data.keys():
            if symbol not in self.config["symbols_to_trade"]:
                continue
                
            # Skip if we already have a position in this symbol
            if any(p["symbol"] == symbol for p in current_positions):
                continue
                
            entry_signal = strategy.should_enter(symbol, market_data)
            
            if entry_signal:
                direction = entry_signal.get("direction")
                entry_price = entry_signal.get("entry_price")
                
                self.logger.info(f"Entry signal from {strategy_name} for {symbol} {direction}")
                
                # Update strategy usage count
                self.strategy_performance[strategy_name]["usage_count"] += 1
                
                return strategy_name, "enter", {
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry_price,
                    "entry_reason": f"{strategy_name} entry signal"
                }
        
        # No actionable signals
        return None, None, None
    
    def _select_strategy_heuristic(self, state, market_data):
        """
        Select strategy based on heuristics when Q-table doesn't have the state
        
        Args:
            state: Current state representation
            market_data: Current market data
            
        Returns:
            str: Selected strategy name
        """
        global_trend, volatility_regime, symbol_states = state
        
        # Count market regimes across symbols
        uptrend_count = sum(1 for s in symbol_states if s[1] == "uptrend")
        downtrend_count = sum(1 for s in symbol_states if s[1] == "downtrend")
        ranging_count = sum(1 for s in symbol_states if s[2] == 1)
        high_vol_count = sum(1 for s in symbol_states if s[3] == "high")
        low_vol_count = sum(1 for s in symbol_states if s[3] == "low")
        
        # Select strategy based on predominant market regime
        if volatility_regime == "high":
            if uptrend_count > downtrend_count:
                return "breakout"
            else:
                return "breakdown"
        elif ranging_count > (uptrend_count + downtrend_count) / 2:
            if volatility_regime == "low":
                return "range"
            else:
                return "mean_reversion"
        elif uptrend_count > downtrend_count * 2:  # Strong uptrend
            return "trend_following"
        elif downtrend_count > uptrend_count * 2:  # Strong downtrend
            return "mean_reversion"
        elif high_vol_count > len(symbol_states) / 2:
            return "volatility"
        else:
            # Default to sentiment-volume or liquidity analysis
            return "sentiment_volume" if np.random.random() > 0.5 else "liquidity"
    
    def update_performance(self, strategy_name, win, profit):
        """
        Update the performance metrics for a strategy
        
        Args:
            strategy_name (str): Name of the strategy
            win (bool): Whether the trade was profitable
            profit (float): Profit/loss amount
        """
        if win:
            self.strategy_performance[strategy_name]["wins"] += 1
        else:
            self.strategy_performance[strategy_name]["losses"] += 1
            
        self.strategy_performance[strategy_name]["profit"] += profit
        
        # Update Q-table
        # This is a simplified update; real implementations would use proper RL algorithms
        if hasattr(self, 'last_state') and hasattr(self, 'last_action'):
            reward = profit
            self.q_table[self.last_state][strategy_name] += self.config.get("learning_rate", 0.1) * reward
    
    def update_model(self, model):
        """
        Update the RL model with a new one from the learning manager
        
        Args:
            model: New model from the learning manager
        """
        # In a real implementation, this would update the RL model
        # with a new one trained by the learning manager
        self.logger.info("Updating strategy selection model")
        
        # For our Q-table approach, we might merge the new Q-values
        if hasattr(model, 'q_table'):
            for state, actions in model.q_table.items():
                for action, value in actions.items():
                    self.q_table[state][action] = value