"""
Self-Learning and Optimization Engine
"""
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

class LearningManager:
    """
    Manages the self-learning and optimization processes
    """
    
    def __init__(self, config, db_manager):
        """
        Initialize the learning manager
        
        Args:
            config (dict): Learning configuration
            db_manager: Database manager for storing/retrieving data
        """
        self.logger = logging.getLogger("learning_manager")
        self.config = config
        self.db = db_manager
        
        # Learning parameters
        self.optimization_interval = config.get("optimization_interval", 86400)  # 24 hours
        self.min_trades_for_learning = config.get("min_trades_for_learning", 20)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 50)
        
        # Model parameters
        self.model_type = config.get("model_type", "reinforcement")  # 'reinforcement', 'classification', or 'regression'
        self.use_feature_selection = config.get("use_feature_selection", True)
        
        # State tracking
        self.last_optimization_time = 0
        self.model = None
        self.scaler = None
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info("Learning manager initialized")
    
    def _initialize_models(self):
        """Initialize or load ML models"""
        try:
            # Try to load existing models
            if self.model_type == "reinforcement":
                self._initialize_rl_model()
            elif self.model_type == "classification":
                self._initialize_classification_model()
            else:  # Regression model
                self._initialize_regression_model()
                
            self.logger.info(f"Initialized {self.model_type} model")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}", exc_info=True)
    
    def _initialize_rl_model(self):
        """Initialize reinforcement learning model"""
        # For Q-learning, initialize Q-table
        self.q_table = {}
        
        # Try to load existing Q-table
        try:
            q_table_data = self.db.get_model_data("q_table")
            if q_table_data:
                self.q_table = json.loads(q_table_data)
                self.logger.info(f"Loaded Q-table with {len(self.q_table)} states")
        except Exception as e:
            self.logger.warning(f"Could not load Q-table: {str(e)}")
    
    def _initialize_classification_model(self):
        """Initialize classification model"""
        # Try to load existing model
        try:
            model_data = self.db.get_model_data("classification_model")
            scaler_data = self.db.get_model_data("feature_scaler")
            
            if model_data and scaler_data:
                self.model = joblib.loads(model_data)
                self.scaler = joblib.loads(scaler_data)
                self.logger.info("Loaded classification model and scaler")
            else:
                # Create new model
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    random_state=42
                )
                self.scaler = StandardScaler()
                self.logger.info("Created new classification model")
                
        except Exception as e:
            self.logger.warning(f"Could not load classification model: {str(e)}")
            
            # Create new model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.logger.info("Created new classification model")
    
    def _initialize_regression_model(self):
        """Initialize regression/deep learning model"""
        # Try to load existing model
        try:
            model_path = self.config.get("model_path", "./models/regression_model.h5")
            scaler_data = self.db.get_model_data("feature_scaler")
            
            try:
                self.model = load_model(model_path)
                if scaler_data:
                    self.scaler = joblib.loads(scaler_data)
                    self.logger.info("Loaded regression model and scaler")
                else:
                    self.scaler = StandardScaler()
            except Exception:
                self._create_new_regression_model()
                
        except Exception as e:
            self.logger.warning(f"Could not load regression model: {str(e)}")
            self._create_new_regression_model()
    
    def _create_new_regression_model(self):
        """Create a new regression/deep learning model"""
        # Create a simple LSTM model
        self.model = Sequential()
        
        # Configuration
        input_dim = self.config.get("input_dim", 30)
        lstm_units = self.config.get("lstm_units", 64)
        dense_units = self.config.get("dense_units", 32)
        
        # LSTM layers
        self.model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(None, input_dim)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=lstm_units))
        self.model.add(Dropout(0.2))
        
        # Dense layers
        self.model.add(Dense(units=dense_units, activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.scaler = StandardScaler()
        self.logger.info("Created new regression model")
    
    def store_trade_entry(self, trade_id, symbol, direction, strategy, entry_price, 
                          position_size, tp_levels, stop_loss, technical_context, 
                          sentiment, volume_profile, correlations, volatility_regime):
        """
        Store entry data for a new trade
        
        Args:
            trade_id (str): Unique trade identifier
            symbol (str): Trading symbol
            direction (str): "long" or "short"
            strategy (str): Strategy name
            entry_price (float): Entry price
            position_size (float): Position size
            tp_levels (list): Take profit levels
            stop_loss (float): Stop loss price
            technical_context (dict): Technical analysis context
            sentiment (float): Sentiment score
            volume_profile (dict): Volume profile data
            correlations (dict): Correlation data
            volatility_regime (str): Volatility regime
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create trade entry record
            trade_data = {
                "trade_id": trade_id,
                "symbol": symbol,
                "direction": direction,
                "strategy": strategy,
                "entry_price": entry_price,
                "position_size": position_size,
                "tp_levels": tp_levels,
                "stop_loss": stop_loss,
                "entry_time": datetime.now().timestamp(),
                "technical_context": json.dumps(technical_context),
                "sentiment": sentiment,
                "volume_profile": json.dumps(volume_profile),
                "correlations": json.dumps(correlations),
                "volatility_regime": volatility_regime,
                "status": "open"
            }
            
            # Store in database
            success = self.db.store_trade_entry(trade_data)
            
            if success:
                self.logger.info(f"Stored entry data for trade {trade_id}")
            else:
                self.logger.error(f"Failed to store entry data for trade {trade_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing trade entry: {str(e)}", exc_info=True)
            return False
    
    def store_trade_exit(self, trade_id, exit_price, pnl, exit_reason):
        """
        Store exit data for a trade
        
        Args:
            trade_id (str): Unique trade identifier
            exit_price (float): Exit price
            pnl (float): Profit/loss amount
            exit_reason (str): Reason for exit
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create trade exit record
            exit_data = {
                "trade_id": trade_id,
                "exit_price": exit_price,
                "pnl": pnl,
                "exit_reason": exit_reason,
                "exit_time": datetime.now().timestamp(),
                "status": "closed"
            }
            
            # Store in database
            success = self.db.store_trade_exit(trade_id, exit_data)
            
            if success:
                self.logger.info(f"Stored exit data for trade {trade_id}")
                
                # Update Q-table if using reinforcement learning
                if self.model_type == "reinforcement":
                    self._update_q_table(trade_id, pnl)
            else:
                self.logger.error(f"Failed to store exit data for trade {trade_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing trade exit: {str(e)}", exc_info=True)
            return False
    
    def _update_q_table(self, trade_id, pnl):
        """
        Update Q-table based on trade result
        
        Args:
            trade_id (str): Trade ID
            pnl (float): Profit/loss amount
        """
        try:
            # Get trade data
            trade_data = self.db.get_trade(trade_id)
            
            if not trade_data:
                self.logger.warning(f"Trade {trade_id} not found for Q-table update")
                return
                
            # Extract state and action
            strategy = trade_data.get("strategy")
            
            # Create state representation (simplified)
            symbol = trade_data.get("symbol")
            direction = trade_data.get("direction")
            volatility_regime = trade_data.get("volatility_regime")
            
            # Technical context (extract key metrics)
            technical_context = json.loads(trade_data.get("technical_context", "{}"))
            trend = technical_context.get("trend", "neutral")
            
            # Create state key
            state_key = f"{symbol}_{volatility_regime}_{trend}"
            
            # Calculate reward (normalize PnL)
            position_size = trade_data.get("position_size", 1.0)
            reward = pnl / position_size if position_size > 0 else pnl
            
            # Scale reward to be between -1 and 1
            max_reward = self.config.get("max_reward_normalization", 0.1)  # 10% max reward
            reward = max(min(reward / max_reward, 1.0), -1.0)
            
            # Update Q-table
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
                
            if strategy not in self.q_table[state_key]:
                self.q_table[state_key][strategy] = 0.0
                
            # Q-learning update formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
            alpha = self.config.get("learning_rate", 0.1)
            gamma = self.config.get("discount_factor", 0.9)
            
            # Simplified update without max future reward (since we don't know future state)
            self.q_table[state_key][strategy] += alpha * (reward - self.q_table[state_key][strategy])
            
            # Save Q-table periodically
            if np.random.random() < 0.1:  # 10% chance to save
                self._save_q_table()
                
            self.logger.debug(f"Updated Q-table for state {state_key}, strategy {strategy}, reward {reward}")
            
        except Exception as e:
            self.logger.error(f"Error updating Q-table: {str(e)}", exc_info=True)
    
    def _save_q_table(self):
        """Save Q-table to database"""
        try:
            q_table_json = json.dumps(self.q_table)
            success = self.db.store_model_data("q_table", q_table_json)
            
            if success:
                self.logger.info(f"Saved Q-table with {len(self.q_table)} states")
            else:
                self.logger.error("Failed to save Q-table")
                
        except Exception as e:
            self.logger.error(f"Error saving Q-table: {str(e)}", exc_info=True)
    
    def should_optimize(self):
        """
        Check if it's time to run optimization
        
        Returns:
            bool: True if optimization should run, False otherwise
        """
        current_time = time.time()
        
        # Check if enough time has passed since last optimization
        if current_time - self.last_optimization_time < self.optimization_interval:
            return False
            
        # Check if we have enough trades to learn from
        recent_trades = self.db.get_closed_trades_count(
            since=current_time - self.config.get("learning_timeframe", 604800)  # Default 7 days
        )
        
        if recent_trades < self.min_trades_for_learning:
            return False
            
        return True
    
    def optimize(self):
        """
        Run the optimization process to improve models
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Starting model optimization")
        
        try:
            # Record optimization time
            self.last_optimization_time = time.time()
            
            # Get closed trades for training
            trades = self.db.get_closed_trades(
                limit=self.config.get("max_trades_for_learning", 1000),
                since=time.time() - self.config.get("learning_timeframe", 604800)
            )
            
            if not trades or len(trades) < self.min_trades_for_learning:
                self.logger.warning(f"Not enough trades for learning: {len(trades) if trades else 0}")
                return False
                
            self.logger.info(f"Optimizing model with {len(trades)} trades")
            
            # Run optimization based on model type
            if self.model_type == "reinforcement":
                success = self._optimize_rl_model(trades)
            elif self.model_type == "classification":
                success = self._optimize_classification_model(trades)
            else:  # Regression model
                success = self._optimize_regression_model(trades)
                
            if success:
                self.logger.info("Model optimization completed successfully")
            else:
                self.logger.error("Model optimization failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}", exc_info=True)
            return False
    
    def _optimize_rl_model(self, trades):
        """
        Optimize reinforcement learning model
        
        Args:
            trades (list): List of trade data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # For RL model, we've been updating the Q-table during trade exits
            # Here we can perform additional optimization
            
            # Calculate strategy performance metrics
            strategy_performance = self._calculate_strategy_performance(trades)
            
            # Save Q-table
            self._save_q_table()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing RL model: {str(e)}", exc_info=True)
            return False
    
    def _optimize_classification_model(self, trades):
        """
        Optimize classification model
        
        Args:
            trades (list): List of trade data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare training data
            X, y = self._prepare_training_data(trades)
            
            if len(X) == 0 or len(y) == 0:
                self.logger.warning("No training data available")
                return False
                
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Save model and scaler
            model_data = joblib.dumps(self.model)
            scaler_data = joblib.dumps(self.scaler)
            
            self.db.store_model_data("classification_model", model_data)
            self.db.store_model_data("feature_scaler", scaler_data)
            
            self.logger.info("Classification model trained and saved")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing classification model: {str(e)}", exc_info=True)
            return False
    
    def _optimize_regression_model(self, trades):
        """
        Optimize regression/deep learning model
        
        Args:
            trades (list): List of trade data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare training data
            X, y = self._prepare_training_data(trades)
            
            if len(X) == 0 or len(y) == 0:
                self.logger.warning("No training data available")
                return False
                
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Reshape for LSTM if needed
            if hasattr(self.model, 'layers') and any('lstm' in str(layer).lower() for layer in self.model.layers):
                X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            
            # Train model
            self.model.fit(
                X_scaled, 
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            # Save model
            model_path = self.config.get("model_path", "./models/regression_model.h5")
            self.model.save(model_path)
            
            # Save scaler
            scaler_data = joblib.dumps(self.scaler)
            self.db.store_model_data("feature_scaler", scaler_data)
            
            self.logger.info("Regression model trained and saved")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing regression model: {str(e)}", exc_info=True)
            return False
    
    def _prepare_training_data(self, trades):
        """
        Prepare training data from trade history
        
        Args:
            trades (list): List of trade data
            
        Returns:
            tuple: (X, y) features and labels
        """
        features = []
        labels = []
        
        for trade in trades:
            try:
                # Extract features
                feature_vector = self._extract_features(trade)
                
                if not feature_vector:
                    continue
                    
                # Extract label (1 for profitable, 0 for unprofitable)
                pnl = trade.get("pnl", 0)
                label = 1 if pnl > 0 else 0
                
                features.append(feature_vector)
                labels.append(label)
                
            except Exception as e:
                self.logger.error(f"Error processing trade for training: {str(e)}")
        
        return np.array(features), np.array(labels)
    
    def _extract_features(self, trade):
        """
        Extract features from trade data
        
        Args:
            trade (dict): Trade data
            
        Returns:
            list: Feature vector
        """
        try:
            feature_vector = []
            
            # Direction (1 for long, 0 for short)
            direction_feature = 1 if trade.get("direction") == "long" else 0
            feature_vector.append(direction_feature)
            
            # Volatility regime (convert to numeric)
            volatility_map = {"low": 0, "medium": 1, "high": 2}
            volatility_feature = volatility_map.get(trade.get("volatility_regime"), 1)
            feature_vector.append(volatility_feature)
            
            # Sentiment
            sentiment_feature = float(trade.get("sentiment", 0))
            feature_vector.append(sentiment_feature)
            
            # Technical context
            technical_context = json.loads(trade.get("technical_context", "{}"))
            
            # Trend (convert to numeric)
            trend_map = {"downtrend": -1, "neutral": 0, "uptrend": 1}
            trend_feature = trend_map.get(technical_context.get("trend"), 0)
            feature_vector.append(trend_feature)
            
            # RSI
            rsi_feature = float(technical_context.get("rsi", 50))
            feature_vector.append(rsi_feature)
            
            # Distance from support/resistance
            support_distance = float(technical_context.get("support_distance", 0))
            resistance_distance = float(technical_context.get("resistance_distance", 0))
            feature_vector.append(support_distance)
            feature_vector.append(resistance_distance)
            
            # Volume profile
            volume_profile = json.loads(trade.get("volume_profile", "{}"))
            volume_increase = float(volume_profile.get("volume_increase", 0))
            feature_vector.append(volume_increase)
            
            # Add more features as needed...
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def _calculate_strategy_performance(self, trades):
        """
        Calculate performance metrics for each strategy
        
        Args:
            trades (list): List of trade data
            
        Returns:
            dict: Strategy performance metrics
        """
        performance = {}
        
        for trade in trades:
            strategy = trade.get("strategy")
            
            if not strategy:
                continue
                
            if strategy not in performance:
                performance[strategy] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pnl": 0,
                    "win_rate": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "profit_factor": 0
                }
                
            # Update counts
            performance[strategy]["total_trades"] += 1
            
            pnl = trade.get("pnl", 0)
            performance[strategy]["total_pnl"] += pnl
            
            if pnl > 0:
                performance[strategy]["winning_trades"] += 1
                performance[strategy]["avg_win"] += pnl
            else:
                performance[strategy]["losing_trades"] += 1
                performance[strategy]["avg_loss"] += abs(pnl)
        
        # Calculate metrics
        for strategy, metrics in performance.items():
            total = metrics["total_trades"]
            winning = metrics["winning_trades"]
            losing = metrics["losing_trades"]
            
            if total > 0:
                metrics["win_rate"] = winning / total
                
            if winning > 0:
                metrics["avg_win"] = metrics["avg_win"] / winning
                
            if losing > 0:
                metrics["avg_loss"] = metrics["avg_loss"] / losing
                
            if metrics["avg_loss"] > 0:
                metrics["profit_factor"] = metrics["avg_win"] / metrics["avg_loss"]
        
        return performance
    
    def get_model(self):
        """
        Get the current model
        
        Returns:
            object: Current model
        """
        if self.model_type == "reinforcement":
            model = type('', (), {})()
            model.q_table = self.q_table
            return model
        else:
            return self.model
