"""
Database Management Module
"""
import logging
import sqlite3
import json
import pymongo
import time
import pandas as pd
import os
from datetime import datetime

class DatabaseManager:
    """
    Handles database operations for storing and retrieving trading data
    """
    
    def __init__(self, config):
        """
        Initialize the database manager
        
        Args:
            config (dict): Database configuration
        """
        self.logger = logging.getLogger("database_manager")
        self.config = config
        
        # Database type and connection
        self.db_type = config.get("type", "sqlite")
        self.connection = None
        
        # Connect to database
        self._connect()
        self._initialize_tables()
        
        self.logger.info(f"Database manager initialized ({self.db_type})")
    
    def _connect(self):
        """Connect to the database"""
        try:
            if self.db_type == "sqlite":
                db_path = self.config.get("path", "./data/trading.db")
                
                # Ensure parent directory exists
                db_parent_dir = os.path.dirname(db_path)
                if db_parent_dir and not os.path.exists(db_parent_dir):
                    os.makedirs(db_parent_dir, exist_ok=True)
                    self.logger.info(f"Created database directory: {db_parent_dir}")
                
                self.connection = sqlite3.connect(db_path, check_same_thread=False)
                self.connection.row_factory = sqlite3.Row
                
            elif self.db_type == "mongodb":
                uri = self.config.get("uri", "mongodb://localhost:27017/")
                db_name = self.config.get("database", "trading_bot")
                
                client = pymongo.MongoClient(uri)
                self.connection = client[db_name]
                
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
            self.logger.info(f"Connected to {self.db_type} database")
            
        except Exception as e:
            self.logger.error(f"Database connection error: {str(e)}", exc_info=True)
            raise
    
    def _initialize_tables(self):
        """Initialize database tables/collections"""
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Trades table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        trade_id TEXT PRIMARY KEY,
                        symbol TEXT,
                        direction TEXT,
                        strategy TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        position_size REAL,
                        tp_levels TEXT,
                        stop_loss REAL,
                        entry_time REAL,
                        exit_time REAL,
                        pnl REAL,
                        exit_reason TEXT,
                        technical_context TEXT,
                        sentiment REAL,
                        volume_profile TEXT,
                        correlations TEXT,
                        volatility_regime TEXT,
                        status TEXT
                    )
                ''')
                
                # Models table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT PRIMARY KEY,
                        model_data BLOB,
                        updated_at REAL
                    )
                ''')
                
                # Market data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        symbol TEXT,
                        timeframe TEXT,
                        timestamp REAL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        PRIMARY KEY (symbol, timeframe, timestamp)
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        date TEXT,
                        balance REAL,
                        daily_pnl REAL,
                        win_count INTEGER,
                        loss_count INTEGER,
                        strategies TEXT,
                        PRIMARY KEY (date)
                    )
                ''')
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                # MongoDB automatically creates collections when needed
                pass
                
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {str(e)}", exc_info=True)
    
    def store_trade_entry(self, trade_data):
        """
        Store entry data for a new trade
        
        Args:
            trade_data (dict): Trade entry data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Insert trade data
                cursor.execute('''
                    INSERT INTO trades (
                        trade_id, symbol, direction, strategy, entry_price, position_size,
                        tp_levels, stop_loss, entry_time, technical_context, sentiment,
                        volume_profile, correlations, volatility_regime, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data["trade_id"],
                    trade_data["symbol"],
                    trade_data["direction"],
                    trade_data["strategy"],
                    trade_data["entry_price"],
                    trade_data["position_size"],
                    json.dumps(trade_data["tp_levels"]),
                    trade_data["stop_loss"],
                    trade_data["entry_time"],
                    trade_data["technical_context"],
                    trade_data["sentiment"],
                    trade_data["volume_profile"],
                    trade_data["correlations"],
                    trade_data["volatility_regime"],
                    trade_data["status"]
                ))
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                # Store in trades collection
                trades_collection = self.connection["trades"]
                trades_collection.insert_one(trade_data)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing trade entry: {str(e)}", exc_info=True)
            return False
    
    def store_trade_exit(self, trade_id, exit_data):
        """
        Store exit data for a trade
        
        Args:
            trade_id (str): Trade ID
            exit_data (dict): Trade exit data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Update existing trade
                cursor.execute('''
                    UPDATE trades
                    SET exit_price = ?, exit_time = ?, pnl = ?, exit_reason = ?, status = ?
                    WHERE trade_id = ?
                ''', (
                    exit_data["exit_price"],
                    exit_data["exit_time"],
                    exit_data["pnl"],
                    exit_data["exit_reason"],
                    exit_data["status"],
                    trade_id
                ))
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                # Update trades collection
                trades_collection = self.connection["trades"]
                trades_collection.update_one(
                    {"trade_id": trade_id},
                    {"$set": exit_data}
                )
                
            # Update daily performance
            self._update_performance_metrics(trade_id, exit_data["pnl"])
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing trade exit: {str(e)}", exc_info=True)
            return False
    
    def _update_performance_metrics(self, trade_id, pnl):
        """
        Update daily performance metrics
        
        Args:
            trade_id (str): Trade ID
            pnl (float): Profit/loss amount
        """
        try:
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Get trade data
                cursor.execute('''
                    SELECT strategy FROM trades
                    WHERE trade_id = ?
                ''', (trade_id,))
                
                row = cursor.fetchone()
                if not row:
                    return
                    
                strategy = row[0]
                
                # Check if today's record exists
                cursor.execute('''
                    SELECT * FROM performance
                    WHERE date = ?
                ''', (today,))
                
                row = cursor.fetchone()
                
                if row:
                    # Update existing record
                    performance = dict(row)
                    
                    # Update strategies
                    strategies = json.loads(performance["strategies"])
                    if strategy in strategies:
                        strategies[strategy]["count"] += 1
                        strategies[strategy]["pnl"] += pnl
                    else:
                        strategies[strategy] = {"count": 1, "pnl": pnl}
                    
                    # Update win/loss count
                    if pnl > 0:
                        win_count = performance["win_count"] + 1
                        loss_count = performance["loss_count"]
                    else:
                        win_count = performance["win_count"]
                        loss_count = performance["loss_count"] + 1
                    
                    # Update daily PnL
                    daily_pnl = performance["daily_pnl"] + pnl
                    
                    # Update record
                    cursor.execute('''
                        UPDATE performance
                        SET daily_pnl = ?, win_count = ?, loss_count = ?, strategies = ?
                        WHERE date = ?
                    ''', (
                        daily_pnl,
                        win_count,
                        loss_count,
                        json.dumps(strategies),
                        today
                    ))
                    
                else:
                    # Create new record
                    strategies = {strategy: {"count": 1, "pnl": pnl}}
                    
                    cursor.execute('''
                        INSERT INTO performance (date, balance, daily_pnl, win_count, loss_count, strategies)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        today,
                        0,  # Balance will be updated separately
                        pnl,
                        1 if pnl > 0 else 0,
                        0 if pnl > 0 else 1,
                        json.dumps(strategies)
                    ))
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                # Get trade data
                trades_collection = self.connection["trades"]
                trade = trades_collection.find_one({"trade_id": trade_id})
                
                if not trade:
                    return
                    
                strategy = trade["strategy"]
                
                # Update performance collection
                performance_collection = self.connection["performance"]
                
                # Check if today's record exists
                performance = performance_collection.find_one({"date": today})
                
                if performance:
                    # Update existing record
                    strategies = performance.get("strategies", {})
                    
                    if strategy in strategies:
                        strategies[strategy]["count"] += 1
                        strategies[strategy]["pnl"] += pnl
                    else:
                        strategies[strategy] = {"count": 1, "pnl": pnl}
                    
                    # Update win/loss count
                    if pnl > 0:
                        win_count = performance["win_count"] + 1
                        loss_count = performance["loss_count"]
                    else:
                        win_count = performance["win_count"]
                        loss_count = performance["loss_count"] + 1
                    
                    # Update daily PnL
                    daily_pnl = performance["daily_pnl"] + pnl
                    
                    # Update record
                    performance_collection.update_one(
                        {"date": today},
                        {"$set": {
                            "daily_pnl": daily_pnl,
                            "win_count": win_count,
                            "loss_count": loss_count,
                            "strategies": strategies
                        }}
                    )
                    
                else:
                    # Create new record
                    strategies = {strategy: {"count": 1, "pnl": pnl}}
                    
                    performance_collection.insert_one({
                        "date": today,
                        "balance": 0,  # Balance will be updated separately
                        "daily_pnl": pnl,
                        "win_count": 1 if pnl > 0 else 0,
                        "loss_count": 0 if pnl > 0 else 1,
                        "strategies": strategies
                    })
                
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}", exc_info=True)
    
    def update_balance(self, balance):
        """
        Update current balance in performance metrics
        
        Args:
            balance (float): Current balance
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Check if today's record exists
                cursor.execute('''
                    SELECT * FROM performance
                    WHERE date = ?
                ''', (today,))
                
                row = cursor.fetchone()
                
                if row:
                    # Update existing record
                    cursor.execute('''
                        UPDATE performance
                        SET balance = ?
                        WHERE date = ?
                    ''', (balance, today))
                    
                else:
                    # Create new record
                    cursor.execute('''
                        INSERT INTO performance (date, balance, daily_pnl, win_count, loss_count, strategies)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        today,
                        balance,
                        0,
                        0,
                        0,
                        json.dumps({})
                    ))
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                performance_collection = self.connection["performance"]
                
                # Check if today's record exists
                performance = performance_collection.find_one({"date": today})
                
                if performance:
                    # Update existing record
                    performance_collection.update_one(
                        {"date": today},
                        {"$set": {"balance": balance}}
                    )
                    
                else:
                    # Create new record
                    performance_collection.insert_one({
                        "date": today,
                        "balance": balance,
                        "daily_pnl": 0,
                        "win_count": 0,
                        "loss_count": 0,
                        "strategies": {}
                    })
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating balance: {str(e)}", exc_info=True)
            return False
    
    def get_trade(self, trade_id):
        """
        Get trade data by ID
        
        Args:
            trade_id (str): Trade ID
            
        Returns:
            dict: Trade data or None if not found
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades
                    WHERE trade_id = ?
                ''', (trade_id,))
                
                row = cursor.fetchone()
                
                if row:
                    trade_data = dict(row)
                    
                    # Convert JSON strings to objects
                    if "tp_levels" in trade_data and trade_data["tp_levels"]:
                        trade_data["tp_levels"] = json.loads(trade_data["tp_levels"])
                    
                    return trade_data
                
                return None
                
            elif self.db_type == "mongodb":
                trades_collection = self.connection["trades"]
                return trades_collection.find_one({"trade_id": trade_id})
                
        except Exception as e:
            self.logger.error(f"Error getting trade: {str(e)}", exc_info=True)
            return None
    
    def get_open_trades(self):
        """
        Get all open trades
        
        Returns:
            list: List of open trades
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM trades
                    WHERE status = 'open'
                ''')
                
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trade_data = dict(row)
                    
                    # Convert JSON strings to objects
                    if "tp_levels" in trade_data and trade_data["tp_levels"]:
                        trade_data["tp_levels"] = json.loads(trade_data["tp_levels"])
                    
                    trades.append(trade_data)
                
                return trades
                
            elif self.db_type == "mongodb":
                trades_collection = self.connection["trades"]
                return list(trades_collection.find({"status": "open"}))
                
        except Exception as e:
            self.logger.error(f"Error getting open trades: {str(e)}", exc_info=True)
            return []
    
    def get_closed_trades(self, limit=100, since=None):
        """
        Get closed trades
        
        Args:
            limit (int): Maximum number of trades to return
            since (float, optional): Only return trades since this timestamp
            
        Returns:
            list: List of closed trades
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                if since:
                    cursor.execute('''
                        SELECT * FROM trades
                        WHERE status = 'closed' AND exit_time >= ?
                        ORDER BY exit_time DESC
                        LIMIT ?
                    ''', (since, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM trades
                        WHERE status = 'closed'
                        ORDER BY exit_time DESC
                        LIMIT ?
                    ''', (limit,))
                
                rows = cursor.fetchall()
                
                trades = []
                for row in rows:
                    trade_data = dict(row)
                    
                    # Convert JSON strings to objects
                    if "tp_levels" in trade_data and trade_data["tp_levels"]:
                        trade_data["tp_levels"] = json.loads(trade_data["tp_levels"])
                    
                    trades.append(trade_data)
                
                return trades
                
            elif self.db_type == "mongodb":
                trades_collection = self.connection["trades"]
                
                query = {"status": "closed"}
                if since:
                    query["exit_time"] = {"$gte": since}
                
                return list(trades_collection.find(query).sort("exit_time", -1).limit(limit))
                
        except Exception as e:
            self.logger.error(f"Error getting closed trades: {str(e)}", exc_info=True)
            return []
    
    def get_closed_trades_count(self, since=None):
        """
        Get count of closed trades
        
        Args:
            since (float, optional): Only count trades since this timestamp
            
        Returns:
            int: Number of closed trades
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                if since:
                    cursor.execute('''
                        SELECT COUNT(*) FROM trades
                        WHERE status = 'closed' AND exit_time >= ?
                    ''', (since,))
                else:
                    cursor.execute('''
                        SELECT COUNT(*) FROM trades
                        WHERE status = 'closed'
                    ''')
                
                return cursor.fetchone()[0]
                
            elif self.db_type == "mongodb":
                trades_collection = self.connection["trades"]
                
                query = {"status": "closed"}
                if since:
                    query["exit_time"] = {"$gte": since}
                
                return trades_collection.count_documents(query)
                
        except Exception as e:
            self.logger.error(f"Error getting closed trades count: {str(e)}", exc_info=True)
            return 0
    
    def store_model_data(self, model_id, model_data):
        """
        Store model data
        
        Args:
            model_id (str): Model identifier
            model_data: Model data to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Check if model exists
                cursor.execute('''
                    SELECT * FROM models
                    WHERE model_id = ?
                ''', (model_id,))
                
                if cursor.fetchone():
                    # Update existing model
                    cursor.execute('''
                        UPDATE models
                        SET model_data = ?, updated_at = ?
                        WHERE model_id = ?
                    ''', (
                        model_data,
                        time.time(),
                        model_id
                    ))
                else:
                    # Insert new model
                    cursor.execute('''
                        INSERT INTO models (model_id, model_data, updated_at)
                        VALUES (?, ?, ?)
                    ''', (
                        model_id,
                        model_data,
                        time.time()
                    ))
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                models_collection = self.connection["models"]
                
                # Update or insert model
                models_collection.update_one(
                    {"model_id": model_id},
                    {"$set": {
                        "model_data": model_data,
                        "updated_at": time.time()
                    }},
                    upsert=True
                )
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing model data: {str(e)}", exc_info=True)
            return False
    
    def get_model_data(self, model_id):
        """
        Get model data
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Model data or None if not found
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute('''
                    SELECT model_data FROM models
                    WHERE model_id = ?
                ''', (model_id,))
                
                row = cursor.fetchone()
                
                return row[0] if row else None
                
            elif self.db_type == "mongodb":
                models_collection = self.connection["models"]
                
                model = models_collection.find_one({"model_id": model_id})
                
                return model["model_data"] if model else None
                
        except Exception as e:
            self.logger.error(f"Error getting model data: {str(e)}", exc_info=True)
            return None
    
    def store_market_data(self, symbol, timeframe, data):
        """
        Store OHLCV market data
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            data (pd.DataFrame): OHLCV data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.db_type == "sqlite":
                if not isinstance(data, pd.DataFrame) or len(data) == 0:
                    return False
                    
                # Ensure timestamp column exists
                if "timestamp" not in data.columns:
                    self.logger.error("Missing timestamp column in market data")
                    return False
                
                cursor = self.connection.cursor()
                
                # Use executemany for better performance
                rows = []
                for _, row in data.iterrows():
                    rows.append((
                        symbol,
                        timeframe,
                        row["timestamp"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"] if "volume" in row else 0
                    ))
                
                cursor.executemany('''
                    INSERT OR REPLACE INTO market_data
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', rows)
                
                self.connection.commit()
                
            elif self.db_type == "mongodb":
                if not isinstance(data, pd.DataFrame) or len(data) == 0:
                    return False
                    
                market_data_collection = self.connection["market_data"]
                
                # Convert DataFrame to records
                records = data.to_dict("records")
                
                # Add symbol and timeframe to each record
                for record in records:
                    record["symbol"] = symbol
                    record["timeframe"] = timeframe
                
                # Use bulk operations for better performance
                operations = []
                for record in records:
                    operations.append(
                        pymongo.UpdateOne(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "timestamp": record["timestamp"]
                            },
                            {"$set": record},
                            upsert=True
                        )
                    )
                
                if operations:
                    market_data_collection.bulk_write(operations)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}", exc_info=True)
            return False
    
    def get_market_data(self, symbol, timeframe, start_time=None, end_time=None, limit=500):
        """
        Get OHLCV market data
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            start_time (float, optional): Start timestamp
            end_time (float, optional): End timestamp
            limit (int): Maximum number of candles to return
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                # Build query
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM market_data
                    WHERE symbol = ? AND timeframe = ?
                '''
                params = [symbol, timeframe]
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                
                rows = cursor.fetchall()
                
                # Convert to DataFrame
                if rows:
                    df = pd.DataFrame([dict(row) for row in rows])
                    return df.sort_values("timestamp").reset_index(drop=True)
                else:
                    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                
            elif self.db_type == "mongodb":
                market_data_collection = self.connection["market_data"]
                
                # Build query
                query = {
                    "symbol": symbol,
                    "timeframe": timeframe
                }
                
                if start_time:
                    query["timestamp"] = {"$gte": start_time}
                
                if end_time:
                    if "timestamp" in query:
                        query["timestamp"]["$lte"] = end_time
                    else:
                        query["timestamp"] = {"$lte": end_time}
                
                # Get data
                data = list(market_data_collection.find(
                    query,
                    {"_id": 0, "symbol": 0, "timeframe": 0}
                ).sort("timestamp", -1).limit(limit))
                
                # Convert to DataFrame
                if data:
                    df = pd.DataFrame(data)
                    return df.sort_values("timestamp").reset_index(drop=True)
                else:
                    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    def get_performance_metrics(self, days=30):
        """
        Get performance metrics
        
        Args:
            days (int): Number of days to return
            
        Returns:
            pd.DataFrame: Performance metrics
        """
        try:
            if self.db_type == "sqlite":
                cursor = self.connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM performance
                    ORDER BY date DESC
                    LIMIT ?
                ''', (days,))
                
                rows = cursor.fetchall()
                
                # Convert to DataFrame
                if rows:
                    data = []
                    for row in rows:
                        row_dict = dict(row)
                        
                        # Parse strategies JSON
                        if "strategies" in row_dict and row_dict["strategies"]:
                            row_dict["strategies"] = json.loads(row_dict["strategies"])
                            
                        data.append(row_dict)
                    
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame(columns=["date", "balance", "daily_pnl", "win_count", "loss_count", "strategies"])
                
            elif self.db_type == "mongodb":
                performance_collection = self.connection["performance"]
                
                # Get data
                data = list(performance_collection.find(
                    {},
                    {"_id": 0}
                ).sort("date", -1).limit(days))
                
                # Convert to DataFrame
                if data:
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame(columns=["date", "balance", "daily_pnl", "win_count", "loss_count", "strategies"])
                
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}", exc_info=True)
            return pd.DataFrame(columns=["date", "balance", "daily_pnl", "win_count", "loss_count", "strategies"])
    
    def close(self):
        """Close database connection"""
        try:
            if self.db_type == "sqlite" and self.connection:
                self.connection.close()
                
            elif self.db_type == "mongodb" and self.connection:
                self.connection.client.close()
                
            self.logger.info("Database connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}", exc_info=True)
