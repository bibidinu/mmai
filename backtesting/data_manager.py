import pandas as pd
import numpy as np
import ccxt
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import requests
import concurrent.futures
import pytz

class HistoricalDataManager:
    """
    Manager for downloading, storing, and processing historical market data.
    Provides methods to fetch data from exchanges and prepare it for backtesting.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the historical data manager.
        
        Args:
            data_dir: Directory to store historical data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("HistoricalDataManager")
        self.logger.setLevel(logging.INFO)
        
        # Initialize exchange connections
        self.exchanges = {}
    
    def initialize_exchange(self, exchange_id: str, config: Dict[str, Any] = None) -> None:
        """
        Initialize connection to an exchange.
        
        Args:
            exchange_id: CCXT exchange ID (e.g., 'bybit', 'binance')
            config: Optional configuration for the exchange
        """
        try:
            # Get the exchange class
            if not hasattr(ccxt, exchange_id):
                self.logger.error(f"Exchange {exchange_id} not supported by CCXT")
                return
            
            exchange_class = getattr(ccxt, exchange_id)
            
            # Initialize with config if provided
            if config:
                exchange = exchange_class(config)
            else:
                exchange = exchange_class()
            
            # Set common options
            exchange.enableRateLimit = True
            
            # Store exchange connection
            self.exchanges[exchange_id] = exchange
            
            self.logger.info(f"Initialized exchange: {exchange_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange {exchange_id}: {str(e)}")
    
    def download_historical_data(self, 
                               exchange_id: str, 
                               symbol: str, 
                               timeframe: str, 
                               start_date: str,
                               end_date: Optional[str] = None,
                               limit: int = 1000,
                               retry_count: int = 3,
                               rate_limit_pause: float = 1.0) -> pd.DataFrame:
        """
        Download historical OHLCV data from an exchange.
        
        Args:
            exchange_id: CCXT exchange ID
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to current date if None
            limit: Maximum number of candles per request
            retry_count: Number of retry attempts if request fails
            rate_limit_pause: Pause between requests in seconds
            
        Returns:
            DataFrame with historical OHLCV data
        """
        # Check if exchange is initialized
        if exchange_id not in self.exchanges:
            self.initialize_exchange(exchange_id)
        
        exchange = self.exchanges.get(exchange_id)
        if not exchange:
            self.logger.error(f"Exchange {exchange_id} not initialized")
            return pd.DataFrame()
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        
        if end_date:
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)
        
        # Check if timeframe is supported
        if timeframe not in exchange.timeframes:
            self.logger.error(f"Timeframe {timeframe} not supported by {exchange_id}")
            return pd.DataFrame()
        
        # Download data in chunks
        all_candles = []
        current_timestamp = start_timestamp
        
        while current_timestamp < end_timestamp:
            for retry in range(retry_count):
                try:
                    self.logger.info(f"Downloading {symbol} {timeframe} data from {exchange_id} starting at {datetime.fromtimestamp(current_timestamp/1000)}")
                    
                    # Fetch OHLCV data
                    candles = exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_timestamp,
                        limit=limit
                    )
                    
                    if not candles:
                        self.logger.warning(f"No data returned for {symbol} {timeframe} from {exchange_id}")
                        break
                    
                    all_candles.extend(candles)
                    
                    # Update timestamp for next request
                    last_timestamp = candles[-1][0]
                    
                    # Break if we reached the end or got less than requested
                    if last_timestamp >= end_timestamp or len(candles) < limit:
                        current_timestamp = end_timestamp  # Exit loop
                        break
                    
                    # Set timestamp for next chunk
                    current_timestamp = last_timestamp + 1
                    
                    # Pause to respect rate limits
                    time.sleep(rate_limit_pause)
                    
                    break  # Successful request, exit retry loop
                    
                except Exception as e:
                    self.logger.warning(f"Error downloading data (retry {retry+1}/{retry_count}): {str(e)}")
                    
                    if retry == retry_count - 1:
                        self.logger.error(f"Failed to download data after {retry_count} retries")
                        # Exit with data collected so far
                        current_timestamp = end_timestamp
                    else:
                        # Wait longer before retrying
                        time.sleep(rate_limit_pause * (retry + 1))
        
        # Convert to DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            self.logger.info(f"Downloaded {len(df)} candles for {symbol} {timeframe} from {exchange_id}")
            
            return df
        else:
            self.logger.warning(f"No data downloaded for {symbol} {timeframe} from {exchange_id}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, exchange_id: str = None) -> str:
        """
        Save historical data to CSV file.
        
        Args:
            df: DataFrame with historical data
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            exchange_id: Optional exchange ID to include in filename
            
        Returns:
            Path to saved file
        """
        if df.empty:
            self.logger.warning(f"Cannot save empty DataFrame for {symbol} {timeframe}")
            return ""
        
        # Create clean filename
        clean_symbol = symbol.replace('/', '_')
        filename = f"{clean_symbol}_{timeframe}"
        if exchange_id:
            filename = f"{exchange_id}_{filename}"
        
        filepath = self.data_dir / f"{filename}.csv"
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Saved {len(df)} candles to {filepath}")
        
        return str(filepath)
    
    def load_data(self, symbol: str, timeframe: str, exchange_id: str = None) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            exchange_id: Optional exchange ID included in filename
            
        Returns:
            DataFrame with historical data
        """
        # Create clean filename
        clean_symbol = symbol.replace('/', '_')
        filename = f"{clean_symbol}_{timeframe}"
        if exchange_id:
            filename = f"{exchange_id}_{filename}"
        
        filepath = self.data_dir / f"{filename}.csv"
        
        if not filepath.exists():
            self.logger.warning(f"Data file not found: {filepath}")
            return pd.DataFrame()
        
        # Load from CSV
        df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.logger.info(f"Loaded {len(df)} candles from {filepath}")
        
        return df
    
    def update_data(self, 
                  symbol: str, 
                  timeframe: str, 
                  exchange_id: str,
                  days_if_empty: int = 365) -> pd.DataFrame:
        """
        Update existing data file with latest data from exchange.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            exchange_id: Exchange ID
            days_if_empty: Number of days to download if file doesn't exist
            
        Returns:
            Updated DataFrame with historical data
        """
        # Try to load existing data
        df = self.load_data(symbol, timeframe, exchange_id)
        
        if df.empty:
            # No existing data, download from scratch
            start_date = (datetime.now() - timedelta(days=days_if_empty)).strftime("%Y-%m-%d")
            self.logger.info(f"No existing data for {symbol} {timeframe}, downloading {days_if_empty} days from {start_date}")
            
            df = self.download_historical_data(
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date
            )
        else:
            # Get last timestamp
            last_timestamp = df['timestamp'].max()
            
            # Add small buffer to avoid duplicates (1 candle duration)
            timeframe_minutes = self._convert_timeframe_to_minutes(timeframe)
            start_date = last_timestamp + timedelta(minutes=timeframe_minutes)
            
            # Don't update if last candle is recent
            now = datetime.now(tz=pytz.UTC) if last_timestamp.tzinfo else datetime.now()
            if (now - last_timestamp).total_seconds() < timeframe_minutes * 60 * 2:
                self.logger.info(f"Data for {symbol} {timeframe} is already up to date")
                return df
            
            # Download new data
            self.logger.info(f"Updating data for {symbol} {timeframe} from {start_date}")
            
            new_df = self.download_historical_data(
                exchange_id=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime("%Y-%m-%d")
            )
            
            if not new_df.empty:
                # Combine old and new data
                df = pd.concat([df, new_df])
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                # Reset index
                df = df.reset_index(drop=True)
                
                self.logger.info(f"Added {len(new_df)} new candles for {symbol} {timeframe}")
        
        # Save updated data
        if not df.empty:
            self.save_data(df, symbol, timeframe, exchange_id)
        
        return df
    
    def _convert_timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert CCXT timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            Number of minutes in the timeframe
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
        elif unit == 'w':
            return value * 60 * 24 * 7
        else:
            self.logger.warning(f"Unknown timeframe unit: {unit}")
            return 0
    
    def download_multiple_symbols(self, 
                               exchange_id: str, 
                               symbols: List[str], 
                               timeframes: List[str],
                               start_date: str,
                               end_date: Optional[str] = None,
                               workers: int = 4) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Download historical data for multiple symbols and timeframes.
        
        Args:
            exchange_id: CCXT exchange ID
            symbols: List of trading pair symbols
            timeframes: List of timeframes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to current date if None
            workers: Number of parallel workers
            
        Returns:
            Nested dictionary with DataFrames indexed by symbol and timeframe
        """
        results = {}
        
        # Function to download single symbol/timeframe
        def download_task(symbol, timeframe):
            try:
                df = self.download_historical_data(
                    exchange_id=exchange_id,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    self.save_data(df, symbol, timeframe, exchange_id)
                
                return (symbol, timeframe, df)
            except Exception as e:
                self.logger.error(f"Error downloading {symbol} {timeframe}: {str(e)}")
                return (symbol, timeframe, pd.DataFrame())
        
        # Create tasks
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append((symbol, timeframe))
        
        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_task, symbol, timeframe) for symbol, timeframe in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    symbol, timeframe, df = future.result()
                    
                    if symbol not in results:
                        results[symbol] = {}
                    
                    results[symbol][timeframe] = df
                    
                except Exception as e:
                    self.logger.error(f"Error processing download task: {str(e)}")
        
        return results
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators to a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # SMA
        result['sma_20'] = result['close'].rolling(window=20).mean()
        result['sma_50'] = result['close'].rolling(window=50).mean()
        result['sma_200'] = result['close'].rolling(window=200).mean()
        
        # EMA
        result['ema_12'] = result['close'].ewm(span=12, adjust=False).mean()
        result['ema_26'] = result['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        result['macd'] = result['ema_12'] - result['ema_26']
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        # RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        result['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        result['bb_middle'] = result['close'].rolling(window=20).mean()
        result['bb_std'] = result['close'].rolling(window=20).std()
        result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
        result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
        
        # ATR (Average True Range)
        high_low = result['high'] - result['low']
        high_close = (result['high'] - result['close'].shift()).abs()
        low_close = (result['low'] - result['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        result['atr_14'] = true_range.rolling(14).mean()
        
        # Volume indicators
        result['volume_sma_20'] = result['volume'].rolling(window=20).mean()
        result['volume_ratio'] = result['volume'] / result['volume_sma_20']
        
        return result
    
    def get_market_data(self, 
                      symbol: str, 
                      timeframe: str, 
                      exchange_id: str = None,
                      add_indicators: bool = True) -> pd.DataFrame:
        """
        Load market data with optional auto-update and indicators.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            exchange_id: Optional exchange ID
            add_indicators: Whether to add technical indicators
            
        Returns:
            DataFrame with market data
        """
        # Try to load existing data
        df = self.load_data(symbol, timeframe, exchange_id)
        
        # Add indicators if requested
        if not df.empty and add_indicators:
            df = self.add_technical_indicators(df)
        
        return df
    
    def split_train_test(self, 
                       df: pd.DataFrame, 
                       train_ratio: float = 0.7, 
                       val_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            df: DataFrame with historical data
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            
        Returns:
            Dict with 'train', 'val', and 'test' DataFrames
        """
        if df.empty:
            return {'train': pd.DataFrame(), 'val': pd.DataFrame(), 'test': pd.DataFrame()}
        
        # Calculate split points
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
