import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import matplotlib.pyplot as plt
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple, Callable, Optional, Union, Any

class BacktestEngine:
    """
    Backtesting engine for cryptocurrency trading strategies.
    Allows testing of strategies against historical data and optimization of parameters.
    """
    
    def __init__(self, data_dir: str = "./data", results_dir: str = "./backtest_results"):
        """
        Initialize the backtesting engine.
        
        Args:
            data_dir: Directory containing historical data files
            results_dir: Directory to save backtest results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("BacktestEngine")
        self.logger.setLevel(logging.INFO)
        
        self.historical_data = {}
        self.results = {}
        self.strategies = {}
        
    def load_data(self, symbol: str, timeframe: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load historical market data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
            start_date: Start date for data filtering (YYYY-MM-DD)
            end_date: End date for data filtering (YYYY-MM-DD)
            
        Returns:
            DataFrame with historical price data
        """
        file_path = self.data_dir / f"{symbol}_{timeframe}.csv"
        
        if not file_path.exists():
            self.logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        # Filter by date range if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
            
        # Store loaded data
        key = f"{symbol}_{timeframe}"
        self.historical_data[key] = data
        
        self.logger.info(f"Loaded {len(data)} candles for {symbol} {timeframe}")
        return data
    
    def register_strategy(self, strategy_name: str, strategy_class: Any) -> None:
        """
        Register a trading strategy for backtesting.
        
        Args:
            strategy_name: Name of the strategy
            strategy_class: Strategy class with initialize, should_enter, and should_exit methods
        """
        self.strategies[strategy_name] = strategy_class
        self.logger.info(f"Registered strategy: {strategy_name}")
    
    def run_backtest(self, 
                    strategy_name: str, 
                    symbol: str, 
                    timeframe: str, 
                    params: Dict[str, Any],
                    initial_capital: float = 10000.0,
                    risk_per_trade: float = 0.02,
                    commission: float = 0.001) -> Dict[str, Any]:
        """
        Run a backtest for a specific strategy and parameters.
        
        Args:
            strategy_name: Name of the registered strategy to test
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            params: Strategy parameters
            initial_capital: Initial capital for the backtest
            risk_per_trade: Risk percentage per trade (0.02 = 2%)
            commission: Trading commission rate
            
        Returns:
            Dict containing backtest results and performance metrics
        """
        # Check if strategy exists
        if strategy_name not in self.strategies:
            self.logger.error(f"Strategy not found: {strategy_name}")
            raise ValueError(f"Strategy not found: {strategy_name}")
        
        # Get data
        data_key = f"{symbol}_{timeframe}"
        if data_key not in self.historical_data:
            self.load_data(symbol, timeframe)
        
        data = self.historical_data[data_key].copy()
        
        # Initialize strategy
        strategy = self.strategies[strategy_name]()
        strategy.initialize(params)
        
        # Initialize backtest variables
        equity = [initial_capital]
        positions = []
        trades = []
        current_position = None
        
        # Run through each candle
        for i in range(1, len(data)):
            # Skip the first candle to have access to previous data
            current_candle = data.iloc[i]
            prev_candle = data.iloc[i-1]
            candle_dict = current_candle.to_dict()
            
            # Check for exit if in position
            if current_position:
                # Check if stop loss hit
                if current_position['direction'] == 'long' and current_candle['low'] <= current_position['stop_loss']:
                    # Stop loss hit for long position
                    exit_price = current_position['stop_loss']
                    profit_loss = (exit_price / current_position['entry_price'] - 1) * current_position['size'] - (commission * current_position['size'])
                    current_position['exit_price'] = exit_price
                    current_position['exit_date'] = current_candle.name
                    current_position['profit_loss'] = profit_loss
                    current_position['exit_reason'] = 'stop_loss'
                    trades.append(current_position)
                    equity.append(equity[-1] + profit_loss)
                    current_position = None
                    
                elif current_position['direction'] == 'short' and current_candle['high'] >= current_position['stop_loss']:
                    # Stop loss hit for short position
                    exit_price = current_position['stop_loss']
                    profit_loss = (current_position['entry_price'] / exit_price - 1) * current_position['size'] - (commission * current_position['size'])
                    current_position['exit_price'] = exit_price
                    current_position['exit_date'] = current_candle.name
                    current_position['profit_loss'] = profit_loss
                    current_position['exit_reason'] = 'stop_loss'
                    trades.append(current_position)
                    equity.append(equity[-1] + profit_loss)
                    current_position = None
                
                # Check for take profit
                elif current_position['direction'] == 'long' and current_candle['high'] >= current_position['take_profit']:
                    # Take profit hit for long position
                    exit_price = current_position['take_profit']
                    profit_loss = (exit_price / current_position['entry_price'] - 1) * current_position['size'] - (commission * current_position['size'])
                    current_position['exit_price'] = exit_price
                    current_position['exit_date'] = current_candle.name
                    current_position['profit_loss'] = profit_loss
                    current_position['exit_reason'] = 'take_profit'
                    trades.append(current_position)
                    equity.append(equity[-1] + profit_loss)
                    current_position = None
                    
                elif current_position['direction'] == 'short' and current_candle['low'] <= current_position['take_profit']:
                    # Take profit hit for short position
                    exit_price = current_position['take_profit']
                    profit_loss = (current_position['entry_price'] / exit_price - 1) * current_position['size'] - (commission * current_position['size'])
                    current_position['exit_price'] = exit_price
                    current_position['exit_date'] = current_candle.name
                    current_position['profit_loss'] = profit_loss
                    current_position['exit_reason'] = 'take_profit'
                    trades.append(current_position)
                    equity.append(equity[-1] + profit_loss)
                    current_position = None
                
                # Check if strategy exit condition met
                elif strategy.should_exit(current_candle, candle_dict, current_position['direction']):
                    # Strategy exit signal
                    if current_position['direction'] == 'long':
                        exit_price = current_candle['close']
                        profit_loss = (exit_price / current_position['entry_price'] - 1) * current_position['size'] - (commission * current_position['size'])
                    else:
                        exit_price = current_candle['close']
                        profit_loss = (current_position['entry_price'] / exit_price - 1) * current_position['size'] - (commission * current_position['size'])
                    
                    current_position['exit_price'] = exit_price
                    current_position['exit_date'] = current_candle.name
                    current_position['profit_loss'] = profit_loss
                    current_position['exit_reason'] = 'strategy_exit'
                    trades.append(current_position)
                    equity.append(equity[-1] + profit_loss)
                    current_position = None
                
                else:
                    # Still in position, update equity with unrealized P&L
                    if current_position['direction'] == 'long':
                        unrealized_pl = (current_candle['close'] / current_position['entry_price'] - 1) * current_position['size']
                    else:
                        unrealized_pl = (current_position['entry_price'] / current_candle['close'] - 1) * current_position['size']
                    
                    equity.append(equity[0] + sum([t['profit_loss'] for t in trades]) + unrealized_pl)
            
            # Check for entry if not in position
            if not current_position:
                entry_signal = strategy.should_enter(current_candle, candle_dict)
                
                if entry_signal and entry_signal['direction'] in ['long', 'short']:
                    # Calculate position size based on risk
                    risk_amount = equity[-1] * risk_per_trade
                    stop_loss = entry_signal['stop_loss']
                    
                    if entry_signal['direction'] == 'long':
                        risk_per_unit = (entry_signal['entry_price'] - stop_loss) / entry_signal['entry_price']
                    else:
                        risk_per_unit = (stop_loss - entry_signal['entry_price']) / entry_signal['entry_price']
                    
                    position_size = risk_amount / (risk_per_unit * entry_signal['entry_price'])
                    
                    # Create position
                    current_position = {
                        'symbol': symbol,
                        'direction': entry_signal['direction'],
                        'entry_price': entry_signal['entry_price'],
                        'stop_loss': stop_loss,
                        'take_profit': entry_signal['take_profit'],
                        'entry_date': current_candle.name,
                        'size': position_size,
                        'params': params.copy()
                    }
                    
                    positions.append(current_position.copy())
                else:
                    # No change in equity if no position
                    equity.append(equity[-1])
        
        # Close any open position at the end of the backtest
        if current_position:
            last_candle = data.iloc[-1]
            exit_price = last_candle['close']
            
            if current_position['direction'] == 'long':
                profit_loss = (exit_price / current_position['entry_price'] - 1) * current_position['size'] - (commission * current_position['size'])
            else:
                profit_loss = (current_position['entry_price'] / exit_price - 1) * current_position['size'] - (commission * current_position['size'])
            
            current_position['exit_price'] = exit_price
            current_position['exit_date'] = last_candle.name
            current_position['profit_loss'] = profit_loss
            current_position['exit_reason'] = 'backtest_end'
            trades.append(current_position)
            equity[-1] = equity[0] + sum([t['profit_loss'] for t in trades])
        
        # Calculate performance metrics
        total_trades = len(trades)
        if total_trades > 0:
            winning_trades = len([t for t in trades if t['profit_loss'] > 0])
            losing_trades = len([t for t in trades if t['profit_loss'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            profits = sum([t['profit_loss'] for t in trades if t['profit_loss'] > 0])
            losses = sum([t['profit_loss'] for t in trades if t['profit_loss'] <= 0])
            profit_factor = abs(profits / losses) if losses != 0 else float('inf')
            
            # Calculate drawdown
            peak = 0
            drawdowns = []
            for value in equity:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                drawdowns.append(drawdown)
            
            max_drawdown = max(drawdowns) * 100  # as percentage
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            equity_array = np.array(equity)
            daily_returns = np.diff(equity_array) / equity_array[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            # Calculate CAGR
            start_date = data.index[1]
            end_date = data.index[-1]
            years = (end_date - start_date).days / 365.25
            
            cagr = (equity[-1] / equity[0]) ** (1 / years) - 1 if years > 0 else 0
            
            # Final values
            final_equity = equity[-1]
            total_return = (final_equity / initial_capital - 1) * 100
            
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'cagr': cagr,
                'total_return': total_return,
                'final_equity': final_equity
            }
        else:
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'cagr': 0,
                'total_return': 0,
                'final_equity': initial_capital
            }
        
        # Store results
        result_id = f"{strategy_name}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        result = {
            'id': result_id,
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'params': params,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'commission': commission,
            'equity_curve': equity,
            'trades': trades,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results[result_id] = result
        
        # Save results to file
        self._save_result(result)
        
        return result
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        Save backtest result to file.
        
        Args:
            result: Backtest result dictionary
        """
        result_copy = result.copy()
        # Convert trades to serializable format
        for trade in result_copy['trades']:
            trade['entry_date'] = trade['entry_date'].isoformat()
            trade['exit_date'] = trade['exit_date'].isoformat() if isinstance(trade['exit_date'], pd.Timestamp) else trade['exit_date']
        
        # Save to file
        file_path = self.results_dir / f"{result['id']}.json"
        with open(file_path, 'w') as f:
            json.dump(result_copy, f, indent=2)
        
        self.logger.info(f"Saved result to {file_path}")
    
    def optimize_parameters(self, 
                           strategy_name: str, 
                           symbol: str, 
                           timeframe: str, 
                           param_grid: Dict[str, List[Any]],
                           initial_capital: float = 10000.0,
                           risk_per_trade: float = 0.02,
                           optimization_metric: str = 'sharpe_ratio',
                           n_jobs: int = -1) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            strategy_name: Name of the registered strategy to optimize
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            param_grid: Dictionary of parameter names and possible values
            initial_capital: Initial capital for the backtest
            risk_per_trade: Risk percentage per trade
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor', etc.)
            n_jobs: Number of parallel jobs (default: use all cores)
            
        Returns:
            Dict containing optimization results, best parameters, and backtest result with best parameters
        """
        # Generate all parameter combinations
        import itertools
        param_keys = param_grid.keys()
        param_values = param_grid.values()
        param_combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Optimizing {strategy_name} with {len(param_combinations)} parameter combinations")
        
        # Function to run backtest with specific parameters
        def run_with_params(params_tuple):
            params = dict(zip(param_keys, params_tuple))
            result = self.run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                params=params,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade
            )
            return (params, result)
        
        # Run backtests in parallel
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            for params, result in executor.map(run_with_params, param_combinations):
                results.append((params, result))
        
        # Sort results by optimization metric
        results.sort(key=lambda x: x[1]['metrics'][optimization_metric], reverse=True)
        
        # Get best result
        best_params, best_result = results[0]
        
        # Create optimization result
        optimization_result = {
            'strategy': strategy_name,
            'symbol': symbol,
            'timeframe': timeframe,
            'param_grid': param_grid,
            'optimization_metric': optimization_metric,
            'num_combinations': len(param_combinations),
            'best_params': best_params,
            'best_metric_value': best_result['metrics'][optimization_metric],
            'best_result_id': best_result['id'],
            'all_results': [(params, result['metrics'][optimization_metric]) for params, result in results],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save optimization result
        optimization_id = f"opt_{strategy_name}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        file_path = self.results_dir / f"{optimization_id}.json"
        with open(file_path, 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        self.logger.info(f"Optimization complete. Best {optimization_metric}: {best_result['metrics'][optimization_metric]}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'optimization_result': optimization_result,
            'best_params': best_params,
            'best_backtest': best_result
        }
    
    def compare_strategies(self, 
                          strategies: List[Tuple[str, Dict[str, Any]]], 
                          symbol: str, 
                          timeframe: str,
                          initial_capital: float = 10000.0,
                          risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """
        Compare multiple strategies or parameter sets.
        
        Args:
            strategies: List of tuples containing (strategy_name, params)
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            initial_capital: Initial capital for each backtest
            risk_per_trade: Risk percentage per trade
            
        Returns:
            Dict containing comparison results and metrics
        """
        # Run backtest for each strategy
        results = []
        for strategy_name, params in strategies:
            result = self.run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                params=params,
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade
            )
            results.append((strategy_name, params, result))
        
        # Extract metrics for comparison
        comparison = {
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'strategies': []
        }
        
        for strategy_name, params, result in results:
            comparison['strategies'].append({
                'strategy_name': strategy_name,
                'params': params,
                'metrics': result['metrics'],
                'result_id': result['id']
            })
        
        # Sort by total return
        comparison['strategies'].sort(key=lambda x: x['metrics']['total_return'], reverse=True)
        
        return comparison
    
    def plot_equity_curve(self, result_id: str, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve for a backtest result.
        
        Args:
            result_id: ID of backtest result
            save_path: Optional path to save the plot (if None, display only)
        """
        if result_id not in self.results:
            # Try to load from file
            file_path = self.results_dir / f"{result_id}.json"
            if not file_path.exists():
                self.logger.error(f"Result not found: {result_id}")
                return
            
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            self.results[result_id] = result
        else:
            result = self.results[result_id]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        equity = result['equity_curve']
        plt.plot(equity, label='Equity Curve')
        
        # Add trade markers
        for trade in result['trades']:
            entry_date_idx = result['equity_curve'].index(result['equity_curve'][0] + sum([t['profit_loss'] for t in result['trades'] if t['entry_date'] < trade['entry_date']]))
            exit_date_idx = result['equity_curve'].index(result['equity_curve'][0] + sum([t['profit_loss'] for t in result['trades'] if t['exit_date'] <= trade['exit_date']]))
            
            if trade['profit_loss'] > 0:
                plt.plot(exit_date_idx, equity[exit_date_idx], 'go', markersize=5)
            else:
                plt.plot(exit_date_idx, equity[exit_date_idx], 'ro', markersize=5)
        
        # Add title and labels
        plt.title(f"Equity Curve: {result['strategy']} on {result['symbol']} {result['timeframe']}")
        plt.xlabel('Trade Number')
        plt.ylabel('Equity')
        plt.grid(True)
        
        # Add metrics annotation
        metrics = result['metrics']
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Total Trades: {metrics['total_trades']}"
        )
        
        plt.figtext(0.15, 0.15, metrics_text, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved equity curve to {save_path}")
        else:
            plt.show()
    
    def plot_drawdown(self, result_id: str, save_path: Optional[str] = None) -> None:
        """
        Plot drawdown curve for a backtest result.
        
        Args:
            result_id: ID of backtest result
            save_path: Optional path to save the plot (if None, display only)
        """
        if result_id not in self.results:
            # Try to load from file
            file_path = self.results_dir / f"{result_id}.json"
            if not file_path.exists():
                self.logger.error(f"Result not found: {result_id}")
                return
            
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            self.results[result_id] = result
        else:
            result = self.results[result_id]
        
        # Calculate drawdown
        equity = result['equity_curve']
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100  # as percentage
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot drawdown curve
        plt.plot(drawdown, label='Drawdown', color='red')
        
        # Add title and labels
        plt.title(f"Drawdown: {result['strategy']} on {result['symbol']} {result['timeframe']}")
        plt.xlabel('Trade Number')
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        
        # Add max drawdown line
        max_dd = np.max(drawdown)
        plt.axhline(y=max_dd, color='blue', linestyle='--', label=f'Max Drawdown: {max_dd:.2f}%')
        plt.legend()
        
        # Save or display
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Saved drawdown curve to {save_path}")
        else:
            plt.show()
    
    def export_trades_report(self, result_id: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Export detailed trades report to CSV.
        
        Args:
            result_id: ID of backtest result
            output_path: Optional path to save the CSV (if None, return DataFrame only)
            
        Returns:
            DataFrame containing trade details
        """
        if result_id not in self.results:
            # Try to load from file
            file_path = self.results_dir / f"{result_id}.json"
            if not file_path.exists():
                self.logger.error(f"Result not found: {result_id}")
                return pd.DataFrame()
            
            with open(file_path, 'r') as f:
                result = json.load(f)
            
            self.results[result_id] = result
        else:
            result = self.results[result_id]
        
        # Create DataFrame from trades
        trades_df = pd.DataFrame(result['trades'])
        
        # Add additional metrics
        trades_df['return_pct'] = trades_df['profit_loss'] / result['initial_capital'] * 100
        trades_df['duration'] = pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])
        
        # Save to CSV if path provided
        if output_path:
            trades_df.to_csv(output_path, index=False)
            self.logger.info(f"Exported trades report to {output_path}")
        
        return trades_df
    
    def get_performance_summary(self, result_ids: List[str]) -> pd.DataFrame:
        """
        Generate performance summary for multiple backtest results.
        
        Args:
            result_ids: List of backtest result IDs
            
        Returns:
            DataFrame containing performance metrics for each result
        """
        summaries = []
        
        for result_id in result_ids:
            # Load result if not in memory
            if result_id not in self.results:
                file_path = self.results_dir / f"{result_id}.json"
                if not file_path.exists():
                    self.logger.warning(f"Result not found: {result_id}")
                    continue
                
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                self.results[result_id] = result
            else:
                result = self.results[result_id]
            
            # Extract summary info
            summary = {
                'result_id': result_id,
                'strategy': result['strategy'],
                'symbol': result['symbol'],
                'timeframe': result['timeframe'],
                **result['metrics']
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
