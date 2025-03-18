import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from pathlib import Path
import time
import pytz
import io
import base64
import math
import jinja2
import pdfkit
import calendar
import csv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
import uuid
import qrcode
from io import BytesIO

class PerformanceCalculator:
    """Calculates trading performance metrics from trade data"""
    
    def __init__(self, timezone: str = 'UTC'):
        """
        Initialize the performance calculator
        
        Args:
            timezone: Timezone for date calculations
        """
        self.logger = logging.getLogger("PerformanceCalculator")
        self.logger.setLevel(logging.INFO)
        self.timezone = pytz.timezone(timezone)
    
    def calculate_metrics(self, trades: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
        """
        Calculate performance metrics from list of trades
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            self.logger.warning("No trades provided for performance calculation")
            return self._empty_metrics()
        
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(trades)
            
            # Ensure required columns exist
            required_cols = ['entry_time', 'exit_time', 'profit_loss', 'symbol', 'direction', 'strategy']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                self.logger.error(f"Missing required columns in trade data: {missing}")
                return self._empty_metrics()
            
            # Convert timestamps to datetime if they are not already
            if not pd.api.types.is_datetime64_dtype(df['entry_time']):
                df['entry_time'] = pd.to_datetime(df['entry_time'])
            
            if not pd.api.types.is_datetime64_dtype(df['exit_time']):
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            
            # Localize timestamps to the configured timezone
            df['entry_time'] = df['entry_time'].dt.tz_localize(pytz.UTC).dt.tz_convert(self.timezone)
            df['exit_time'] = df['exit_time'].dt.tz_localize(pytz.UTC).dt.tz_convert(self.timezone)
            
            # Sort by exit time
            df = df.sort_values('exit_time')
            
            # Calculate basic metrics
            total_trades = len(df)
            winning_trades = len(df[df['profit_loss'] > 0])
            losing_trades = len(df[df['profit_loss'] <= 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit metrics
            total_profit = df[df['profit_loss'] > 0]['profit_loss'].sum()
            total_loss = abs(df[df['profit_loss'] <= 0]['profit_loss'].sum())
            net_profit = total_profit - total_loss
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate average trade metrics
            avg_profit = df[df['profit_loss'] > 0]['profit_loss'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['profit_loss'] <= 0]['profit_loss'].mean() if losing_trades > 0 else 0
            avg_trade = df['profit_loss'].mean()
            
            # Calculate trade duration metrics
            df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600  # in hours
            avg_duration = df['duration'].mean()
            
            # Calculate return metrics
            returns = net_profit / initial_capital
            annualized_return = self._calculate_annualized_return(df, net_profit, initial_capital)
            
            # Calculate drawdown
            drawdown_info = self._calculate_drawdown(df, initial_capital)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(df, initial_capital)
            
            # Calculate per-symbol metrics
            symbols_metrics = self._calculate_per_symbol_metrics(df)
            
            # Calculate per-strategy metrics
            strategy_metrics = self._calculate_per_strategy_metrics(df)
            
            # Calculate per-direction metrics
            long_trades = len(df[df['direction'] == 'long'])
            short_trades = len(df[df['direction'] == 'short'])
            long_profit = df[df['direction'] == 'long']['profit_loss'].sum()
            short_profit = df[df['direction'] == 'short']['profit_loss'].sum()
            
            # Calculate monthly/weekly performance
            time_performance = self._calculate_time_performance(df)
            
            # Calculate consecutive win/loss streaks
            consecutive_metrics = self._calculate_consecutive_metrics(df)
            
            # Compile all metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net_profit': net_profit,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'avg_trade': avg_trade,
                'avg_duration': avg_duration,
                'return': returns,
                'annualized_return': annualized_return,
                'max_drawdown': drawdown_info['max_drawdown'],
                'max_drawdown_duration': drawdown_info['max_drawdown_duration'],
                'sharpe_ratio': sharpe_ratio,
                'symbols': symbols_metrics,
                'strategies': strategy_metrics,
                'long_trades': long_trades,
                'short_trades': short_trades,
                'long_profit': long_profit,
                'short_profit': short_profit,
                'monthly_performance': time_performance['monthly'],
                'weekly_performance': time_performance['weekly'],
                'max_consecutive_wins': consecutive_metrics['max_consecutive_wins'],
                'max_consecutive_losses': consecutive_metrics['max_consecutive_losses'],
                'current_consecutive_wins': consecutive_metrics['current_consecutive_wins'],
                'current_consecutive_losses': consecutive_metrics['current_consecutive_losses'],
                'equity_curve': self._calculate_equity_curve(df, initial_capital),
                'calculation_time': datetime.now(self.timezone).isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """
        Return empty metrics dictionary
        
        Returns:
            Dictionary with default metrics values
        """
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_loss': 0,
            'net_profit': 0,
            'profit_factor': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'avg_trade': 0,
            'avg_duration': 0,
            'return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'max_drawdown_duration': 0,
            'sharpe_ratio': 0,
            'symbols': {},
            'strategies': {},
            'long_trades': 0,
            'short_trades': 0,
            'long_profit': 0,
            'short_profit': 0,
            'monthly_performance': {},
            'weekly_performance': {},
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_consecutive_wins': 0,
            'current_consecutive_losses': 0,
            'equity_curve': [],
            'calculation_time': datetime.now(self.timezone).isoformat()
        }
    
    def _calculate_drawdown(self, df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
        """
        Calculate maximum drawdown and drawdown duration
        
        Args:
            df: DataFrame of trades
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary with drawdown metrics
        """
        if df.empty:
            return {'max_drawdown': 0, 'max_drawdown_duration': 0}
        
        # Sort by exit time
        df = df.sort_values('exit_time')
        
        # Calculate cumulative P&L
        cumulative_pnl = df['profit_loss'].cumsum()
        equity = initial_capital + cumulative_pnl
        
        # Calculate drawdown
        peak = equity.expanding().max()
        drawdown = (peak - equity) / peak
        max_drawdown = drawdown.max()
        
        # Calculate drawdown duration
        is_drawdown = drawdown > 0
        drawdown_start = df['exit_time'][is_drawdown & ~is_drawdown.shift(1).fillna(False)]
        drawdown_end = df['exit_time'][~is_drawdown & is_drawdown.shift(1).fillna(False)]
        
        if len(drawdown_start) > len(drawdown_end):
            # Currently in drawdown
            drawdown_end = drawdown_end.append(pd.Series([df['exit_time'].iloc[-1]]))
        
        durations = []
        for start, end in zip(drawdown_start, drawdown_end):
            durations.append((end - start).total_seconds() / (24 * 3600))  # in days
        
        max_drawdown_duration = max(durations) if durations else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration
        }
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, initial_capital: float) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            df: DataFrame of trades
            initial_capital: Initial capital amount
            
        Returns:
            Sharpe ratio
        """
        if df.empty:
            return 0.0
        
        # Group trades by day
        df['date'] = df['exit_time'].dt.date
        daily_returns = df.groupby('date')['profit_loss'].sum() / initial_capital
        
        # Calculate Sharpe ratio (annualized, assuming risk-free rate of 0)
        if len(daily_returns) < 2:
            return 0.0
        
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        return float(sharpe_ratio)
    
    def _calculate_annualized_return(self, df: pd.DataFrame, net_profit: float, initial_capital: float) -> float:
        """
        Calculate annualized return
        
        Args:
            df: DataFrame of trades
            net_profit: Net profit amount
            initial_capital: Initial capital amount
            
        Returns:
            Annualized return
        """
        if df.empty or net_profit <= 0:
            return 0.0
        
        # Calculate trading period in years
        first_trade = df['entry_time'].min()
        last_trade = df['exit_time'].max()
        trading_days = (last_trade - first_trade).days
        
        if trading_days < 1:
            return 0.0
        
        years = trading_days / 365.25
        return ((1 + (net_profit / initial_capital)) ** (1 / years)) - 1 if years > 0 else 0
    
    def _calculate_per_symbol_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics per trading symbol
        
        Args:
            df: DataFrame of trades
            
        Returns:
            Dictionary of metrics per symbol
        """
        if df.empty:
            return {}
        
        symbols = {}
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            
            total_trades = len(symbol_df)
            winning_trades = len(symbol_df[symbol_df['profit_loss'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = symbol_df[symbol_df['profit_loss'] > 0]['profit_loss'].sum()
            total_loss = abs(symbol_df[symbol_df['profit_loss'] <= 0]['profit_loss'].sum())
            net_profit = total_profit - total_loss
            
            symbols[symbol] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'net_profit': net_profit
            }
        
        return symbols
    
    def _calculate_per_strategy_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics per trading strategy
        
        Args:
            df: DataFrame of trades
            
        Returns:
            Dictionary of metrics per strategy
        """
        if df.empty or 'strategy' not in df.columns:
            return {}
        
        strategies = {}
        
        for strategy in df['strategy'].unique():
            strategy_df = df[df['strategy'] == strategy]
            
            total_trades = len(strategy_df)
            winning_trades = len(strategy_df[strategy_df['profit_loss'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = strategy_df[strategy_df['profit_loss'] > 0]['profit_loss'].sum()
            total_loss = abs(strategy_df[strategy_df['profit_loss'] <= 0]['profit_loss'].sum())
            net_profit = total_profit - total_loss
            
            strategies[strategy] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'net_profit': net_profit
            }
        
        return strategies
    
    def _calculate_time_performance(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics per time period
        
        Args:
            df: DataFrame of trades
            
        Returns:
            Dictionary of performance metrics by time period
        """
        if df.empty:
            return {'monthly': {}, 'weekly': {}}
        
        # Monthly performance
        df['month'] = df['exit_time'].dt.strftime('%Y-%m')
        monthly_profit = df.groupby('month')['profit_loss'].sum()
        monthly = {month: float(profit) for month, profit in monthly_profit.items()}
        
        # Weekly performance
        df['week'] = df['exit_time'].dt.strftime('%Y-%U')
        weekly_profit = df.groupby('week')['profit_loss'].sum()
        weekly = {week: float(profit) for week, profit in weekly_profit.items()}
        
        return {
            'monthly': monthly,
            'weekly': weekly
        }
    
    def _calculate_consecutive_metrics(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Calculate consecutive win/loss streaks
        
        Args:
            df: DataFrame of trades
            
        Returns:
            Dictionary of consecutive streak metrics
        """
        if df.empty:
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_consecutive_wins': 0,
                'current_consecutive_losses': 0
            }
        
        # Sort by exit time
        df = df.sort_values('exit_time')
        
        # Mark winning and losing trades
        df['is_win'] = df['profit_loss'] > 0
        
        # Initialize counters
        current_streak = 1
        max_win_streak = 0
        max_loss_streak = 0
        
        # Calculate streaks
        for i in range(1, len(df)):
            if df.iloc[i]['is_win'] == df.iloc[i-1]['is_win']:
                current_streak += 1
            else:
                # Reset streak
                current_streak = 1
            
            if df.iloc[i]['is_win']:
                max_win_streak = max(max_win_streak, current_streak)
            else:
                max_loss_streak = max(max_loss_streak, current_streak)
        
        # Calculate current streaks
        current_wins = 0
        current_losses = 0
        
        # Start from the last trade and go backwards
        for i in range(len(df)-1, -1, -1):
            is_win = df.iloc[i]['is_win']
            
            if current_wins == 0 and current_losses == 0:
                # First iteration
                if is_win:
                    current_wins = 1
                else:
                    current_losses = 1
            elif is_win and current_wins > 0:
                current_wins += 1
            elif not is_win and current_losses > 0:
                current_losses += 1
            else:
                break
        
        return {
            'max_consecutive_wins': max_win_streak,
            'max_consecutive_losses': max_loss_streak,
            'current_consecutive_wins': current_wins,
            'current_consecutive_losses': current_losses
        }
    
    def _calculate_equity_curve(self, df: pd.DataFrame, initial_capital: float) -> List[Dict[str, Any]]:
        """
        Calculate equity curve data
        
        Args:
            df: DataFrame of trades
            initial_capital: Initial capital amount
            
        Returns:
            List of equity points
        """
        if df.empty:
            return []
        
        # Sort by exit time
        df = df.sort_values('exit_time')
        
        # Calculate cumulative P&L
        df['cumulative_pnl'] = df['profit_loss'].cumsum()
        df['equity'] = initial_capital + df['cumulative_pnl']
        
        # Create equity curve
        equity_curve = []
        
        for _, row in df.iterrows():
            equity_curve.append({
                'timestamp': row['exit_time'].isoformat(),
                'equity': float(row['equity']),
                'trade_id': row.get('id', None),
                'symbol': row['symbol'],
                'profit_loss': float(row['profit_loss'])
            })
        
        return equity_curve
    
    def generate_equity_curve_chart(self, trades: List[Dict[str, Any]], initial_capital: float) -> Optional[bytes]:
        """
        Generate equity curve chart image
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            
        Returns:
            PNG image bytes or None if error
        """
        try:
            if not trades:
                self.logger.warning("No trades provided for chart generation")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Ensure required columns exist
            if 'exit_time' not in df.columns or 'profit_loss' not in df.columns:
                self.logger.error("Missing required columns for chart generation")
                return None
            
            # Convert timestamps if needed
            if not pd.api.types.is_datetime64_dtype(df['exit_time']):
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            
            # Sort by exit time
            df = df.sort_values('exit_time')
            
            # Calculate equity curve
            df['cumulative_pnl'] = df['profit_loss'].cumsum()
            df['equity'] = initial_capital + df['cumulative_pnl']
            
            # Calculate drawdown
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = (df['peak'] - df['equity']) / df['peak']
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(df['exit_time'], df['equity'], label='Equity Curve', color='blue')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Equity')
            ax1.grid(True)
            
            # Highlight winning and losing trades
            for _, row in df.iterrows():
                color = 'green' if row['profit_loss'] > 0 else 'red'
                ax1.scatter(row['exit_time'], row['equity'], color=color, s=20)
            
            # Plot drawdown
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.fill_between(df['exit_time'], df['drawdown'], 0, color='red', alpha=0.3)
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown %')
            ax2.set_ylim(0, df['drawdown'].max() * 1.1 if df['drawdown'].max() > 0 else 0.1)
            ax2.grid(True)
            ax2.invert_yaxis()
            
            # Add metrics annotation
            metrics = self.calculate_metrics(trades, initial_capital)
            metrics_text = (
                f"Total Return: {metrics['return']:.2%}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Win Rate: {metrics['win_rate']:.2%}\n"
                f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                f"Total Trades: {metrics['total_trades']}"
            )
            
            plt.figtext(0.15, 0.05, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating equity curve chart: {str(e)}")
            return None


class PerformanceReportGenerator:
    """Generates performance reports in various formats"""
    
    def __init__(self, templates_dir: str = "./performance/templates", output_dir: str = "./reports"):
        """
        Initialize the report generator
        
        Args:
            templates_dir: Directory containing report templates
            output_dir: Directory to save generated reports
        """
        self.logger = logging.getLogger("PerformanceReportGenerator")
        self.logger.setLevel(logging.INFO)
        
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 template environment
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Initialize performance calculator
        self.calculator = PerformanceCalculator()
    
    def generate_html_report(self, 
                            trades: List[Dict[str, Any]], 
                            initial_capital: float,
                            report_title: str = "Trading Performance Report",
                            include_trade_list: bool = True,
                            template_name: str = "html_report.html") -> str:
        """
        Generate HTML performance report
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            report_title: Title of the report
            include_trade_list: Whether to include full trade list
            template_name: Template file name
            
        Returns:
            Path to generated HTML file
        """
        try:
            if not trades:
                self.logger.warning("No trades provided for report generation")
                return ""
            
            # Calculate performance metrics
            metrics = self.calculator.calculate_metrics(trades, initial_capital)
            
            # Generate equity curve chart
            equity_chart = self.calculator.generate_equity_curve_chart(trades, initial_capital)
            if equity_chart:
                equity_chart_b64 = base64.b64encode(equity_chart).decode('utf-8')
            else:
                equity_chart_b64 = ""
            
            # Generate monthly performance chart
            monthly_chart = self._generate_monthly_performance_chart(metrics['monthly_performance'])
            if monthly_chart:
                monthly_chart_b64 = base64.b64encode(monthly_chart).decode('utf-8')
            else:
                monthly_chart_b64 = ""
            
            # Generate strategy performance chart
            strategy_chart = self._generate_strategy_performance_chart(metrics['strategies'])
            if strategy_chart:
                strategy_chart_b64 = base64.b64encode(strategy_chart).decode('utf-8')
            else:
                strategy_chart_b64 = ""
            
            # Generate symbol performance chart
            symbol_chart = self._generate_symbol_performance_chart(metrics['symbols'])
            if symbol_chart:
                symbol_chart_b64 = base64.b64encode(symbol_chart).decode('utf-8')
            else:
                symbol_chart_b64 = ""
            
            # Prepare template data
            template_data = {
                'report_title': report_title,
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': metrics,
                'initial_capital': initial_capital,
                'equity_chart': f"data:image/png;base64,{equity_chart_b64}",
                'monthly_chart': f"data:image/png;base64,{monthly_chart_b64}",
                'strategy_chart': f"data:image/png;base64,{strategy_chart_b64}",
                'symbol_chart': f"data:image/png;base64,{symbol_chart_b64}",
                'include_trade_list': include_trade_list,
                'trades': trades if include_trade_list else []
            }
            
            # Load and render template
            template = self.template_env.get_template(template_name)
            html_content = template.render(**template_data)
            
            # Save to file
            report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"performance_report_{report_id}.html"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Generated HTML report: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return ""
    
    def generate_pdf_report(self, 
                          trades: List[Dict[str, Any]], 
                          initial_capital: float,
                          report_title: str = "Trading Performance Report",
                          include_trade_list: bool = True,
                          template_name: str = "pdf_report.html") -> str:
        """
        Generate PDF performance report
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            report_title: Title of the report
            include_trade_list: Whether to include full trade list
            template_name: Template file name
            
        Returns:
            Path to generated PDF file
        """
        try:
            # First generate HTML
            html_path = self.generate_html_report(
                trades, 
                initial_capital, 
                report_title, 
                include_trade_list, 
                template_name
            )
            
            if not html_path:
                self.logger.error("Failed to generate HTML for PDF conversion")
                return ""
            
            # Convert HTML to PDF
            report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"performance_report_{report_id}.pdf"
            
            # Configure PDF options
            options = {
                'page-size': 'A4',
                'margin-top': '1cm',
                'margin-right': '1cm',
                'margin-bottom': '1cm',
                'margin-left': '1cm',
                'encoding': 'UTF-8',
                'no-outline': None,
                'enable-local-file-access': None
            }
            
            # Convert to PDF
            pdfkit.from_file(html_path, output_file, options=options)
            
            self.logger.info(f"Generated PDF report: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
            return ""
    
    def generate_csv_report(self, 
                           trades: List[Dict[str, Any]], 
                           initial_capital: float) -> str:
        """
        Generate CSV performance report
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            
        Returns:
            Path to generated CSV file
        """
        try:
            if not trades:
                self.logger.warning("No trades provided for CSV report generation")
                return ""
            
            # Calculate performance metrics
            metrics = self.calculator.calculate_metrics(trades, initial_capital)
            
            # Create report ID
            report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save summary metrics
            summary_file = self.output_dir / f"performance_summary_{report_id}.csv"
            
            with open(summary_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Trades', metrics['total_trades']])
                writer.writerow(['Winning Trades', metrics['winning_trades']])
                writer.writerow(['Losing Trades', metrics['losing_trades']])
                writer.writerow(['Win Rate', metrics['win_rate']])
                writer.writerow(['Net Profit', metrics['net_profit']])
                writer.writerow(['Profit Factor', metrics['profit_factor']])
                writer.writerow(['Average Trade', metrics['avg_trade']])
                writer.writerow(['Average Profit', metrics['avg_profit']])
                writer.writerow(['Average Loss', metrics['avg_loss']])
                writer.writerow(['Return', metrics['return']])
                writer.writerow(['Annualized Return', metrics['annualized_return']])
                writer.writerow(['Max Drawdown', metrics['max_drawdown']])
                writer.writerow(['Sharpe Ratio', metrics['sharpe_ratio']])
                writer.writerow(['Max Consecutive Wins', metrics['max_consecutive_wins']])
                writer.writerow(['Max Consecutive Losses', metrics['max_consecutive_losses']])
            
            # Save trade list
            trades_file = self.output_dir / f"trade_list_{report_id}.csv"
            
            # Convert trades to DataFrame for easier CSV export
            df = pd.DataFrame(trades)
            df.to_csv(trades_file, index=False)
            
            self.logger.info(f"Generated CSV report: {summary_file} and {trades_file}")
            return str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {str(e)}")
            return ""
    
    def generate_json_report(self, 
                           trades: List[Dict[str, Any]], 
                           initial_capital: float) -> str:
        """
        Generate JSON performance report
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            
        Returns:
            Path to generated JSON file
        """
        try:
            if not trades:
                self.logger.warning("No trades provided for JSON report generation")
                return ""
            
            # Calculate performance metrics
            metrics = self.calculator.calculate_metrics(trades, initial_capital)
            
            # Create report data
            report_data = {
                'initial_capital': initial_capital,
                'metrics': metrics,
                'trades': trades,
                'generation_time': datetime.now().isoformat()
            }
            
            # Save to file
            report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"performance_report_{report_id}.json"
            
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Generated JSON report: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            return ""
    
    def _generate_monthly_performance_chart(self, monthly_data: Dict[str, float]) -> Optional[bytes]:
        """
        Generate monthly performance chart
        
        Args:
            monthly_data: Dictionary of monthly performance data
            
        Returns:
            PNG image bytes or None if error
        """
        try:
            if not monthly_data:
                return None
            
            # Convert to DataFrame
            months = []
            profits = []
            
            for month, profit in sorted(monthly_data.items()):
                months.append(month)
                profits.append(profit)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create bar colors based on profit/loss
            colors = ['green' if p > 0 else 'red' for p in profits]
            
            # Plot bars
            plt.bar(months, profits, color=colors)
            plt.title('Monthly Performance')
            plt.xlabel('Month')
            plt.ylabel('Profit/Loss')
            plt.xticks(rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add profit values on top of bars
            for i, p in enumerate(profits):
                plt.text(i, p + (0.05 * max(profits) if p >= 0 else -0.05 * min(profits)),
                         f"{p:.2f}", ha='center', va='center', fontsize=8,
                         rotation=90 if len(months) > 6 else 0)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating monthly chart: {str(e)}")
            return None
    
    def _generate_strategy_performance_chart(self, strategy_data: Dict[str, Dict[str, Any]]) -> Optional[bytes]:
        """
        Generate strategy performance chart
        
        Args:
            strategy_data: Dictionary of strategy performance data
            
        Returns:
            PNG image bytes or None if error
        """
        try:
            if not strategy_data:
                return None
            
            # Extract data
            strategies = []
            profits = []
            win_rates = []
            
            for strategy, data in strategy_data.items():
                strategies.append(strategy)
                profits.append(data['net_profit'])
                win_rates.append(data['win_rate'] * 100)  # Convert to percentage
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot profits
            colors1 = ['green' if p > 0 else 'red' for p in profits]
            ax1.bar(strategies, profits, color=colors1)
            ax1.set_title('Strategy Net Profit')
            ax1.set_ylabel('Profit/Loss')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Rotate x labels if many strategies
            if len(strategies) > 4:
                ax1.set_xticklabels(strategies, rotation=45, ha='right')
            
            # Plot win rates
            ax2.bar(strategies, win_rates, color='blue')
            ax2.set_title('Strategy Win Rate')
            ax2.set_ylabel('Win Rate (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Rotate x labels if many strategies
            if len(strategies) > 4:
                ax2.set_xticklabels(strategies, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating strategy chart: {str(e)}")
            return None
    
    def _generate_symbol_performance_chart(self, symbol_data: Dict[str, Dict[str, Any]]) -> Optional[bytes]:
        """
        Generate symbol performance chart
        
        Args:
            symbol_data: Dictionary of symbol performance data
            
        Returns:
            PNG image bytes or None if error
        """
        try:
            if not symbol_data:
                return None
            
            # Extract data
            symbols = []
            profits = []
            trade_counts = []
            
            for symbol, data in symbol_data.items():
                symbols.append(symbol)
                profits.append(data['net_profit'])
                trade_counts.append(data['total_trades'])
            
            # Sort by profit
            sorted_indices = np.argsort(profits)
            symbols = [symbols[i] for i in sorted_indices]
            profits = [profits[i] for i in sorted_indices]
            trade_counts = [trade_counts[i] for i in sorted_indices]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot profits
            colors1 = ['green' if p > 0 else 'red' for p in profits]
            ax1.barh(symbols, profits, color=colors1)
            ax1.set_title('Symbol Net Profit')
            ax1.set_xlabel('Profit/Loss')
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Plot trade counts
            ax2.barh(symbols, trade_counts, color='blue')
            ax2.set_title('Symbol Trade Count')
            ax2.set_xlabel('Number of Trades')
            ax2.grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            buf.seek(0)
            return buf.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating symbol chart: {str(e)}")
            return None


class TaxReportGenerator:
    """Generates tax reports for trading activity"""
    
    def __init__(self, output_dir: str = "./reports/tax"):
        """
        Initialize the tax report generator
        
        Args:
            output_dir: Directory to save generated tax reports
        """
        self.logger = logging.getLogger("TaxReportGenerator")
        self.logger.setLevel(logging.INFO)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_trade_history_report(self, 
                                    trades: List[Dict[str, Any]], 
                                    year: int,
                                    include_unrealized: bool = False,
                                    group_by_symbol: bool = True) -> str:
        """
        Generate trade history report for tax purposes
        
        Args:
            trades: List of trade dictionaries
            year: Tax year
            include_unrealized: Whether to include unrealized gains/losses
            group_by_symbol: Whether to group trades by symbol
            
        Returns:
            Path to generated CSV file
        """
        try:
            if not trades:
                self.logger.warning("No trades provided for tax report generation")
                return ""
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Filter for the specified year
            if 'exit_time' not in df.columns:
                self.logger.error("Missing 'exit_time' column in trade data")
                return ""
            
            # Convert timestamps if needed
            if not pd.api.types.is_datetime64_dtype(df['exit_time']):
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            
            # Filter trades for the specified year
            df['year'] = df['exit_time'].dt.year
            year_df = df[df['year'] == year]
            
            if year_df.empty:
                self.logger.warning(f"No trades found for year {year}")
                return ""
            
            # Create report filename
            output_file = self.output_dir / f"tax_report_{year}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # If grouping by symbol
            if group_by_symbol:
                # Group by symbol and aggregate
                symbol_groups = year_df.groupby('symbol')
                
                with open(output_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'Symbol', 'Total Trades', 'Total Profit/Loss', 
                        'Short-Term Trades', 'Short-Term P/L', 
                        'Long-Term Trades', 'Long-Term P/L'
                    ])
                    
                    # Process each symbol
                    for symbol, group in symbol_groups:
                        # Calculate holding period for each trade
                        if 'entry_time' in group.columns:
                            group['holding_period'] = (group['exit_time'] - group['entry_time']).dt.days
                            short_term = group[group['holding_period'] <= 365]
                            long_term = group[group['holding_period'] > 365]
                        else:
                            # If entry_time not available, assume all are short-term
                            short_term = group
                            long_term = pd.DataFrame()
                        
                        # Calculate totals
                        total_trades = len(group)
                        total_pl = group['profit_loss'].sum()
                        
                        short_term_trades = len(short_term)
                        short_term_pl = short_term['profit_loss'].sum() if not short_term.empty else 0
                        
                        long_term_trades = len(long_term)
                        long_term_pl = long_term['profit_loss'].sum() if not long_term.empty else 0
                        
                        # Write row
                        writer.writerow([
                            symbol, total_trades, total_pl,
                            short_term_trades, short_term_pl,
                            long_term_trades, long_term_pl
                        ])
                    
                    # Write summary row
                    writer.writerow([])
                    writer.writerow([
                        'TOTAL', len(year_df), year_df['profit_loss'].sum(),
                        len(year_df[year_df.get('holding_period', 0) <= 365]), 
                        year_df[year_df.get('holding_period', 0) <= 365]['profit_loss'].sum(),
                        len(year_df[year_df.get('holding_period', 0) > 365]), 
                        year_df[year_df.get('holding_period', 0) > 365]['profit_loss'].sum()
                    ])
            else:
                # Detailed trade report
                # Select and rename columns for tax reporting
                tax_columns = [
                    'symbol', 'entry_time', 'exit_time', 'direction',
                    'entry_price', 'exit_price', 'size', 'profit_loss',
                    'fees', 'commission'
                ]
                
                # Filter columns that exist in the DataFrame
                available_columns = [col for col in tax_columns if col in year_df.columns]
                
                # Add holding period if entry_time is available
                if 'entry_time' in year_df.columns:
                    if not pd.api.types.is_datetime64_dtype(year_df['entry_time']):
                        year_df['entry_time'] = pd.to_datetime(year_df['entry_time'])
                    
                    year_df['holding_period_days'] = (year_df['exit_time'] - year_df['entry_time']).dt.days
                    available_columns.append('holding_period_days')
                
                # Select columns and export
                export_df = year_df[available_columns]
                export_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Generated tax report for year {year}: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error generating tax report: {str(e)}")
            return ""
    
    def generate_profit_loss_summary(self, 
                                   trades: List[Dict[str, Any]], 
                                   year: Optional[int] = None) -> str:
        """
        Generate profit/loss summary for tax purposes
        
        Args:
            trades: List of trade dictionaries
            year: Optional tax year (all years if None)
            
        Returns:
            Path to generated CSV file
        """
        try:
            if not trades:
                self.logger.warning("No trades provided for P/L summary generation")
                return ""
            
            # Convert to DataFrame
            df = pd.DataFrame(trades)
            
            # Ensure required columns exist
            if 'exit_time' not in df.columns or 'profit_loss' not in df.columns:
                self.logger.error("Missing required columns for P/L summary")
                return ""
            
            # Convert timestamps if needed
            if not pd.api.types.is_datetime64_dtype(df['exit_time']):
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            
            # Add year and month columns
            df['year'] = df['exit_time'].dt.year
            df['month'] = df['exit_time'].dt.month
            
            # Filter by year if specified
            if year is not None:
                df = df[df['year'] == year]
                year_str = str(year)
            else:
                year_str = "all"
            
            if df.empty:
                self.logger.warning(f"No trades found for the specified period")
                return ""
            
            # Create output file
            output_file = self.output_dir / f"pl_summary_{year_str}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            # Group by year and month, calculate summary
            summary = df.groupby(['year', 'month']).agg({
                'profit_loss': ['sum', 'count'],
                'symbol': 'nunique'
            })
            
            # Flatten column names
            summary.columns = ['total_profit_loss', 'trade_count', 'symbol_count']
            summary = summary.reset_index()
            
            # Add month names
            summary['month_name'] = summary['month'].apply(lambda x: calendar.month_name[x])
            
            # Sort by year and month
            summary = summary.sort_values(['year', 'month'])
            
            # Export summary
            summary.to_csv(output_file, index=False)
            
            # Calculate year totals
            yearly_summary = df.groupby('year').agg({
                'profit_loss': ['sum', 'count'],
                'symbol': 'nunique'
            })
            
            yearly_summary.columns = ['total_profit_loss', 'trade_count', 'symbol_count']
            yearly_summary = yearly_summary.reset_index()
            
            # Calculate grand total
            grand_total = {
                'year': 'TOTAL',
                'month': '',
                'month_name': '',
                'total_profit_loss': df['profit_loss'].sum(),
                'trade_count': len(df),
                'symbol_count': df['symbol'].nunique()
            }
            
            # Append yearly summary and grand total
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(['Year Totals'])
                
                for _, row in yearly_summary.iterrows():
                    writer.writerow([
                        row['year'], '', '', 
                        row['total_profit_loss'], 
                        row['trade_count'], 
                        row['symbol_count']
                    ])
                
                writer.writerow([])
                writer.writerow([
                    grand_total['year'], grand_total['month'], grand_total['month_name'],
                    grand_total['total_profit_loss'], 
                    grand_total['trade_count'], 
                    grand_total['symbol_count']
                ])
            
            self.logger.info(f"Generated P/L summary report: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Error generating P/L summary: {str(e)}")
            return ""


class ReportingScheduler:
    """Schedules and manages periodic performance reports"""
    
    def __init__(self, report_config: Dict[str, Any] = None):
        """
        Initialize the reporting scheduler
        
        Args:
            report_config: Configuration for scheduled reports
        """
        self.logger = logging.getLogger("ReportingScheduler")
        self.logger.setLevel(logging.INFO)
        
        self.config = report_config or {}
        self.report_generators = {
            'performance': PerformanceReportGenerator(),
            'tax': TaxReportGenerator()
        }
        
        self.report_history = []
        self.email_config = self.config.get('email', {})
        self.schedule = self.config.get('schedule', {})
    
    def schedule_reports(self, schedule_config: Dict[str, Any]) -> None:
        """
        Set up report scheduling configuration
        
        Args:
            schedule_config: Report scheduling configuration
        """
        self.schedule = schedule_config
        self.logger.info("Updated report scheduling configuration")
    
    def configure_email(self, email_config: Dict[str, Any]) -> None:
        """
        Configure email settings for report delivery
        
        Args:
            email_config: Email configuration
        """
        self.email_config = email_config
        self.logger.info("Updated email configuration")
    
    def generate_scheduled_reports(self, 
                                trades: List[Dict[str, Any]], 
                                initial_capital: float,
                                report_time: datetime = None) -> List[str]:
        """
        Generate reports based on current schedule
        
        Args:
            trades: List of trade dictionaries
            initial_capital: Initial capital amount
            report_time: Time to use for scheduling decisions (default: now)
            
        Returns:
            List of paths to generated reports
        """
        if not report_time:
            report_time = datetime.now()
        
        # Initialize list of generated report paths
        generated_reports = []
        
        try:
            # Check which reports should be generated
            reports_to_generate = self._determine_reports_to_generate(report_time)
            
            if not reports_to_generate:
                self.logger.info("No reports scheduled for this time")
                return []
            
            # Generate each scheduled report
            for report in reports_to_generate:
                report_type = report.get('type', 'performance')
                format = report.get('format', 'html')
                title = report.get('title', f"{report_type.capitalize()} Report")
                
                report_path = ""
                
                # Generate appropriate report
                if report_type == 'performance':
                    if format == 'html':
                        report_path = self.report_generators['performance'].generate_html_report(
                            trades, initial_capital, title
                        )
                    elif format == 'pdf':
                        report_path = self.report_generators['performance'].generate_pdf_report(
                            trades, initial_capital, title
                        )
                    elif format == 'csv':
                        report_path = self.report_generators['performance'].generate_csv_report(
                            trades, initial_capital
                        )
                    elif format == 'json':
                        report_path = self.report_generators['performance'].generate_json_report(
                            trades, initial_capital
                        )
                
                elif report_type == 'tax':
                    current_year = report_time.year
                    report_path = self.report_generators['tax'].generate_trade_history_report(
                        trades, current_year
                    )
                
                # Add to list of generated reports
                if report_path:
                    generated_reports.append(report_path)
                    
                    # Send email if configured
                    if report.get('email', False) and self.email_config:
                        self._send_report_email(report_path, title, report_type)
                    
                    # Add to report history
                    self.report_history.append({
                        'type': report_type,
                        'format': format,
                        'path': report_path,
                        'time': datetime.now().isoformat(),
                        'title': title
                    })
            
            return generated_reports
            
        except Exception as e:
            self.logger.error(f"Error generating scheduled reports: {str(e)}")
            return []
    
    def _determine_reports_to_generate(self, current_time: datetime) -> List[Dict[str, Any]]:
        """
        Determine which reports should be generated at the current time
        
        Args:
            current_time: Current time
            
        Returns:
            List of report configurations to generate
        """
        reports_to_generate = []
        
        for report_config in self.schedule.get('reports', []):
            frequency = report_config.get('frequency', 'daily')
            
            if frequency == 'daily':
                # Check if current hour matches scheduled hour
                scheduled_hour = report_config.get('hour', 0)
                if current_time.hour == scheduled_hour and current_time.minute < 15:
                    reports_to_generate.append(report_config)
            
            elif frequency == 'weekly':
                # Check if current day of week matches scheduled day
                scheduled_day = report_config.get('day', 0)  # 0 = Monday, 6 = Sunday
                scheduled_hour = report_config.get('hour', 0)
                
                if current_time.weekday() == scheduled_day and current_time.hour == scheduled_hour and current_time.minute < 15:
                    reports_to_generate.append(report_config)
            
            elif frequency == 'monthly':
                # Check if current day of month matches scheduled day
                scheduled_day = report_config.get('day', 1)
                scheduled_hour = report_config.get('hour', 0)
                
                if current_time.day == scheduled_day and current_time.hour == scheduled_hour and current_time.minute < 15:
                    reports_to_generate.append(report_config)
            
            elif frequency == 'quarterly':
                # Check if current month is the start of a quarter
                quarterly_months = [1, 4, 7, 10]
                scheduled_day = report_config.get('day', 1)
                scheduled_hour = report_config.get('hour', 0)
                
                if current_time.month in quarterly_months and current_time.day == scheduled_day and current_time.hour == scheduled_hour and current_time.minute < 15:
                    reports_to_generate.append(report_config)
            
            elif frequency == 'yearly':
                # Check if current month/day matches scheduled month/day
                scheduled_month = report_config.get('month', 1)
                scheduled_day = report_config.get('day', 1)
                scheduled_hour = report_config.get('hour', 0)
                
                if current_time.month == scheduled_month and current_time.day == scheduled_day and current_time.hour == scheduled_hour and current_time.minute < 15:
                    reports_to_generate.append(report_config)
        
        return reports_to_generate
    
    def _send_report_email(self, report_path: str, title: str, report_type: str) -> bool:
        """
        Send a report via email
        
        Args:
            report_path: Path to the report file
            title: Report title
            report_type: Type of report
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            if not self.email_config:
                self.logger.warning("Email configuration not set")
                return False
            
            if not report_path or not os.path.exists(report_path):
                self.logger.error(f"Report file not found: {report_path}")
                return False
            
            # Create email message
            msg = MIMEMultipart()
            msg['Subject'] = f"{title} - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = self.email_config.get('sender', '')
            
            # Add recipients
            recipients = self.email_config.get('recipients', [])
            if isinstance(recipients, list):
                msg['To'] = ', '.join(recipients)
            else:
                msg['To'] = recipients
            
            # Add email body
            body = f"""
            <html>
            <body>
                <h2>{title}</h2>
                <p>Please find attached the {report_type} report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.</p>
                <p>This is an automated message from your trading bot.</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html'))
            
            # Attach report file
            with open(report_path, 'rb') as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header(
                    'Content-Disposition', 
                    'attachment', 
                    filename=os.path.basename(report_path)
                )
                msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(self.email_config.get('smtp_server', ''), self.email_config.get('smtp_port', 587)) as server:
                server.starttls()
                server.login(
                    self.email_config.get('username', ''),
                    self.email_config.get('password', '')
                )
                server.send_message(msg)
            
            self.logger.info(f"Sent report email: {title}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending report email: {str(e)}")
            return False
    
    def get_report_history(self) -> List[Dict[str, Any]]:
        """
        Get history of generated reports
        
        Returns:
            List of report history entries
        """
        return self.report_history
