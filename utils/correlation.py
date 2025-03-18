"""
Correlation Matrix Analysis Module
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class CorrelationMatrix:
    """
    Analyzes correlations between crypto assets
    """
    
    def __init__(self, config):
        """
        Initialize the correlation matrix analyzer
        
        Args:
            config (dict): Correlation analysis configuration
        """
        self.logger = logging.getLogger("correlation_matrix")
        self.config = config
        
        # Analysis parameters
        self.lookback_periods = config.get("lookback_periods", 50)
        self.min_correlation = config.get("min_correlation", 0.7)
        self.reference_assets = config.get("reference_assets", ["BTC", "ETH"])
        
        # Data storage
        self.correlation_matrix = pd.DataFrame()
        self.price_data = {}
        self.grouped_assets = {}
        
        self.logger.info("Correlation matrix analyzer initialized")
    
    def update(self, market_data):
        """
        Update correlation matrix based on market data
        
        Args:
            market_data (dict): Market data indexed by symbol and timeframe
        """
        # Extract price data
        self._extract_price_data(market_data)
        
        # Calculate correlation matrix
        self._calculate_correlation_matrix()
        
        # Group correlated assets
        self._group_correlated_assets()
    
    def _extract_price_data(self, market_data):
        """
        Extract price data from market data
        
        Args:
            market_data (dict): Market data indexed by symbol and timeframe
        """
        self.price_data = {}
        
        # Extract closing prices from preferred timeframe
        preferred_tf = self.config.get("preferred_timeframe", "1h")
        
        for symbol in market_data:
            if preferred_tf in market_data[symbol]:
                df = market_data[symbol][preferred_tf]
                
                if df is not None and len(df) >= self.lookback_periods:
                    # Extract most recent prices
                    self.price_data[symbol] = df["close"].iloc[-self.lookback_periods:].values
            else:
                # If preferred timeframe not available, try any available timeframe
                for tf in market_data[symbol]:
                    df = market_data[symbol][tf]
                    
                    if df is not None and len(df) >= self.lookback_periods:
                        self.price_data[symbol] = df["close"].iloc[-self.lookback_periods:].values
                        break
    
    def _calculate_correlation_matrix(self):
        """Calculate the correlation matrix between assets"""
        if not self.price_data:
            self.logger.warning("No price data available for correlation analysis")
            return
            
        # Create DataFrame for correlation calculation
        df = pd.DataFrame(self.price_data)
        
        # Calculate percentage changes
        pct_changes = df.pct_change().dropna()
        
        if len(pct_changes) < 2:
            self.logger.warning("Not enough data for correlation calculation")
            return
            
        # Calculate correlation matrix
        self.correlation_matrix = pct_changes.corr()
        
        self.logger.debug("Correlation matrix calculated with %d assets", len(self.correlation_matrix))
    
    def _group_correlated_assets(self):
        """Group assets based on correlation strength"""
        if self.correlation_matrix.empty:
            return
            
        # Reset groups
        self.grouped_assets = {}
        
        # Use reference assets as group anchors
        for ref_asset in self.reference_assets:
            # Skip if reference asset not in data
            if ref_asset not in self.correlation_matrix.columns:
                continue
                
            # Create group for this reference asset
            self.grouped_assets[ref_asset] = []
            
            # Find highly correlated assets
            for asset in self.correlation_matrix.columns:
                if asset == ref_asset:
                    continue
                    
                correlation = self.correlation_matrix.loc[ref_asset, asset]
                
                # Add to group if correlation exceeds threshold
                if abs(correlation) >= self.min_correlation:
                    self.grouped_assets[ref_asset].append({
                        "symbol": asset,
                        "correlation": correlation,
                        "direction": "positive" if correlation > 0 else "negative"
                    })
        
        # Find assets not in any group and create additional groups
        ungrouped = set(self.correlation_matrix.columns) - set(self.reference_assets)
        
        for asset in list(ungrouped):
            # Skip if already processed
            if asset not in ungrouped:
                continue
                
            # Create new group
            self.grouped_assets[asset] = []
            ungrouped.remove(asset)
            
            # Find correlated assets
            for other in ungrouped.copy():
                correlation = self.correlation_matrix.loc[asset, other]
                
                if abs(correlation) >= self.min_correlation:
                    self.grouped_assets[asset].append({
                        "symbol": other,
                        "correlation": correlation,
                        "direction": "positive" if correlation > 0 else "negative"
                    })
                    ungrouped.remove(other)
    
    def get_correlation(self, symbol1, symbol2):
        """
        Get correlation between two symbols
        
        Args:
            symbol1 (str): First symbol
            symbol2 (str): Second symbol
            
        Returns:
            float: Correlation coefficient (-1 to 1)
        """
        if symbol1 not in self.correlation_matrix.columns or symbol2 not in self.correlation_matrix.columns:
            # If not in matrix, calculate on-the-fly if data available
            if symbol1 in self.price_data and symbol2 in self.price_data:
                returns1 = np.diff(self.price_data[symbol1]) / self.price_data[symbol1][:-1]
                returns2 = np.diff(self.price_data[symbol2]) / self.price_data[symbol2][:-1]
                
                if len(returns1) > 1 and len(returns2) > 1:
                    correlation, _ = pearsonr(returns1, returns2)
                    return correlation
            
            return 0
            
        return self.correlation_matrix.loc[symbol1, symbol2]
    
    def get_correlations(self, symbol):
        """
        Get all correlations for a symbol
        
        Args:
            symbol (str): Symbol to get correlations for
            
        Returns:
            dict: Dictionary of correlations with other symbols
        """
        correlations = {}
        
        if symbol in self.correlation_matrix.columns:
            for other in self.correlation_matrix.columns:
                if other != symbol:
                    correlations[other] = self.correlation_matrix.loc[symbol, other]
        
        return correlations
    
    def get_correlated_symbols(self, symbol, min_correlation=None):
        """
        Get symbols correlated with the given symbol
        
        Args:
            symbol (str): Symbol to find correlations for
            min_correlation (float, optional): Minimum correlation threshold
                (defaults to self.min_correlation)
            
        Returns:
            list: List of correlated symbols with correlation values
        """
        if min_correlation is None:
            min_correlation = self.min_correlation
            
        correlations = self.get_correlations(symbol)
        
        # Filter by threshold and sort by correlation strength
        correlated = []
        for other, corr in correlations.items():
            if abs(corr) >= min_correlation:
                correlated.append({
                    "symbol": other,
                    "correlation": corr,
                    "direction": "positive" if corr > 0 else "negative"
                })
        
        # Sort by absolute correlation (strongest first)
        correlated.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return correlated
    
    def get_correlation_groups(self):
        """
        Get groups of correlated assets
        
        Returns:
            dict: Dictionary of correlation groups
        """
        return self.grouped_assets
    
    def get_risk_exposure(self, positions):
        """
        Calculate correlation-based risk exposure
        
        Args:
            positions (list): List of open positions
            
        Returns:
            dict: Risk exposure by correlation group
        """
        exposure = {}
        
        # Initialize exposure for each group
        for group in self.grouped_assets:
            exposure[group] = {
                "direct": 0,  # Direct exposure in the group asset
                "correlated": 0,  # Exposure in correlated assets
                "total": 0,  # Total exposure
                "positions": []  # List of positions contributing to exposure
            }
            
        # Calculate exposure for each position
        for position in positions:
            symbol = position.get("symbol")
            size = position.get("position_size", 0)
            direction = position.get("direction", "long")
            
            # Skip if symbol not in correlation matrix
            if symbol not in self.correlation_matrix.columns:
                continue
                
            # Find which group this position belongs to
            for group, members in self.grouped_assets.items():
                # Check if this is the group asset itself
                if symbol == group:
                    exposure[group]["direct"] += size if direction == "long" else -size
                    exposure[group]["positions"].append(position)
                    continue
                    
                # Check if in group members
                for member in members:
                    if symbol == member["symbol"]:
                        # Apply correlation direction
                        corr_direction = member["direction"]
                        
                        # For negative correlation, long position reduces exposure
                        if corr_direction == "negative":
                            correlated_size = -size if direction == "long" else size
                        else:
                            correlated_size = size if direction == "long" else -size
                            
                        exposure[group]["correlated"] += correlated_size
                        exposure[group]["positions"].append(position)
                        break
            
        # Calculate total exposure
        for group in exposure:
            exposure[group]["total"] = exposure[group]["direct"] + exposure[group]["correlated"]
            
        return exposure
