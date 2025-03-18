"""
On-Chain Analysis Strategy Implementation
"""
import logging
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class OnChainStrategy:
    """
    Strategy that trades based on on-chain data analysis
    """
    
    def __init__(self, config):
        """
        Initialize the on-chain analysis strategy
        
        Args:
            config (dict): Strategy configuration
        """
        self.logger = logging.getLogger("on_chain_strategy")
        self.config = config
        
        # Configuration parameters with defaults
        self.api_key = config.get("api_key", "")
        self.update_interval = config.get("update_interval", 3600)  # 1 hour
        self.significant_threshold = config.get("significant_threshold", 1000000)  # $1M
        self.exchanges_to_track = config.get("exchanges_to_track", ["binance", "coinbase", "ftx"])
        
        # Cache for on-chain data
        self.on_chain_data = {}
        self.whale_movements = {}
        self.exchange_flows = {}
        self.last_update_time = 0
        
        self.logger.info("On-chain analysis strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if an on-chain based entry signal is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Update on-chain data if needed
        self._update_on_chain_data(symbol)
        
        # Get on-chain data for this symbol
        symbol_data = self._get_symbol_data(symbol)
        
        if not symbol_data:
            return None
        
        # Get OHLCV data for price context
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < 10:
            return None
        
        # Check for significant whale movements
        whale_signal = self._check_whale_movements(symbol, symbol_data.get("whale_movements", []))
        
        # Check for exchange flow imbalances
        flow_signal = self._check_exchange_flows(symbol, symbol_data.get("exchange_flows", {}))
        
        # Check for staking/minting activities
        staking_signal = self._check_staking_activity(symbol, symbol_data.get("staking_data", {}))
        
        # Get current price
        current_price = candles["close"].iloc[-1]
        
        # Combine signals
        entry_signal = None
        
        if whale_signal and whale_signal["direction"] == "buy" and (flow_signal is None or flow_signal["direction"] == "buy"):
            # Bullish signal: whales buying + neutral/bullish exchange flows
            atr = self._calculate_atr(candles)
            stop_loss = current_price - (atr * 1.5)
            target_price = current_price + (atr * 3.0)
            
            self.logger.info(f"Bullish on-chain signal for {symbol} at {current_price}: whale accumulation")
            
            entry_signal = {
                "direction": "long",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "signal_type": "whale_accumulation",
                "whale_data": whale_signal,
                "flow_data": flow_signal
            }
            
        elif whale_signal and whale_signal["direction"] == "sell" and (flow_signal is None or flow_signal["direction"] == "sell"):
            # Bearish signal: whales selling + neutral/bearish exchange flows
            atr = self._calculate_atr(candles)
            stop_loss = current_price + (atr * 1.5)
            target_price = current_price - (atr * 3.0)
            
            self.logger.info(f"Bearish on-chain signal for {symbol} at {current_price}: whale distribution")
            
            entry_signal = {
                "direction": "short",
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "signal_type": "whale_distribution",
                "whale_data": whale_signal,
                "flow_data": flow_signal
            }
            
        # Staking can override other signals if very strong
        if staking_signal and staking_signal["strength"] > 0.8:
            atr = self._calculate_atr(candles)
            
            if staking_signal["direction"] == "buy":
                stop_loss = current_price - (atr * 1.5)
                target_price = current_price + (atr * 3.0)
                
                self.logger.info(f"Bullish on-chain signal for {symbol} at {current_price}: staking activity")
                
                entry_signal = {
                    "direction": "long",
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "target_price": target_price,
                    "signal_type": "staking_activity",
                    "staking_data": staking_signal
                }
                
            elif staking_signal["direction"] == "sell":
                stop_loss = current_price + (atr * 1.5)
                target_price = current_price - (atr * 3.0)
                
                self.logger.info(f"Bearish on-chain signal for {symbol} at {current_price}: unstaking activity")
                
                entry_signal = {
                    "direction": "short",
                    "entry_price": current_price,
                    "stop_loss": stop_loss,
                    "target_price": target_price,
                    "signal_type": "unstaking_activity",
                    "staking_data": staking_signal
                }
        
        return entry_signal
    
    def should_exit(self, symbol, direction, position, market_data):
        """
        Check if we should exit an existing position
        
        Args:
            symbol (str): Symbol to check
            direction (str): Position direction
            position (dict): Position details
            market_data (dict): Market data
            
        Returns:
            bool: True if we should exit, False otherwise
        """
        # Update on-chain data if needed
        self._update_on_chain_data(symbol)
        
        # Get on-chain data for this symbol
        symbol_data = self._get_symbol_data(symbol)
        
        if not symbol_data:
            return False
        
        # Get OHLCV data for price context
        candles = self._get_candles(symbol, market_data)
        
        if candles is None or len(candles) < 10:
            return False
        
        # Get current price
        current_price = candles["close"].iloc[-1]
        
        # Get position details
        entry_price = position.get("entry_price", current_price)
        signal_type = position.get("signal_type", "")
        
        # Check for reversal in on-chain metrics
        exit_signal = False
        
        # Whale movement reversal
        if "whale_accumulation" in signal_type and direction == "long":
            # Check if whales are now selling
            whale_signal = self._check_whale_movements(symbol, symbol_data.get("whale_movements", []))
            
            if whale_signal and whale_signal["direction"] == "sell" and whale_signal["strength"] > 0.5:
                self.logger.info(f"Whale selling detected for {symbol} long at {current_price}")
                exit_signal = True
                
        elif "whale_distribution" in signal_type and direction == "short":
            # Check if whales are now buying
            whale_signal = self._check_whale_movements(symbol, symbol_data.get("whale_movements", []))
            
            if whale_signal and whale_signal["direction"] == "buy" and whale_signal["strength"] > 0.5:
                self.logger.info(f"Whale buying detected for {symbol} short at {current_price}")
                exit_signal = True
                
        # Exchange flow reversal
        if "flow" in signal_type:
            flow_signal = self._check_exchange_flows(symbol, symbol_data.get("exchange_flows", {}))
            
            if flow_signal:
                if direction == "long" and flow_signal["direction"] == "sell" and flow_signal["strength"] > 0.7:
                    self.logger.info(f"Exchange outflow detected for {symbol} long at {current_price}")
                    exit_signal = True
                    
                elif direction == "short" and flow_signal["direction"] == "buy" and flow_signal["strength"] > 0.7:
                    self.logger.info(f"Exchange inflow detected for {symbol} short at {current_price}")
                    exit_signal = True
                    
        # Staking activity reversal
        if "staking" in signal_type:
            staking_signal = self._check_staking_activity(symbol, symbol_data.get("staking_data", {}))
            
            if staking_signal:
                if direction == "long" and staking_signal["direction"] == "sell" and staking_signal["strength"] > 0.7:
                    self.logger.info(f"Unstaking activity detected for {symbol} long at {current_price}")
                    exit_signal = True
                    
                elif direction == "short" and staking_signal["direction"] == "buy" and staking_signal["strength"] > 0.7:
                    self.logger.info(f"Staking activity detected for {symbol} short at {current_price}")
                    exit_signal = True
        
        # Check price targets
        target_hit = False
        target_price = position.get("target_price")
        
        if target_price:
            if direction == "long" and current_price >= target_price:
                self.logger.info(f"Target reached for {symbol} long at {current_price}, target: {target_price:.5f}")
                target_hit = True
                
            elif direction == "short" and current_price <= target_price:
                self.logger.info(f"Target reached for {symbol} short at {current_price}, target: {target_price:.5f}")
                target_hit = True
        
        return exit_signal or target_hit
    
    def _update_on_chain_data(self, symbol):
        """
        Update on-chain data if it's outdated
        
        Args:
            symbol (str): Symbol to update data for
        """
        current_time = time.time()
        
        # Check if update is needed
        if current_time - self.last_update_time < self.update_interval:
            return
            
        # In a real implementation, this would call blockchain APIs
        # This is a placeholder with simulated data
        try:
            # Extract base asset from symbol (e.g., BTC from BTCUSDT)
            base_asset = symbol.split('/')[0] if '/' in symbol else symbol.split('-')[0] if '-' in symbol else symbol[:3]
            
            if base_asset in ["BTC", "ETH", "USDT", "USDC"]:
                # Simulate whale movements
                self._simulate_whale_movements(base_asset)
                
                # Simulate exchange flows
                self._simulate_exchange_flows(base_asset)
                
                # Simulate staking activity
                self._simulate_staking_activity(base_asset)
                
                self.logger.debug(f"Updated on-chain data for {base_asset}")
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating on-chain data: {str(e)}")
    
    def _simulate_whale_movements(self, asset):
        """
        Simulate whale wallet movements
        
        Args:
            asset (str): Asset symbol
        """
        if asset not in self.whale_movements:
            self.whale_movements[asset] = []
            
        # Generate some random whale movements
        current_time = time.time()
        
        # Clear old data (> 24 hours)
        self.whale_movements[asset] = [m for m in self.whale_movements[asset] if current_time - m["timestamp"] < 86400]
        
        # Add new movements
        num_movements = np.random.randint(0, 3)  # 0-2 new movements
        
        for _ in range(num_movements):
            # Generate random amount (1-10M)
            amount = np.random.uniform(1, 10) * 1000000
            
            # Generate random direction (0=in, 1=out)
            direction = "inflow" if np.random.random() < 0.5 else "outflow"
            
            # Generate random exchange
            exchange = np.random.choice(self.exchanges_to_track)
            
            # Add to movements
            self.whale_movements[asset].append({
                "timestamp": current_time - np.random.uniform(0, 3600),  # 0-1 hour ago
                "amount": amount,
                "direction": direction,
                "exchange": exchange,
                "txid": f"0x{np.random.randint(0, 2**64):016x}"
            })
    
    def _simulate_exchange_flows(self, asset):
        """
        Simulate exchange inflows/outflows
        
        Args:
            asset (str): Asset symbol
        """
        if asset not in self.exchange_flows:
            self.exchange_flows[asset] = {}
            
        # Generate exchange flows
        for exchange in self.exchanges_to_track:
            if exchange not in self.exchange_flows[asset]:
                self.exchange_flows[asset][exchange] = {
                    "inflow_24h": 0,
                    "outflow_24h": 0,
                    "last_update": 0
                }
                
            # Update with random values
            self.exchange_flows[asset][exchange] = {
                "inflow_24h": np.random.uniform(10, 50) * 1000000,  # 10-50M
                "outflow_24h": np.random.uniform(10, 50) * 1000000,  # 10-50M
                "last_update": time.time()
            }
    
    def _simulate_staking_activity(self, asset):
        """
        Simulate staking/unstaking activity
        
        Args:
            asset (str): Asset symbol
        """
        # Only certain assets can be staked
        if asset not in ["ETH", "SOL", "DOT", "ADA", "MATIC"]:
            return
            
        if "staking_data" not in self.on_chain_data:
            self.on_chain_data["staking_data"] = {}
            
        if asset not in self.on_chain_data["staking_data"]:
            self.on_chain_data["staking_data"][asset] = {}
            
        # Update with random values
        self.on_chain_data["staking_data"][asset] = {
            "total_staked": np.random.uniform(1, 10) * 1000000000,  # 1-10B
            "staking_rate": np.random.uniform(0.3, 0.8),  # 30-80%
            "staking_apr": np.random.uniform(0.03, 0.2),  # 3-20%
            "staking_24h": np.random.uniform(-0.05, 0.05),  # -5% to +5% change
            "unstaking_24h": np.random.uniform(-0.05, 0.05),  # -5% to +5% change
            "last_update": time.time()
        }
    
    def _get_symbol_data(self, symbol):
        """
        Get on-chain data for a symbol
        
        Args:
            symbol (str): Symbol to get data for
            
        Returns:
            dict: On-chain data
        """
        # Extract base asset from symbol
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol.split('-')[0] if '-' in symbol else symbol[:3]
        
        # Get data for this asset
        data = {
            "base_asset": base_asset,
            "whale_movements": self.whale_movements.get(base_asset, []),
            "exchange_flows": self.exchange_flows.get(base_asset, {}),
            "staking_data": self.on_chain_data.get("staking_data", {}).get(base_asset, {})
        }
        
        return data
    
    def _get_candles(self, symbol, market_data):
        """Get candle data for the symbol"""
        if symbol not in market_data:
            return None
            
        # Select any available timeframe
        for tf in market_data[symbol]:
            return market_data[symbol][tf]
            
        return None
    
    def _check_whale_movements(self, symbol, movements):
        """
        Analyze whale movements for trading signals
        
        Args:
            symbol (str): Symbol to check
            movements (list): List of whale movements
            
        Returns:
            dict or None: Signal details or None if no signal
        """
        if not movements:
            return None
            
        # Calculate net flow
        inflow = sum(m["amount"] for m in movements if m["direction"] == "inflow")
        outflow = sum(m["amount"] for m in movements if m["direction"] == "outflow")
        
        # Check if there's significant activity
        if inflow + outflow < self.significant_threshold:
            return None
            
        # Calculate net flow
        net_flow = inflow - outflow
        
        # Determine signal direction
        if net_flow > 0:
            # Net inflow - bullish signal
            direction = "buy"
            strength = min(1.0, net_flow / (2 * self.significant_threshold))
        else:
            # Net outflow - bearish signal
            direction = "sell"
            strength = min(1.0, -net_flow / (2 * self.significant_threshold))
            
        return {
            "direction": direction,
            "strength": strength,
            "net_flow": net_flow,
            "inflow": inflow,
            "outflow": outflow,
            "movement_count": len(movements)
        }
    
    def _check_exchange_flows(self, symbol, flows):
        """
        Analyze exchange flows for trading signals
        
        Args:
            symbol (str): Symbol to check
            flows (dict): Exchange flow data
            
        Returns:
            dict or None: Signal details or None if no signal
        """
        if not flows:
            return None
            
        # Calculate total inflows and outflows
        total_inflow = sum(e["inflow_24h"] for e in flows.values())
        total_outflow = sum(e["outflow_24h"] for e in flows.values())
        
        # Check if there's significant activity
        if total_inflow + total_outflow < self.significant_threshold:
            return None
            
        # Calculate net flow
        net_flow = total_inflow - total_outflow
        
        # Calculate flow ratio
        if total_outflow > 0:
            flow_ratio = total_inflow / total_outflow
        else:
            flow_ratio = float('inf')
            
        # Determine signal direction
        if flow_ratio > 1.5:
            # Much more inflow than outflow - bearish (selling pressure incoming)
            direction = "sell"
            strength = min(1.0, (flow_ratio - 1.5) / 3.5)
        elif flow_ratio < 0.67:
            # Much more outflow than inflow - bullish (buying pressure elsewhere)
            direction = "buy"
            strength = min(1.0, (0.67 - flow_ratio) / 0.67)
        else:
            # Balanced flows - no clear signal
            return None
            
        return {
            "direction": direction,
            "strength": strength,
            "net_flow": net_flow,
            "flow_ratio": flow_ratio,
            "total_inflow": total_inflow,
            "total_outflow": total_outflow
        }
    
    def _check_staking_activity(self, symbol, staking_data):
        """
        Analyze staking activity for trading signals
        
        Args:
            symbol (str): Symbol to check
            staking_data (dict): Staking data
            
        Returns:
            dict or None: Signal details or None if no signal
        """
        if not staking_data:
            return None
            
        # Extract staking metrics
        staking_rate = staking_data.get("staking_rate", 0)
        staking_24h = staking_data.get("staking_24h", 0)
        unstaking_24h = staking_data.get("unstaking_24h", 0)
        
        # Calculate net staking
        net_staking = staking_24h - unstaking_24h
        
        # Check if there's significant activity
        if abs(net_staking) < 0.01:
            return None
            
        # Determine signal direction
        if net_staking > 0:
            # Net staking - bullish signal (holders locking up tokens)
            direction = "buy"
            strength = min(1.0, net_staking / 0.05)
        else:
            # Net unstaking - bearish signal (holders freeing up tokens)
            direction = "sell"
            strength = min(1.0, -net_staking / 0.05)
            
        return {
            "direction": direction,
            "strength": strength,
            "net_staking": net_staking,
            "staking_rate": staking_rate,
            "staking_24h": staking_24h,
            "unstaking_24h": unstaking_24h
        }
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        if not all(col in df.columns for col in ["high", "low", "close"]) or len(df) < period:
            return df["close"].iloc[-1] * 0.01  # Default to 1% of current price
            
        try:
            high = df["high"].values
            low = df["low"].values
            close = df["close"].values
            
            # Calculate true range
            tr1 = abs(high[1:] - low[1:])
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])
            
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            
            # Calculate ATR
            atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
            
            return atr
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return df["close"].iloc[-1] * 0.01  # Default to 1% of current price
