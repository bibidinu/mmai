"""
Arbitrage Strategy Implementation
"""
import logging
import time
import asyncio
import numpy as np

class ArbitrageStrategy:
    """
    Strategy that identifies and exploits cross-exchange price differences
    """
    
    def __init__(self, config):
        """
        Initialize the arbitrage strategy
        
        Args:
            config (dict): Strategy configuration
        """
        self.logger = logging.getLogger("arbitrage_strategy")
        self.config = config
        
        # Configuration parameters with defaults
        self.min_price_difference = config.get("min_price_difference", 0.005)  # 0.5% minimum difference
        self.max_execution_time = config.get("max_execution_time", 5)  # Maximum seconds for execution
        self.min_profit = config.get("min_profit", 0.001)  # 0.1% minimum profit after fees
        self.exchanges = config.get("exchanges", ["bybit", "binance"])
        
        # Trading fees by exchange (maker fees)
        self.exchange_fees = {
            "bybit": 0.0001,  # 0.01%
            "binance": 0.0001,  # 0.01%
            "coinbase": 0.0005,  # 0.05%
            "ftx": 0.0002,  # 0.02%
            "kucoin": 0.0001  # 0.01%
        }
        
        # Cache for arbitrage opportunities
        self.opportunities = {}
        
        self.logger.info("Arbitrage strategy initialized")
    
    def should_enter(self, symbol, market_data):
        """
        Check if an arbitrage opportunity is present
        
        Args:
            symbol (str): Symbol to check
            market_data (dict): Market data
            
        Returns:
            dict or None: Entry signal parameters or None if no signal
        """
        # Get ticker data from all available exchanges
        ticker_data = self._get_ticker_data(symbol, market_data)
        
        if not ticker_data or len(ticker_data) < 2:
            return None
        
        # Find arbitrage opportunities
        opportunities = self._find_arbitrage_opportunities(symbol, ticker_data)
        
        if not opportunities:
            return None
        
        # Get the best opportunity
        best_opportunity = max(opportunities, key=lambda x: x["profit_pct"])
        
        # Check if profit is above threshold
        if best_opportunity["profit_pct"] < self.min_profit:
            return None
        
        # Cache the opportunity
        self.opportunities[symbol] = best_opportunity
        
        self.logger.info(f"Arbitrage opportunity found for {symbol}: buy at {best_opportunity['buy_exchange']} for {best_opportunity['buy_price']:.8f}, sell at {best_opportunity['sell_exchange']} for {best_opportunity['sell_price']:.8f}, profit: {best_opportunity['profit_pct']:.4%}")
        
        return {
            "direction": "arbitrage",
            "entry_price": best_opportunity["buy_price"],
            "exit_price": best_opportunity["sell_price"],
            "buy_exchange": best_opportunity["buy_exchange"],
            "sell_exchange": best_opportunity["sell_exchange"],
            "profit_pct": best_opportunity["profit_pct"],
            "execution_time": self.max_execution_time
        }
    
    def should_exit(self, symbol, direction, position, market_data):
        """
        Arbitrage positions are typically executed immediately
        This is only used if the arbitrage couldn't be executed right away
        
        Args:
            symbol (str): Symbol to check
            direction (str): Position direction
            position (dict): Position details
            market_data (dict): Market data
            
        Returns:
            bool: True if we should exit, False otherwise
        """
        # Arbitrage positions should be closed immediately after execution
        # This is more of a safety check
        
        # Get entry time from position
        entry_time = position.get("entry_time", 0)
        buy_exchange = position.get("buy_exchange")
        sell_exchange = position.get("sell_exchange")
        
        # Exit if position has been open for too long
        if time.time() - entry_time > self.max_execution_time:
            self.logger.warning(f"Arbitrage position for {symbol} open too long, closing")
            return True
        
        # Get latest prices to check if opportunity still exists
        ticker_data = self._get_ticker_data(symbol, market_data)
        
        if not ticker_data or buy_exchange not in ticker_data or sell_exchange not in ticker_data:
            return True
        
        # Check if prices have moved unfavorably
        buy_price = ticker_data[buy_exchange].get("price", 0)
        sell_price = ticker_data[sell_exchange].get("price", 0)
        
        # Calculate new profit
        buy_fee = buy_price * self.exchange_fees.get(buy_exchange, 0.001)
        sell_fee = sell_price * self.exchange_fees.get(sell_exchange, 0.001)
        
        # Calculate profit after fees
        profit = sell_price - buy_price - buy_fee - sell_fee
        profit_pct = profit / buy_price if buy_price > 0 else 0
        
        # Exit if profit is below threshold
        if profit_pct < self.min_profit:
            self.logger.warning(f"Arbitrage opportunity for {symbol} no longer profitable, closing")
            return True
        
        return False
    
    def _get_ticker_data(self, symbol, market_data):
        """
        Extract ticker data for all exchanges from market data
        
        Args:
            symbol (str): Symbol to get data for
            market_data (dict): Market data
            
        Returns:
            dict: Ticker data by exchange
        """
        ticker_data = {}
        
        for exchange_name, exchange_data in market_data.items():
            # Skip if not a dict or not an exchange we're interested in
            if not isinstance(exchange_data, dict) or exchange_name not in self.exchanges:
                continue
            
            # Check if we have ticker data for this symbol
            if "ticker_data" in exchange_data and symbol in exchange_data["ticker_data"]:
                ticker = exchange_data["ticker_data"][symbol]
                
                # Extract price and timestamp
                price = ticker.get("last_price", 0)
                timestamp = ticker.get("timestamp", 0)
                
                if price > 0:
                    ticker_data[exchange_name] = {
                        "price": price,
                        "timestamp": timestamp
                    }
        
        return ticker_data
    
    def _find_arbitrage_opportunities(self, symbol, ticker_data):
        """
        Find arbitrage opportunities across exchanges
        
        Args:
            symbol (str): Symbol to check
            ticker_data (dict): Ticker data by exchange
            
        Returns:
            list: List of arbitrage opportunities
        """
        opportunities = []
        
        # Compare prices across all exchange pairs
        exchanges = list(ticker_data.keys())
        
        for i in range(len(exchanges)):
            for j in range(i + 1, len(exchanges)):
                exchange1 = exchanges[i]
                exchange2 = exchanges[j]
                
                price1 = ticker_data[exchange1]["price"]
                price2 = ticker_data[exchange2]["price"]
                
                # Calculate price difference percentage
                diff_pct = abs(price1 - price2) / min(price1, price2)
                
                # Skip if difference is too small
                if diff_pct < self.min_price_difference:
                    continue
                
                # Determine buy and sell exchanges
                if price1 < price2:
                    buy_exchange = exchange1
                    buy_price = price1
                    sell_exchange = exchange2
                    sell_price = price2
                else:
                    buy_exchange = exchange2
                    buy_price = price2
                    sell_exchange = exchange1
                    sell_price = price1
                
                # Calculate fees
                buy_fee = buy_price * self.exchange_fees.get(buy_exchange, 0.001)
                sell_fee = sell_price * self.exchange_fees.get(sell_exchange, 0.001)
                
                # Calculate profit after fees
                profit = sell_price - buy_price - buy_fee - sell_fee
                profit_pct = profit / buy_price if buy_price > 0 else 0
                
                # Skip if profit is negative
                if profit_pct <= 0:
                    continue
                
                # Add to opportunities
                opportunities.append({
                    "symbol": symbol,
                    "buy_exchange": buy_exchange,
                    "sell_exchange": sell_exchange,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "price_diff": diff_pct,
                    "buy_fee": buy_fee,
                    "sell_fee": sell_fee,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "timestamp": time.time()
                })
        
        # Sort by profit percentage (descending)
        opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)
        
        return opportunities
    
    async def execute_arbitrage(self, symbol, opportunity, exchanges):
        """
        Execute an arbitrage trade
        
        Args:
            symbol (str): Symbol to trade
            opportunity (dict): Arbitrage opportunity details
            exchanges (dict): Dictionary of exchange objects
            
        Returns:
            dict: Execution results
        """
        # This would be implemented to execute the trade on both exchanges
        # It's a placeholder for now, as actual implementation would depend on the exchange APIs
        
        buy_exchange_name = opportunity["buy_exchange"]
        sell_exchange_name = opportunity["sell_exchange"]
        
        # Get exchange objects
        buy_exchange = exchanges.get(buy_exchange_name)
        sell_exchange = exchanges.get(sell_exchange_name)
        
        if not buy_exchange or not sell_exchange:
            self.logger.error(f"Missing exchange objects for {buy_exchange_name} or {sell_exchange_name}")
            return {"success": False, "error": "Missing exchange objects"}
        
        try:
            # Execute trades concurrently
            buy_task = asyncio.create_task(self._execute_buy(buy_exchange, symbol, opportunity["buy_price"]))
            sell_task = asyncio.create_task(self._execute_sell(sell_exchange, symbol, opportunity["sell_price"]))
            
            # Wait for both trades to complete
            buy_result, sell_result = await asyncio.gather(buy_task, sell_task)
            
            if buy_result["success"] and sell_result["success"]:
                # Calculate actual profit
                actual_profit = sell_result["amount"] - buy_result["amount"]
                actual_profit_pct = actual_profit / buy_result["amount"]
                
                self.logger.info(f"Arbitrage executed successfully for {symbol}: profit: {actual_profit:.8f} ({actual_profit_pct:.4%})")
                
                return {
                    "success": True,
                    "buy_result": buy_result,
                    "sell_result": sell_result,
                    "profit": actual_profit,
                    "profit_pct": actual_profit_pct
                }
            else:
                # If one trade failed, we need to revert the other
                if buy_result["success"]:
                    await self._revert_buy(buy_exchange, symbol, buy_result["order_id"])
                
                if sell_result["success"]:
                    await self._revert_sell(sell_exchange, symbol, sell_result["order_id"])
                
                self.logger.error(f"Arbitrage execution failed for {symbol}: buy success: {buy_result['success']}, sell success: {sell_result['success']}")
                
                return {
                    "success": False,
                    "buy_result": buy_result,
                    "sell_result": sell_result,
                    "error": "Execution failed"
                }
                
        except Exception as e:
            self.logger.error(f"Error executing arbitrage for {symbol}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _execute_buy(self, exchange, symbol, price):
        """
        Execute a buy order
        
        Args:
            exchange: Exchange object
            symbol (str): Symbol to buy
            price (float): Price to buy at
            
        Returns:
            dict: Execution result
        """
        # This would be implemented to execute the buy order
        # It's a placeholder for now
        return {"success": True, "amount": price, "order_id": "buy-123"}
    
    async def _execute_sell(self, exchange, symbol, price):
        """
        Execute a sell order
        
        Args:
            exchange: Exchange object
            symbol (str): Symbol to sell
            price (float): Price to sell at
            
        Returns:
            dict: Execution result
        """
        # This would be implemented to execute the sell order
        # It's a placeholder for now
        return {"success": True, "amount": price, "order_id": "sell-123"}
    
    async def _revert_buy(self, exchange, symbol, order_id):
        """
        Revert a buy order (sell the bought amount)
        
        Args:
            exchange: Exchange object
            symbol (str): Symbol to sell
            order_id (str): Order ID to revert
            
        Returns:
            dict: Revert result
        """
        # This would be implemented to revert the buy order
        # It's a placeholder for now
        return {"success": True}
    
    async def _revert_sell(self, exchange, symbol, order_id):
        """
        Revert a sell order (buy back the sold amount)
        
        Args:
            exchange: Exchange object
            symbol (str): Symbol to buy
            order_id (str): Order ID to revert
            
        Returns:
            dict: Revert result
        """
        # This would be implemented to revert the sell order
        # It's a placeholder for now
        return {"success": True}
