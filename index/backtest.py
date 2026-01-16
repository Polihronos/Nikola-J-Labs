import pandas as pd
import numpy as np
import json
from index import AICBrain, IndexManager, StateManager, PerformanceAnalyst, EligibilityFilter
from supply_manager import SupplyManager

class Backtester:
    def __init__(self, price_history, universe_assets):
        # price_history: DataFrame of Prices (Index=Date, Cols=Symbol)
        # universe_assets: List of dicts (metadata for filtering)
        self.prices = price_history
        self.universe = universe_assets
        self.results = []
        self.state = {"divisor": 1000000, "constituents": []}
        self.supply_manager = SupplyManager() 
        # For backtest, we might want to mock the supply manager or it will fail 
        # on historical lookups via simple API. 
        # We will assume a static supply for the backtest demo or use a mock.
        self.supply_manager._fetch_circulating_supply = lambda s: 100_000_000 # Mock Override
        
    def run(self, start_date=None, end_date=None, rebalance_freq='M'):
        # Simulation Loop
        if start_date:
            dates = self.prices.loc[start_date:end_date].index
        else:
            dates = self.prices.index
            
        # Resample to rebalance frequency (e.g., Monthly End)
        rebalance_dates = pd.Series(dates).dt.to_period(rebalance_freq).unique().to_timestamp(freq=rebalance_freq)
        
        # Identify valid trading days in history that match rebalance schedule
        # Simple backtest: Just iterate through rebalance points
        
        portfolio_value = 100.0 # Base 100
        current_holdings = {} # {Symbol: Units} not Weights
        
        print(f"=== Starting Backtest ({len(rebalance_dates)} periods) ===")
        
        for date in rebalance_dates:
            if date not in self.prices.index:
                # Find closest business day
                try:
                    date = self.prices.index[self.prices.index.get_indexer([date], method='nearest')[0]]
                except:
                    continue
            
            print(f"Processing {date.date()}...")
            
            # 1. Update Portfolio Value (Mark to Market)
            # Before rebalance, what is our value?
            day_value = 0
            current_prices = self.prices.loc[date]
            
            if current_holdings:
                for sym, units in current_holdings.items():
                    if sym in current_prices:
                        day_value += units * current_prices[sym]
                
                # Check Divisor Continuity (Index Level = Sum(Price * Units) / Divisor ?? 
                # Actually, standard crypto index calc: Index = Sum(Circ_Supply * Price) / Divisor
                # Or Price-Weighted?
                # Our Manager calculates "Weights" (Target %).
                # So we re-allocate total equity based on target weights.
                
                # Let's track Total Equity ($) and convert to Index Level
                portfolio_value = day_value
            
            # 2. Rebalancing Logic (Run the Pipeline)
            # Filter -> AIC -> MinVar
            
            # Snapshot of history up to NOW for covariance
            history_window = self.prices.loc[:date].iloc[-90:] # 90-day lookback
            if len(history_window) < 30: 
                continue # Not enough data to start
            
            # Mock Eligibility (Assume all inputs valid for backtest simplicity, or use metadata)
            # In real backtest, check age/vol at 'date'
            eligible = [a for a in self.universe] # Pass all
            
            # AIC (Need returns)
            returns_window = history_window.pct_change().dropna()
            brain = AICBrain()
            try:
                k = brain.find_optimal_k(returns_window)
            except:
                k = 5
            
            # Selection
            # Rank by Mcap (Price * Dummy Supply)
            # We need Mcap at 'date'.
            # Assume mcap = price * 1e8 for demo
            
            candidates = []
            for a in eligible:
                sym = a['symbol']
                if sym in current_prices:
                if sym in current_prices:
                     # Use mock supply manager (ECS logic)
                     eff_supply = self.supply_manager.get_effective_supply(sym) or 100_000_000
                     mcap = current_prices[sym] * eff_supply
                     candidates.append({'symbol': sym, 'mcap': mcap})
            
            candidates.sort(key=lambda x: x['mcap'], reverse=True)
            constituents = candidates[:k]
            
            # Buffer Rule (Simplified: Skip check for this MVP, assume unconditional rebalance)
            
            # Weighting (MinVar)
            const_df = pd.DataFrame(constituents)
            const_df.set_index('symbol', inplace=True)
            subset_returns = returns_window[[c['symbol'] for c in constituents]]
            
            manager = IndexManager()
            try:
                weights = manager.calculate_min_variance_weights(const_df, subset_returns)
            except:
                weights = manager.calculate_capped_mcap_weights(const_df)
                
            # 3. Execution (Update Holdings)
            # Allocate PortfolioValue into new Units
            # Units = (Equity * Weight) / Price
            new_holdings = {}
            for sym, w in weights.items():
                if sym in current_prices:
                    price = current_prices[sym]
                    units = (portfolio_value * w) / price
                    new_holdings[sym] = units
            
            current_holdings = new_holdings
            
            # 4. Divisor Update?
            # If we track strict Index Level, we update Divisor to match Old Level.
            # Here we track "Portfolio Equity" which naturally drifts. 
            # Index Level = Portfolio Equity (if divisor=const).
            
            self.results.append({
                "date": date,
                "value": portfolio_value,
                "constituents": [c['symbol'] for c in constituents]
            })
            
        return pd.DataFrame(self.results).set_index("date")

    def analyze(self, results_df, benchmark_df=None):
        returns = results_df['value'].pct_change().dropna()
        analyst = PerformanceAnalyst()
        
        print("\n=== Backtest Performance Report ===")
        print(f"Total Return: {(results_df['value'].iloc[-1] / results_df['value'].iloc[0] - 1):.2%}")
        analyst.calculate_max_drawdown(results_df['value'])
        analyst.calculate_psr(returns)
        
        if benchmark_df is not None:
             # Match dates
             common_idx = returns.index.intersection(benchmark_df.index)
             analyst.calculate_tracking_error(returns[common_idx], benchmark_df.loc[common_idx])
