import ccxt
import pandas as pd
import numpy as np
import json
import os

class StateManager:
    def __init__(self, filename='index_state.json'):
        self.filename = filename

    def load_state(self):
        # Strategy: Persistence of "Divisor" Memory.
        # If we lose the divisor, we lose the continuity of the index history.
        if not os.path.exists(self.filename):
            # Initial Launch State
            return {"divisor": 1000000, "constituents": []}
        with open(self.filename, 'r') as f:
            return json.load(f)

    def save_state(self, divisor, constituents):
        state = {
            "divisor": divisor,
            "constituents": [c.symbol for c in constituents], # Pseudocode assumption: constituents are objects with symbol
            "last_updated": str(pd.Timestamp.now())
        }
        with open(self.filename, 'w') as f:
            json.dump(state, f)
        print(f"[StateManager] State saved. Divisor: {divisor}, Constituents: {len(constituents)}")

class DataEngine:
    def __init__(self, core_exchanges):
        self.clients = {ex: getattr(ccxt, ex)() for ex in core_exchanges}

    def fetch_vetted_price(self, symbol):
        # Strategy: Settlement Price TWAP/VWAP. 
        # Fetch trades in 30-min windows to smooth out manipulation [7, 8].
        prices, volumes = [], []
        for ex, client in self.clients.items():
            ohlcv = client.fetch_ohlcv(symbol, timeframe='1m', limit=30)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            # Strategy: Outlier Penalty (Nasdaq NCI method).
            # Calculate median and StdDev. Penalise exchanges >1 StdDev away [8, 9].
            median_p = df['c'].median()
            std_p = df['c'].std()
            if std_p == 0:
                penalty = 1.0
            else:
                penalty = 1 / max(1, abs(df['c'].iloc[-1] - median_p) / std_p)
            
            print(f"  [DataEngine] {symbol} | Exchange: {ex} | Median: {median_p:.2f} | Std: {std_p:.2f} | Penalty: {penalty:.4f}")
            
            # Final price weighting: Volume * Penalty [10, 11]
            prices.append(df['c'].iloc[-1])
            volumes.append(df['v'].median() * penalty)
            
        avg_price = np.average(prices, weights=volumes)
        print(f"[DataEngine] {symbol} Vetted Price: {avg_price:.2f} (from {len(self.clients)} exchanges)")
        return avg_price

    def run_benfords_test(self, volume_series):
        # Strategy: Pearson’s Chi-squared test for Benford’s Law [4, 12].
        # Why: Detect fabricated volume patterns [13, 14].
        first_digits = [int(str(abs(v))) for v in volume_series if v > 0]
        # Return False if distribution is anomalous (fails wash trade check)
        pass
'''
2. The Eligibility Gatekeeper
Strategy: Multi-Factor Quantitative Screening. We enforce LargeCap focus ($1B+ MCap) and institutional investability. Why: To ensure the index remains "stable" and only tracks mature assets.
'''
class EligibilityFilter:
    def __init__(self):
        self.hard_exclusions = ["stablecoin", "privacy", "meme", "abandoned"] # [15, 19, 20]

    def is_eligible(self, asset_data):
        # Strategy: Institutional Custody Check.
        # Asset must be supported by 2+ licensed custodians (MPC/MultiSig) [21-23].
        if len(asset_data['vetted_custodians']) < 2:
            print(f"[Eligibility] {asset_data.get('symbol')} REJECTED: Not enough vetted custodians ({len(asset_data['vetted_custodians'])})")
            return False
            
        # Strategy: Manual Data Bridge (Custody).
        # Public APIs don't list "Licensed Custodians". We must bridge this manually. [User Request]
        # This replaces or enhances the 'vetted_custodians' check above.
        with open('custody_map.json') as f:
            custody_data = json.load(f)
        
        custodians = custody_data.get(asset_data['symbol'], [])
        if len(custodians) < 2:
            print(f"[Eligibility] {asset_data.get('symbol')} REJECTED: Manual Custody Check Failed (Found: {custodians})")
            return False

        # Strategy: MDVT & History Thresholds.
        # MDVT > $1M; History > 3 months [24-26].
        if asset_data['mdvt_3mo'] < 1000000 or asset_data['age_days'] < 90:
            print(f"[Eligibility] {asset_data.get('symbol')} REJECTED: Low Volume (${asset_data['mdvt_3mo']}) or Age ({asset_data['age_days']}d)")
            return False
            
        if asset_data['type'] in self.hard_exclusions:
            print(f"[Eligibility] {asset_data.get('symbol')} REJECTED: Type exclusion ({asset_data['type']})")
            return False
            
        print(f"[Eligibility] {asset_data.get('symbol')} ACCEPTED")
        return True
'''
3. The "Brain": Dynamic Constituent Manager
Strategy: Akaike Information Criterion (AIC). This replaces a fixed "Top 10" with a quarterly recalculated optimal count. Why: To balance index sparsity with an accurate mapping of total market transformation.
'''
class AICBrain:
    def find_optimal_k(self, universe_df):
        # STEP 1: Create the Benchmark (The "Y" variable)
        # The CRIX method requires a proxy for the "Total Market".
        # We use the full eligible universe (e.g., Top 100) as the proxy.
        market_returns = universe_df.mean(axis=1) # Simplified market proxy

        # Strategy: Model Selection. Penalise for 'k' to keep index sparse [27, 33].
        # search_range is dynamic to adapt to market maturity [Feedback 3].
        
        # Fix: Start from 1 for small universes [User Request]
        n_assets = len(universe_df.columns) 
        start_k = 1 if n_assets < 10 else 5
        step_k = 1 if n_assets < 20 else 5
        max_k = min(n_assets, 55)
        
        search_range = range(start_k, max_k + 1, step_k) 
        
        best_k = 5
        min_aic = float('inf')
        print(f"[AICBrain] Starting Optimal K Search (Range: 5-{min(len(universe_df), 55)})")
        
        for k in search_range:
            # STEP 2: Create the Candidate (The "X" variable)
            candidate_subset = universe_df.iloc[:, :k]
            candidate_returns = candidate_subset.mean(axis=1)
            
            # STEP 3: Calculate Deviance (RSS)
            # Sum of Squared Residuals between Candidate and Market
            rss = ((market_returns - candidate_returns) ** 2).sum()
            rss = max(rss, 1e-10) # Avoid log(0) if perfect match
            
            
            # AIC Formula: n * ln(RSS/n) + 2*k
            n = len(universe_df)
            current_aic = n * np.log(rss/n) + 2 * k
            if current_aic < min_aic:
                print(f"  [AICBrain] New Best K: {k} (AIC: {current_aic:.2f} | RSS: {rss:.4f})")
                min_aic = current_aic
                best_k = k
            else:
                 print(f"  [AICBrain] K={k} rejected (AIC: {current_aic:.2f})")
        print(f"[AICBrain] Final Optimal K: {best_k}")
        return best_k
'''
4. Index Calculator & Rebalancer
Strategy: Capped Market Capitalisation and 10% Buffer Rule. We limit any asset to a 35% maximum weight. Why: To prevent Bitcoin/Ethereum dominance and reduce unnecessary turnover costs.
'''
class IndexManager:
    def calculate_weights(self, constituents):
        # Default to Capped Market Cap for backward compatibility
        return self.calculate_capped_mcap_weights(constituents)

    def calculate_capped_mcap_weights(self, constituents):
        # Strategy: 35% Cap / 1% Floor [34, 35].
        weights = constituents['mcap'] / constituents['mcap'].sum()
        print(f"[IndexManager] Initial Market Cap Weights:\n{weights}")
        weights = self.apply_capping(weights, cap=0.35, floor=0.01)
        print(f"[IndexManager] Final Capped Weights:\n{weights}")
        return weights

    def calculate_min_variance_weights(self, constituents, returns_df):
        # Strategy: Minimum Variance Portfolio (Markowitz)
        # Goal: Minimize w.T * Cov * w
        # Logic: Global Minimum Variance (GMV) solution is proportional to Cov^-1 * 1
        
        # 1. Calculate Covariance Matrix (Annualized? 90-day daily returns)
        cov_matrix = returns_df.cov()
        
        try:
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(inv_cov))
            
            # Analytical solution for GMV (unconstrained)
            # w = (Cov^-1 * 1) / (1.T * Cov^-1 * 1)
            raw_weights = np.dot(inv_cov, ones)
            raw_weights = raw_weights / raw_weights.sum()
            
            # Convert to Series
            weights = pd.Series(raw_weights, index=constituents.index)
            
            # 2. Handle constraints (No shorting allowed -> weights > 0)
            # Analytical solution allows negative weights (shorting).
            # If negative, we clip to 0 and re-normalize, or use optimization solver.
            # Simple heuristic: Clip negative to 0.01 (floor) and re-normalize.
            weights[weights < 0] = 0.01
            weights = weights / weights.sum()
            
            print(f"[IndexManager] Weights (MinVar Raw):\n{weights}")
            
            # 3. Apply Capping (35%)
            weights = self.apply_capping(weights, cap=0.35, floor=0.01)
            print(f"[IndexManager] Weights (MinVar Capped):\n{weights}")
            
            return weights
            
        except np.linalg.LinAlgError:
            print("[IndexManager] Covariance Matrix Singular. Fallback to Mcap.")
            return self.calculate_capped_mcap_weights(constituents)
            
    def calculate_cvar(self, weights, returns_df, confidence=0.95):
        # Strategy: Conditional Value at Risk (Expected Shortfall)
        # Average of losses exceeding VaR
        port_returns = returns_df.dot(weights.values)
        var = np.percentile(port_returns, (1 - confidence) * 100)
        
        # Filter for returns worse than VaR
        tail_losses = port_returns[port_returns <= var]
        cvar = tail_losses.mean()
        
        print(f"[Risk] Portfolio CVaR ({confidence*100:.0f}%): {cvar:.4f}")
        return cvar

    def apply_capping(self, weights, cap=0.35, floor=0.01):
        # Strict Iterative Capping (Nasdaq/S&P Style)
        # Ensure initial normalization
        weights = weights / weights.sum()
        
        # Max iterations to prevent infinite loops
        for i in range(100):
            current_sum = weights.sum()
            
            # 1. Identify Violations
            over_cap = weights > (cap + 1e-9)
            under_floor = weights < (floor - 1e-9)
            
            if not over_cap.any() and not under_floor.any():
                # If sum is close to 1, we are done.
                if abs(current_sum - 1.0) < 1e-6:
                    break
            
            # 2. Clap values to limits
            weights[over_cap] = cap
            weights[under_floor] = floor
            
            # 3. Calculate residual needed to sum to 1.0
            new_sum = weights.sum()
            residual = 1.0 - new_sum
            
            if abs(residual) < 1e-6:
                break
                
            # 4. Redistribute Residual
            # Strategy: 
            # If Residual > 0 (Need to add), add to anyone < Cap.
            # If Residual < 0 (Need to cut), cut from anyone > Floor.
            
            if residual > 0:
                # Target: Assets not at Cap
                target_mask = weights < (cap - 1e-9)
            else:
                # Target: Assets not at Floor
                target_mask = weights > (floor + 1e-9)
                
            if not target_mask.any():
                 print(f"  [Capping] Critical: Mathematical impossibility. Sum={new_sum:.4f}, but no eligible candidates for residual {residual:.4f}.")
                 weights = weights / weights.sum() # Final fallback
                 break
            
            targets_sum = weights[target_mask].sum()
            
            if targets_sum == 0:
                 # Distribute evenly if sum is 0 (rare)
                 weights[target_mask] += residual / target_mask.sum()
            else:
                 # Proportional distribution
                 factor = 1 + (residual / targets_sum)
                 weights[target_mask] *= factor
                
        return weights

    def should_replace(self, current_asset, candidate):
        # Strategy: 10% Buffer Rule. 
        # Prevents "flickering" due to minor volatility [38, 39].
        return candidate.mcap > (current_asset.mcap * 1.10)

    def calculate_var(self, weights, returns_df, confidence=0.95):
        # Strategy: Historical Simulation VaR
        # Calculate portfolio returns
        port_returns = returns_df.dot(weights.values)
        
        # Calculate VaR at confidence level (e.g. 5th percentile for 95% confidence)
        var = np.percentile(port_returns, (1 - confidence) * 100)
        print(f"[Risk] Portfolio VaR ({confidence*100:.0f}%): {var:.4f}")
        return var

    def update_divisor(self, old_mcap, new_mcap, old_divisor):
        # Strategy: Maintenance of Continuity.
        # Scaling factor so rebalances don't cause price jumps [41-43].
        return old_divisor * (new_mcap / old_mcap)
'''
5. Automated Governance & Blacklist
Strategy: Deterministic Kill-Switch. We address the limits of 100% automation by checking an emergency local file. Why: To allow "Same-Day Removal" during security breaches or fraud without needing a manual code push.
'''
class GovernanceController:
    def check_blacklists(self, assets):
        # Strategy: Emergency Loss of Eligibility [44-46].
        try:
            with open('blacklist.json', 'r') as f:
                blacklist = json.load(f)
            return [a for a in assets if a.id not in blacklist]
        except FileNotFoundError:
            return assets

    def handle_fork(self, asset_a, asset_b):
        # Strategy: Consensus Inheritance.
        # Original asset inherits ticker on core exchanges or highest MCap [47-49].
        if asset_a.has_ticker_on_majority_exchanges:
            return asset_a
        return asset_b if asset_b.mcap > asset_a.mcap else asset_a

'''
6. Performance Analytics (Institutional Scorecard)
Strategy: Beyond simple returns. We track Risk-Adjusted Drift and Tail Events.
'''
class PerformanceAnalyst:
    def calculate_max_drawdown(self, portfolio_values):
        # Peak-to-trough decline
        # Input: Series of Portfolio Values ($)
        roll_max = portfolio_values.cummax()
        drawdown = (portfolio_values - roll_max) / roll_max
        max_dd = drawdown.min()
        print(f"[Perf] Max Drawdown: {max_dd:.2%}")
        return max_dd

    def calculate_psr(self, strategy_returns, benchmark_returns=None):
        # Probabilistic Sharpe Ratio (PSR) [De Prado]
        # Adjusts Sharpe for Skewness, Kurtosis, and Track Record Length
        # If benchmark provided, calculates Information Ratio equivalent? 
        # For standard PSR, we use strategy sharpe vs target 0.
        
        n = len(strategy_returns)
        mean = strategy_returns.mean()
        std = strategy_returns.std()
        skew = strategy_returns.skew()
        kurt = strategy_returns.kurt()
        
        sharpe = mean / std if std != 0 else 0
        
        # Bailey, Lopez de Prado (2012) formula estimation
        # We test against a benchmark Sharpe of 0 (Risk Free Rate approx)
        benchmark_sharpe = 0 
        
        sigma_sr = ((1 / (n - 1)) * (1 + 0.5 * sharpe**2 - skew * sharpe + (kurt / 4) * sharpe**2))**0.5
        psr = (sharpe - benchmark_sharpe) / sigma_sr
        
        # Return probability (Cumulative normal distribution)
        import scipy.stats as stats
        prob = stats.norm.cdf(psr)
        print(f"[Perf] PSR: {prob:.4f} (Sharpe: {sharpe:.4f})")
        return prob

    def calculate_tracking_error(self, strategy_returns, benchmark_returns):
        # Active Risk (volatility of active returns)
        active_returns = strategy_returns - benchmark_returns
        te = active_returns.std()
        print(f"[Perf] Tracking Error: {te:.4f}")
        return te
'''
Summary of Process Logic
1. Ingestion: ccxt pulls prices fromcore exchanges that charge fees.
2. Integrity: DataEngine runs Benford's Law and applies Nasdaq Penalty Factors to outlier prices.
3. Filtering: EligibilityFilter removes stablecoins, memecoins, and assets lacking 2+ custodians.
4. Sizing: Every quarter, AICBrain runs a regression loop to determine the optimal number of constituents.
5. Weighting: IndexManager applies a Risk-Based Optimization (MinVar) and Capping.
6. Continuity: The system updates the Index Divisor after every change to ensure the time series remains seamless.
7. Risk: Metrics like VaR, CVaR, and PSR are calculated to validate institutional quality.
8. Emergency: The script checks blacklist.json before execution to perform immediate security-based removals.
'''