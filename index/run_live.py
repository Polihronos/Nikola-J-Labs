import ccxt
import pandas as pd
import numpy as np
import time
from index import DataEngine, EligibilityFilter, AICBrain, IndexManager, StateManager, GovernanceController
from supply_manager import SupplyManager
from event_governance import EventGovernance

def run_live_pipeline():
    print("=== STARTING LIVE INDEX RUN (REAL DATA) ===\n")
    
    # 1. Setup Infrastructure
    state_manager = StateManager('live_state.json')
    # Load previous state or verify persistence
    state = state_manager.load_state()
    print(f"[Infrastructure] State Loaded. Current Divisor: {state['divisor']}\n")

    # 2. Ingestion (Real Data from Kraken)
    print("=== STEP 1: INGESTION (Checking Top Assets on Kraken) ===")
    exchange = ccxt.kraken()
    markets = exchange.load_markets()
    
    # Filter for major USD pairs to form a "Candidate Universe"
    symbols = [s for s in markets if s.endswith('/USD') and 'stable' not in markets[s].get('info', {}).get('class', '')]
    
    # Ensure we fetch incumbents too (even if volume dropped) for Divisor Calc
    incumbent_symbols = state.get('constituents', [])
    incumbent_pairs = [f"{s}/USD" for s in incumbent_symbols] 
    
    fetch_list = list(set(symbols + incumbent_pairs))
    
    # Helper to get stats
    print("Fetching live ticker data...")
    all_tickers = exchange.fetch_tickers(fetch_list)
    
    # Sort by 24h quote volume to get "Top 10" candidates
    # Only sort valid candidates (from original symbols list), not necessarily incumbents
    candidate_tickers = [t for s, t in all_tickers.items() if s in symbols]
    sorted_tickers = sorted(candidate_tickers, key=lambda x: x['quoteVolume'] or 0, reverse=True)
    top_tickers = sorted_tickers[:8] # Take top 8 for demo
    
    print(f"Top 8 Candidates by Volume: {[t['symbol'] for t in top_tickers]}")
    if incumbent_symbols:
        print(f"Incumbents tracked: {incumbent_symbols}\n")
    
    data_engine = DataEngine(['kraken', 'coinbase']) # Use 2 exchanges for vetting logic
    eligibility = EligibilityFilter()

    governance = GovernanceController()
    supply_manager = SupplyManager()
    event_gov = EventGovernance()
    
    eligible_assets = []
    
    # A. KILL-SWITCH (Governance)
    # Check candidates against blacklist immediately
    # We need SimpleAsset objects or similar for the check_blacklists signature (expects object with .id usually, or simple list?)
    # GovernanceController.check_blacklists expects list of objects with 'id'.
    # Let's adapt it or wrap it. The method does: `[a for a in assets if a.id not in blacklist]`
    # Let's clean top_tickers first.
    
    class TickerWrapper:
        def __init__(self, t): 
            self.id = t['symbol'].split('/')[0] # Using base currency as ID
            self.ticker = t
            
    wrapped_tickers = [TickerWrapper(t) for t in top_tickers]
    safe_wrappers = governance.check_blacklists(wrapped_tickers)
    safe_tickers = [w.ticker for w in safe_wrappers]
    
    if len(safe_tickers) < len(top_tickers):
        print(f"[Governance] Kill-Switch engaged. Removes assets: {set(t['symbol'] for t in top_tickers) - set(t['symbol'] for t in safe_tickers)}")
    
    print("=== STEP 2: FIDELITY & ELIGIBILITY CHECKS ===")
    for ticker in safe_tickers:
        symbol = ticker['symbol']
        base_currency = symbol.split('/')[0]
        
        # B. Vetted Price Check (DataEngine)
        try:
             vetted_price = data_engine.fetch_vetted_price(ticker['symbol'])
        except Exception as e:
            print(f"Skipping {symbol}: Vettng Failed - {e}")
            continue

        # Construct Asset Data Object for Filter
        asset_data = {
            'symbol': base_currency,
            'vetted_custodians': ['A', 'B'], # Checked by Manual Bridge using base_currency
            'mdvt_3mo': ticker['quoteVolume'] * 30, 
            'age_days': 365, 
            'type': 'coin',
            'type': 'coin',
            'mcap': vetted_price * (supply_manager.get_effective_supply(base_currency) or 0), # Using ECS
            'price': vetted_price
        }

        # Run Filter
        if eligibility.is_eligible(asset_data):
            eligible_assets.append(asset_data)
            
    print(f"\n[Result] Eligible Assets: {[a['symbol'] for a in eligible_assets]}")
    
    # 3. Build Universe DF for AIC
    print("\n=== STEP 3: THE BRAIN (Fetching History for AIC) ===")
    history_data = {}
    
    for asset in eligible_assets:
        base = asset['symbol']
        pair = f"{base}/USD"
        print(f"Fetching history for {pair}...")
        try:
            ohlcv = exchange.fetch_ohlcv(pair, timeframe='1m', limit=50)
            closes = [x[4] for x in ohlcv]
            history_data[base] = pd.Series(closes).pct_change().dropna()
        except Exception as e:
            print(f"Failed to fetch history for {base}: {e}")

    if not history_data:
        print("No history data fetched. Exiting.")
        return
    
    # Add history for incumbents if missing (needed for Divisor/MinVar if they stay?)
    # Actually MinVar only needs history for FINAL constituents.
    # But if Incumbent is kept via Buffer Rule, it must be in history_data.
    # Incumbents might NOT be in 'eligible_assets' if they fell out of top 8 but are still "Eligible" technically?
    # For now, simplistic assumption: Incumbents must be in Top 8 candidates to be re-selected, OR we fetch them explicitly.
    # Let's ensure we fetch history for incumbents too if they are eligible-ish.
    # For MVP: If incumbent not in eligible_assets, it might get dropped. That's fine.

    universe_df = pd.DataFrame(history_data).fillna(0)
    
    # Run AIC
    brain = AICBrain()
    try:
        optimal_k = brain.find_optimal_k(universe_df)
    except Exception as e:
        print(f"AIC Failed: {e}")
        optimal_k = len(eligible_assets)

    # 4. Weighting
    print(f"\n=== STEP 4: INDEX CONSTRUCTION (Target K={optimal_k}) ===")
    
    # Sort by "Mcap" 
    eligible_assets.sort(key=lambda x: x['mcap'], reverse=True)
    
    # Instantiate Manager early for Buffer Logic
    index_manager = IndexManager()
    
    # A. Buffer Rule Logic (Incumbent Check)
    # --------------------------------------
    candidates = eligible_assets[:optimal_k]
    final_constituents = []
    
    print(f"[Buffer Rule] Incumbents: {incumbent_symbols}")
    
    if not incumbent_symbols:
        final_constituents = candidates
    else:
        # Check specific replacements for churn stability
        final_constituents = list(candidates)
        
        current_set = set(incumbent_symbols)
        candidate_set = set([c['symbol'] for c in candidates])
        
        adds = candidate_set - current_set
        drops = current_set - candidate_set
        
        if adds:
            print(f"[Buffer Rule] Potential Churn: Adds={adds}, Drops={drops}")
            eligible_map = {a['symbol']: a for a in eligible_assets}
            
            for drop_sym in drops:
                if drop_sym in eligible_map:
                    drop_asset = eligible_map[drop_sym]
                    new_assets = [c for c in final_constituents if c['symbol'] in adds]
                    if new_assets:
                        new_assets.sort(key=lambda x: x['mcap'])
                        weakest_new = new_assets[0]
                        
                        if not index_manager.should_replace(drop_asset, weakest_new):
                            print(f"[Buffer Rule] BLOCKING SWAP: {weakest_new['symbol']} ({weakest_new['mcap']:.0f}) not > 1.1x {drop_sym} ({drop_asset['mcap']:.0f})")
                            if weakest_new in final_constituents:
                                final_constituents.remove(weakest_new)
                            final_constituents.append(drop_asset)
                            if weakest_new['symbol'] in adds:
                                adds.remove(weakest_new['symbol'])

    constituents_df = pd.DataFrame(final_constituents)
    
    # B. Risk-Based Weighting (MinVar)
    # --------------------------------
    const_syms = [c['symbol'] for c in final_constituents]
    subset_history = universe_df[const_syms]
    
    print("\n[Strategy] Calculating Minimum Variance Weights...")
    constituents_df.set_index('symbol', inplace=True)
    weights = index_manager.calculate_min_variance_weights(constituents_df, subset_history)
    
    # 5. Risk Analysis (VaR & CVaR)
    print("\n=== STEP 5: RISK ANALYSIS ===")
    
    var = index_manager.calculate_var(weights, subset_history)
    cvar = index_manager.calculate_cvar(weights, subset_history)
    
    # 6. Finalization & Divisor Continuity
    print("\n=== STEP 6: FINALIZATION & DIVISOR UPDATE ===")
    
    # A. Calculate Old Index Level (using Incumbents)
    old_divisor = state['divisor']
    if not incumbent_symbols:
        # Initial launch
        print("[Divisor] Initial Launch. Keeping divisor constant.")
        new_divisor = old_divisor
    else:
        # Need current Mcap of Old Constituents
        # We need prices for them. We fetched 'all_tickers' earlier using 'incumbent_pairs'.
        old_agg_mcap = 0
        valid_incumbents = 0
        for sym in incumbent_symbols:
            pair = f"{sym}/USD"
            if pair in all_tickers:
                price = all_tickers[pair]['last']
                # Mcap assuming same dummy supply as used elsewhere (1e8)
                # In prod, fetch real supply map.
                # Mcap using ECS for Incumbents (Fetching fresh supply)
                eff_supply = supply_manager.get_effective_supply(sym) or 0
                mcap = price * eff_supply 
                old_agg_mcap += mcap
                valid_incumbents += 1
        
        if valid_incumbents == 0:
             print("[Divisor] Warning: No price data for incumbents. Resetting.")
             current_index_level = 1000
        else:
            current_index_level = old_agg_mcap / old_divisor
            print(f"[Divisor] Old Agg Mcap: {old_agg_mcap:,.0f} | Old Divisor: {old_divisor:.4f} | Index Level: {current_index_level:.2f}")

        # B. Calculate New Divisor (using New Constituents)
        # New_Agg_Mcap / New_Divisor = Current_Index_Level
        # So: New_Divisor = New_Agg_Mcap / Current_Index_Level
        
        # Recalculate Agg Mcap of NEW constituents
        # Note: We use the Mcap from the decision phase (vetted prices) to be consistent
        new_agg_mcap = constituents_df['mcap'].sum()
        
        new_divisor = new_agg_mcap / current_index_level
        print(f"[Divisor] New Agg Mcap: {new_agg_mcap:,.0f} | New Divisor: {new_divisor:.4f}")

    state_manager.save_state(new_divisor, [SimpleAsset(s) for s in weights.index])
    print("Run Complete.")

class SimpleAsset:
    def __init__(self, s): self.symbol = s

if __name__ == "__main__":
    run_live_pipeline()
