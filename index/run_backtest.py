import pandas as pd
import numpy as np
import ccxt
import time
from backtest import Backtester

def fetch_real_history(days=365, top_n=12):
    print(f"Connecting to Kraken to fetch {days} days of history for top {top_n} assets...")
    exchange = ccxt.kraken()
    markets = exchange.load_markets()
    
    # 1. Select Candidates (Top Volume USD pairs)
    symbols = [s for s in markets if s.endswith('/USD') and 'stable' not in markets[s].get('info', {}).get('class', '')]
    tickers = exchange.fetch_tickers(symbols)
    sorted_tickers = sorted(tickers.values(), key=lambda x: x['quoteVolume'] or 0, reverse=True)
    top_assets = [t['symbol'] for t in sorted_tickers[:top_n]]
    
    print(f"Candidates: {top_assets}")
    
    # 2. Fetch History
    all_data = {}
    
    for symbol in top_assets:
        print(f"Fetching {symbol}...")
        try:
            # Kraken limit is usually 720 candles. 1d candles for 365 days is fine.
            # since parameter depends on exchange, we use loop if needed, but for 1y 1d it's usually 1 call.
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=days)
            if not ohlcv:
                continue
                
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df['date'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('date', inplace=True)
            
            # Use Close price
            all_data[symbol.split('/')[0]] = df['c']
            
            time.sleep(exchange.rateLimit / 1000) # Respect rate limits
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            
    # 3. Align
    price_df = pd.DataFrame(all_data)
    price_df.fillna(method='ffill', inplace=True) # LOCF
    price_df.dropna(how='all', inplace=True)
    
    return price_df, list(price_df.columns)

def run_backtest_with_real_data():
    price_df, symbols = fetch_real_history(days=365)
    
    if price_df.empty:
        print("No data fetched. Aborting.")
        return

    universe = [{'symbol': s} for s in symbols]
    
    print(f"\n=== Initializing Backtest Engine (Range: {price_df.index[0].date()} to {price_df.index[-1].date()}) ===")
    backtester = Backtester(price_df, universe)
    
    # Run Simulation (Monthly Rebalance)
    results = backtester.run(rebalance_freq='M')
    
    if not results.empty:
        print("\n=== Real Data Simulation Complete ===")
        print(results.tail())
        
        # Analyze
        backtester.analyze(results)
    else:
        print("Backtest produced no results.")

if __name__ == "__main__":
    run_backtest_with_real_data()
