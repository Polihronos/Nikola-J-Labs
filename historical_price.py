import requests
from datetime import datetime
from average_price import WeightedPriceAggregator

class CryptoHistoryManager:
    def __init__(self, asset_symbol, lookback_days=90):
        """
        Constructor accepts the asset and the timeframe.
        """
        self.asset = asset_symbol.upper()
        self.days = lookback_days
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'CryptoAggregator/1.0'})

    def _fetch_binance_data(self):
        """Internal method to fetch Binance history."""
        print(f"   [API] Fetching Binance data for {self.asset}...")
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': f"{self.asset}USDT",
            'interval': '1d',
            'limit': self.days
        }
        data_map = {}
        try:
            resp = self.session.get(url, params=params).json()
            for k in resp:
                # k[0]=time, k[4]=close_price, k[5]=volume
                date_str = datetime.fromtimestamp(k[0]/1000).strftime('%Y-%m-%d')
                data_map[date_str] = {'price': float(k[4]), 'volume': float(k[5])}
        except Exception as e:
            print(f"   [ERR] Binance fetch failed: {e}")
        return data_map

    def _fetch_coinbase_data(self):
        """Internal method to fetch Coinbase history."""
        print(f"   [API] Fetching Coinbase data for {self.asset}...")
        url = f"https://api.exchange.coinbase.com/products/{self.asset}-USD/candles"
        params = {'granularity': 86400} # 1 day in seconds
        data_map = {}
        try:
            resp = self.session.get(url, params=params).json()
            for k in resp:
                # k[0]=time, k[4]=close_price, k[5]=volume
                date_str = datetime.fromtimestamp(k[0]).strftime('%Y-%m-%d')
                data_map[date_str] = {'price': float(k[4]), 'volume': float(k[5])}
        except Exception as e:
            print(f"   [ERR] Coinbase fetch failed: {e}")
        return data_map

    def generate_aggregated_history(self):
        """
        Main method to execute the logic for the last X days.
        """
        # 1. Fetch raw data
        bin_data = self._fetch_binance_data()
        cb_data = self._fetch_coinbase_data()

        # 2. Find common dates
        common_dates = sorted(list(set(bin_data.keys()) & set(cb_data.keys())))
        
        # We slice to ensure we respect the exact 'days' limit if API returns more
        # taking the last N days
        common_dates = common_dates[-self.days:]

        results = []

        print(f"\n--- Starting Aggregation for {self.asset} ({len(common_dates)} days) ---")

        # 3. Iterate through every day
        for date in common_dates:
            b_rec = bin_data[date]
            c_rec = cb_data[date]

            # --- DYNAMIC WEIGHT CALCULATION ---
            vol_bin = b_rec['volume']
            vol_cb = c_rec['volume']
            total_vol = vol_bin + vol_cb

            if total_vol == 0:
                continue

            # Calculate relative weights based on volume
            w_bin = vol_bin / total_vol
            w_cb = vol_cb / total_vol

            
            daily_prices = {
                "Binance": b_rec['price'],
                "Coinbase": c_rec['price']
            }
            
            daily_weights = {
                "Binance": w_bin,
                "Coinbase": w_cb
            }

            
            aggregator = WeightedPriceAggregator(prices=daily_prices, weights=daily_weights)
            final_price = aggregator.aggregate()

            if final_price:
                results.append({
                    "date": date,
                    "aggregated_price": final_price,
                    "volume_share": {
                        "Binance": round(w_bin, 2),
                        "Coinbase": round(w_cb, 2)
                    }
                })
                print(f"Date: {date} | Aggregated Price: {final_price}")

        return results

    def calculate_period_statistics(self):
        """
        NEW METHOD:
        Calculates the weighted average price for the entire period (VWAP)
        and prints the overall market share (weights).
        """
        # Fetch data again (or you could store it in self)
        bin_data = self._fetch_binance_data()
        cb_data = self._fetch_coinbase_data()

        common_dates = sorted(list(set(bin_data.keys()) & set(cb_data.keys())))[-self.days:]

        # Variables for Grand Totals
        grand_vol_bin = 0.0
        grand_vol_cb = 0.0
        grand_value_sum = 0.0 # (Price * Volume) accumulator

        for date in common_dates:
            b = bin_data[date]
            c = cb_data[date]

            # Accumulate Volume
            grand_vol_bin += b['volume']
            grand_vol_cb += c['volume']

            # Accumulate Value (Price * Volume) for both markets
            # This represents the total money flowed into the asset across both exchanges
            grand_value_sum += (b['price'] * b['volume'])
            grand_value_sum += (c['price'] * c['volume'])

        grand_total_vol = grand_vol_bin + grand_vol_cb

        if grand_total_vol == 0:
            print("No volume found for the period.")
            return

        # 1. Calculate Period Weights
        weight_bin = grand_vol_bin / grand_total_vol
        weight_cb = grand_vol_cb / grand_total_vol

        # 2. Calculate Period Aggregated Price (VWAP)
        period_aggregated_price = grand_value_sum / grand_total_vol

        print(f"\n{'='*40}")
        print(f" STATISTICS FOR LAST {self.days} DAYS ({self.asset})")
        print(f"{'='*40}")
        print(f"Total Volume Binance : {grand_vol_bin:,.2f}")
        print(f"Total Volume Coinbase: {grand_vol_cb:,.2f}")
        print(f"Grand Total Volume   : {grand_total_vol:,.2f}")
        print(f"{'-'*40}")
        print(f"MARKET WEIGHT BINANCE : {weight_bin*100:.2f}%")
        print(f"MARKET WEIGHT COINBASE: {weight_cb*100:.2f}%")
        print(f"{'-'*40}")
        print(f"AGGREGATED PERIOD PRICE: ${period_aggregated_price:,.2f}")
        print(f"{'='*40}\n")
            
        return period_aggregated_price

manager = CryptoHistoryManager(asset_symbol="BTC", lookback_days=90)

# 2. Run the process
history_data = manager.generate_aggregated_history()

# 3. Show Summary
print(f"\nSuccessfully generated {len(history_data)} historical records.")
    
# Show the last entry as verification
if history_data:
    last_entry = history_data[-1]
    print(f"Last record ({last_entry['date']}): Price = {last_entry['aggregated_price']}")

print(f"Generated {len(history_data)} daily records.")
manager.calculate_period_statistics()
