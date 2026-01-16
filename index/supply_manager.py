import requests
import json
import time
import numpy as np

class SupplyManager:
    def __init__(self, map_file='supply_map.json'):
        self.map_file = map_file
        self.supply_map = self._load_map()
        self.api_url = "https://api.coingecko.com/api/v3"
        self.cache = {} # Simple in-memory cache to avoid rate limits
        
    def _load_map(self):
        try:
            with open(self.map_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[SupplyManager] Warning: {self.map_file} not found. Using empty map.")
            return {}

    def get_effective_supply(self, symbol):
        """
        Calculates Effective Coin Supply (ECS) = Circulating Supply * Free-Float Factor.
        """
        # 1. Fetch Circulating Supply (Raw)
        circulating_supply = self._fetch_circulating_supply(symbol)
        
        if circulating_supply is None:
            # Fallback if API fails or coin not found
            print(f"[SupplyManager] Failed to fetch supply for {symbol}. Returning None.")
            return None

        # 2. Get Free-Float Factor (FFF)
        fff = self._get_free_float_factor(symbol)
        
        # 3. Calculate ECS
        ecs = circulating_supply * fff
        print(f"  [SupplyManager] {symbol}: Circ={circulating_supply:,.0f} * FFF={fff:.2f} = ECS={ecs:,.0f}")
        return ecs

    def _fetch_circulating_supply(self, symbol):
        # Check cache first
        if symbol in self.cache:
            return self.cache[symbol]

        # Fetch from CoinGecko
        # Note: CoinGecko uses IDs (bitcoin) not Symbols (BTC). 
        # For a robust system, we need a mapping. For this implementation, we try to guess or use a search.
        # Cost-saving: we can search once and cache the ID.
        
        coin_id = self._resolve_coingecko_id(symbol)
        if not coin_id:
            return None
            
        try:
            # Rate limit handling (simple sleep)
            time.sleep(1.5) # CoinGecko Free API limit ~10-30 calls/min
            
            url = f"{self.api_url}/coins/{coin_id}"
            params = {'localization': 'false', 'tickers': 'false', 'market_data': 'true', 'community_data': 'false', 'developer_data': 'false'}
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'market_data' in data:
                supply = data['market_data']['circulating_supply']
                self.cache[symbol] = supply
                return supply
            else:
                return None
        except Exception as e:
            print(f"[SupplyManager] API Error for {symbol}: {e}")
            return None

    def _resolve_coingecko_id(self, symbol):
        # Simplified mapping for common assets. 
        # In production, this should be a robust lookup or maintained map.
        mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'USDC': 'usd-coin',
            'USDT': 'tether',
            'UNI': 'uniswap',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash'
        }
        return mapping.get(symbol.upper())

    def _get_free_float_factor(self, symbol):
        # Strategy: Look up in map, else use Median of universe
        symbol = symbol.upper()
        if symbol in self.supply_map:
            return self.supply_map[symbol].get('free_float_factor', 1.0)
        
        # Fallback: Median of known factors
        factors = [d['free_float_factor'] for d in self.supply_map.values() if 'free_float_factor' in d]
        if factors:
            median_fff = float(np.median(factors))
            print(f"[SupplyManager] {symbol} not in map. Using Median FFF: {median_fff:.2f}")
            return median_fff
        
        return 1.0 # Absolute fallback
