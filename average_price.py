class WeightedPriceAggregator:
    def __init__(self, prices, weights):
        """
        prices:
        {
            "Binance": 100.0,
            "Coinbase": 110.0
        }

        weights:
        {
            "Binance": 0.8,
            "Coinbase": 0.2
        }
        """
        self.prices = prices
        self.weights = weights

    def aggregate(self):
        weighted_sum = 0.0
        weight_sum = 0.0

        # Prices define which exchanges actually have data
        for exchange, price in self.prices.items():
            # Checks whether the current exchange has a corresponding weight
            if exchange in self.weights:
                # Retrieves the weight for the current exchange
                weight = self.weights[exchange]
                
                # Multiplies the exchange price by its weight
                weighted_sum += price * weight
                
                # Adds the exchange’s weight to the total weight sum
                weight_sum += weight
                

        # none of the exchanges with prices had a corresponding weight
        if weight_sum == 0:
            raise ValueError("No matching exchanges between prices and weights")

        
        normalized_price = round((weighted_sum / weight_sum), 4)
        

        return normalized_price


prices = {
    "Binance": 100.0,
    "Coinbase": 110.0,
    "Bitget": 105.0,
    "Kraken": 108.0,
    "Kucoin": 107.0
}

weights = {
    "Binance": 0.4,
    "Coinbase": 0.3,
    "Bitget": 0.1,
    "Kraken": 0.1,
    "Kucoin": 0.1
}

aggregator = WeightedPriceAggregator(prices, weights)
result = aggregator.aggregate()

print(result)

# prices = {"Binance": 200.0}  # Coinbase missing
# weights = {"Binance": 0.8, "Coinbase": 0.2}

# aggregator = WeightedPriceAggregator(prices, weights)
# result = aggregator.aggregate()
# print(result)