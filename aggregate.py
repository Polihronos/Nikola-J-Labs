import json
import pandas as pd

class Aggregator:      
    def __init__(self, data):
        self.data = data
        
        # Check structure immediately upon creation
        self._validate_structure()
    
    def _validate_structure(self):
        # Dictionary check
        if not isinstance(self.data, dict):
            raise ValueError("Data Error: Input is not a valid Dictionary.")

        # Structure Check
        for entity_id, exchanges in self.data.items():
            if not isinstance(exchanges, dict):
                raise ValueError(f"Data Error: ID {entity_id} does not contain a dictionary of exchanges.")
            
            for exchange_name, book in exchanges.items():
                if not isinstance(book, dict) or 'bid' not in book or 'ask' not in book:
                    raise ValueError(f"Data Error: Exchange '{exchange_name}' in ID {entity_id} is missing 'bid' or 'ask'.")
    
    # Private aggregation helper
    def _aggregate(self, orders, ascending=True):
        # Put orders into a pandas table with 'price' and 'volume' columns
        df = pd.DataFrame(orders, columns=['price', 'volume'])
        
        # Group by price and aggregates all volumes for each price
        grouped = df.groupby('price', as_index=False)['volume'].sum()
        
        # Sort values based on standard financial order book rules
        grouped = grouped.sort_values('price', ascending=ascending)
        
        # Convert back to list of lists
        result = grouped.values.tolist()
        
        return result
    
    def aggregate(self):
        result = {}
        
        # Go through each id (for every key value pair in the data)
        for entity_id, exchanges in self.data.items():
            
            # Collect all orders from all exchanges for this ID
            all_bids = []
            all_asks = []
            
            for exchange_name, order_book in exchanges.items():
                all_bids.extend(order_book.get('bid', []))
                all_asks.extend(order_book.get('ask', []))
            
            # Aggregate combined lists. 
            # Bids = Descending (Highest price first), Asks = Ascending (Lowest price first)
            result[entity_id] = {
                'bid': self._aggregate(all_bids, ascending=False),
                'ask': self._aggregate(all_asks, ascending=True)
            }
                
        return result

    def aggregate_json(self):
        return json.dumps(self.aggregate())
    
    '''
    Testing part below
    '''
    
    def total_volume_test(self, data):
        total = 0.0
        
        for entity_id, content in data.items():
            
            # Case 1: Aggregated Data (Structure: ID -> Bid/Ask)
            # If 'bid' is directly here, we don't loop through exchanges
            if 'bid' in content:
                for order in content['bid']: 
                    total = total + order[1]
                for order in content['ask']: 
                    total = total + order[1]
            
            # Case 2: Raw Data (Structure: ID -> Exchange -> Bid/Ask)
            else:
                for exchange_name, order_book in content.items():
                    for order in order_book['bid']:
                        total = total + order[1]
                    for order in order_book['ask']:
                        total = total + order[1]
    
        return total

    def duplicates_test(self, data):
        # Go through every ID
        for entity_id, book in data.items():
            # Check both bid and ask sides
            for side in ['bid', 'ask']:
                # Create a list of JUST the prices (index 0)
                prices = [order[0] for order in book[side]]
                
                # Compare length of list vs length of set
                # Sets remove duplicates, so if lengths differ, we have duplicates
                if len(prices) != len(set(prices)):
                    return f"FAIL: Duplicates found in {entity_id} {side}"
        
        return "PASS: No duplicates found"
    
    def volume_side_test(self, aggregated_data):
        # 1. Calculate Raw Totals (from self.data)
        raw_bids = 0.0
        raw_asks = 0.0
        for exchanges in self.data.values():
            for book in exchanges.values():
                for order in book['bid']: raw_bids += order[1]
                for order in book['ask']: raw_asks += order[1]

        # 2. Calculate Aggregated Totals (from argument)
        agg_bids = 0.0
        agg_asks = 0.0
        for book in aggregated_data.values():
            for order in book['bid']: agg_bids += order[1]
            for order in book['ask']: agg_asks += order[1]

        # 3. Print Comparison
        print("-" * 60)
        print(f"BIDS Check: Raw {raw_bids:.4f} | Agg {agg_bids:.4f}")
        print(f"Diff (Agg - Raw): {agg_bids - raw_bids:.10f}")
        print("-" * 60)
        print(f"ASKS Check: Raw {raw_asks:.4f} | Agg {agg_asks:.4f}")
        print(f"Diff (Agg - Raw): {agg_asks - raw_asks:.10f}")
        print("-" * 60)

    


# Load data and it will crash if it's not json
with open("depth_data_new.json") as file:
    raw_data = json.load(file)

# create object
aggregator = Aggregator(raw_data)

aggregated_data = aggregator.aggregate_json()

# print(aggregated_data)


print(f"total volume raw data:        {aggregator.total_volume_test(raw_data)}")
print(f"total volume aggregated data: {aggregator.total_volume_test(aggregator.aggregate())}")
print(f"duplicates test:              {aggregator.duplicates_test(aggregator.aggregate())}")



aggregator.volume_side_test(aggregator.aggregate())








