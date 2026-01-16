import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional
from decimal import Decimal, getcontext

getcontext().prec = 28

def construct_depth_df_for_aggregation(depth_data_dict: Dict[str, Any]) -> pd.DataFrame:
    if not depth_data_dict:
        raise ValueError("Input depth data dictionary is empty.")
    if not isinstance(depth_data_dict, dict):
        raise ValueError("Input must be a dictionary.")
    data_rows = []
    for asset_id, exchanges in depth_data_dict.items():
        for exchange, depth in exchanges.items():
            for side in ("bid", "ask"):
                levels = depth.get(side, [])
                for price, volume in levels:
                    data_rows.append({
                        "asset_id": int(asset_id),
                        "exchange": exchange,
                        "side": side,
                        "price": float(price),
                        "volume": float(volume),
                    })
    return pd.DataFrame(data_rows)


class DepthAggregator:
    def __init__(self, constructed_depth_df: pd.DataFrame, price_decimals: Optional[int] = None) -> None:

        if constructed_depth_df.empty:
            raise ValueError("Input depth DataFrame is empty.")
        
        if not isinstance(constructed_depth_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        self.depth_df = constructed_depth_df
        self.price_decimals = price_decimals 

    def aggregate_depths(self) -> pd.DataFrame:
        df = self.depth_df.copy()

        price_col = "price"
        if self.price_decimals is not None:
            df["price_key"] = df["price"].round(self.price_decimals)
            price_col = "price_key"

        agg_df = (
            df.groupby(["asset_id", "side", price_col], as_index=False)
              .agg(volume=("volume", "sum"))
              .rename(columns={price_col: "price"})
        )
        return agg_df

    def generate_order_aggregated_books(self, as_json: bool = False):
        agg_df = self.aggregate_depths()

        order_books = {}
        for asset_id, g in agg_df.groupby("asset_id"):
            bids = g[g["side"] == "bid"].sort_values("price", ascending=False)[["price", "volume"]].values.tolist()
            asks = g[g["side"] == "ask"].sort_values("price", ascending=True)[["price", "volume"]].values.tolist()
            order_books[int(asset_id)] = {"bid": bids, "ask": asks}

        return json.dumps(order_books) if as_json else order_books

    # Check total volumes by asset_id and side before and after aggregation match
    def total_volume_sanity_check(self) -> pd.DataFrame:
        agg_df = self.aggregate_depths()

        raw_totals = (
            self.depth_df.groupby(["asset_id", "side"], as_index=False)
              .agg(expected_total_volume=("volume", "sum"))
        )

        agg_totals = (
            agg_df.groupby(["asset_id", "side"], as_index=False)
                  .agg(actual_total_volume=("volume", "sum"))
        )

        recon = raw_totals.merge(agg_totals, on=["asset_id", "side"], how="left")
        recon["diff"] = recon["expected_total_volume"] - recon["actual_total_volume"]

        total_volume_exprected_bids = recon[recon["side"] == "bid"]["expected_total_volume"].sum()
        total_volume_actual_bids = recon[recon["side"] == "bid"]["actual_total_volume"].sum()

        total_volume_exprected_ask = recon[recon["side"] == "ask"]["expected_total_volume"].sum()
        total_volume_actual_ask = recon[recon["side"] == "ask"]["actual_total_volume"].sum()

        total_diff_bids = total_volume_exprected_bids - total_volume_actual_bids
        total_diff_ask = total_volume_exprected_ask - total_volume_actual_ask

        return recon

    # Check volumes at each price level before and after aggregation match
    def price_level_sanity_check(self) -> pd.DataFrame:
        agg_df = self.aggregate_depths()

        raw_by_price = (
            self.depth_df.groupby(["asset_id", "side", "price"], as_index=False)
              .agg(expected_volume=("volume", "sum"))
        )

        recon = raw_by_price.merge(
            agg_df.rename(columns={"volume": "actual_volume"}),
            on=["asset_id", "side", "price"],
            how="left"
        )

        recon["diff"] = recon["expected_volume"] - recon["actual_volume"]

        return recon

    # Check for agregation duplicates: same asset_id, side, price from multiple exchanges
    # use min exchanges to filter results based on the rule: How many distinct exchanges has this exact price level for the same asset_id and side.
    def detect_duplicate_entries(self, min_exchanges: int = 2) -> pd.DataFrame:
        dupes = (
            self.depth_df.groupby(["asset_id", "side", "price"])
              .agg(
                  row_count=("exchange", "size"),
                  exchanges=("exchange", lambda x: sorted(set(x))),
                  exchange_count=("exchange", "nunique"),
                  total_volume=("volume", "sum")
              )
              .reset_index()
              .query("exchange_count >= @min_exchanges")
              .sort_values(["asset_id", "side", "exchange_count"], ascending=[True, True, False])
        )
        return dupes
    
    def agregation_quality_check(self, precision: float = 1e-9) -> None:
        
        total_recon = self.total_volume_sanity_check()
        price_recon = self.price_level_sanity_check()

        if price_recon["diff"].abs().max() > precision:
            raise ValueError("Price level volume mismatch exceeds precision threshold.")
        
        
        if total_recon["diff"].abs().max() > precision:
            raise ValueError("Total volume mismatch exceeds precision threshold.")

    def volume_side_test(self, aggregated_data):
        # 1. Calculate Raw Totals (from self.depth_df DataFrame)
        raw_bids = Decimal('0')
        raw_asks = Decimal('0')
        for _, row in self.depth_df.iterrows():
            if row['side'] == 'bid':
                raw_bids += Decimal(str(row['volume']))
            else:
                raw_asks += Decimal(str(row['volume']))

        # 2. Calculate Aggregated Totals (from argument)
        agg_bids = Decimal('0')
        agg_asks = Decimal('0')
        for book in aggregated_data.values():
            for order in book['bid']: agg_bids += Decimal(str(order[1]))
            for order in book['ask']: agg_asks += Decimal(str(order[1]))

        # 3. Print Comparison
        print("-" * 60)
        print(f"BIDS Check: Raw {raw_bids:.4f} | Agg {agg_bids:.4f}")
        print(f"Diff (Agg - Raw): {agg_bids - raw_bids:.10f}")
        print("-" * 60)
        print(f"ASKS Check: Raw {raw_asks:.4f} | Agg {agg_asks:.4f}")
        print(f"Diff (Agg - Raw): {agg_asks - raw_asks:.10f}")
        print("-" * 60)


# Load data
with open("depth_data_new.json") as file:
    raw_data = json.load(file)

# Create DataFrame
df = construct_depth_df_for_aggregation(raw_data)
print(f"Loaded {len(df)} rows from {len(raw_data)} assets")

# Create aggregator object
aggregator = DepthAggregator(df)

# Run aggregation
order_books = aggregator.generate_order_aggregated_books()
print(f"Generated order books for {len(order_books)} assets")

# Run quality checks
print("\n--- Quality Checks ---")
aggregator.agregation_quality_check()
print("All quality checks passed!")

# Show totals
total_recon = aggregator.total_volume_sanity_check()
pd.set_option('display.float_format', '{:.4f}'.format)
print(f"\nTotal volume reconciliation:\n{total_recon}")

# Show duplicates (price levels appearing on multiple exchanges)
dupes = aggregator.detect_duplicate_entries(min_exchanges=2)
print(f"\nDuplicate price levels (across exchanges): {len(dupes)} found")
if not dupes.empty:
    print(dupes.head(10))

# Volume side test with Decimal precision
aggregator.volume_side_test(order_books)