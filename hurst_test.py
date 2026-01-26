from hurst import compute_Hc
from hurst_drago import MarketBehaviorAnalyzer
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from hurst_nikola import QuantitativeHurst

# df = yf.download("SPY", start="2022-06-01", progress=False, auto_adjust=False, multi_level_index=False)
# coin = "BTC-USD"
coin = "SOL-USD"
interval="4h"
period="730d"
# window = 257 
window = 1024
df = yf.download(coin, period=period, progress=False, auto_adjust=False, multi_level_index=False, interval=interval)

if isinstance(df.columns, pd.MultiIndex):
    df = df.xs('Close', axis=1, level=0)
else:
    df = df['Close']

df = df.astype(float)
hurst_values     = pd.Series(dtype=float)

# Library Hurst
lib_hurst_values = pd.Series(dtype=float)

# drago tests
runs_z_values    = pd.Series(dtype=float) 
runs_p_values    = pd.Series(dtype=float)

# nikola
nikola_values    = pd.Series(dtype=float)

for i in range(window, len(df)):
    if i % 50 == 0:
        print(f"Progress: {i}/{len(df)}")
    # Slice the price
    price_subset = df.iloc[i-window:i]
    
    # CALCULATE LOG RETURNS
    # This is the line that makes it work like DeMark/Symbolik
    sample = np.log(price_subset / price_subset.shift(1)).dropna()
    
    # Initialize analyzer with Returns, not Price
    analyzer = MarketBehaviorAnalyzer(sample)

    #  Nikola
    q_hurst = QuantitativeHurst(sample, kind='returns')
    h_dfa = q_hurst.get_dfa()
    if h_dfa is not None:
        nikola_values.loc[df.index[i]] = h_dfa

    runs = analyzer.runs_test()
    runs_z_values.loc[df.index[i]] = runs['stats']['z-stat']
    runs_p_values.loc[df.index[i]] = runs['stats']['p-value']
    
    # Calculate
    try:
        # We use power_max=8 because 2^8 = 256, which matches our sample size
        # h = analyzer.hurst_exponent(power_max=8)
        h = analyzer.hurst_exponent(power_max=9)
        if h is not None:
            hurst_values.loc[df.index[i]] = h['Hc']
    except ValueError:
        pass

    # hurst lib
    try:
        # kind='change' tells the lib these are returns, not raw prices
        H_lib, c, _ = compute_Hc(sample, kind='change', simplified=False)
        lib_hurst_values.loc[df.index[i]] = H_lib
    except:
        pass

print("Plotting...")
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 0.5]})

start_idx = hurst_values.index[0]
df_plot = df.loc[start_idx:]

# Price Chart
ax1.plot(df_plot.index, df_plot.values, color='black', alpha=0.6, lw=1)
ax1.set_title(f"{coin} | {interval} | {period}")
ax1.set_ylabel('Price')
ax1.grid(True, alpha=0.3)

# Hurst Chart
ax2.plot(hurst_values.index, hurst_values.values, color='#333333', lw=1, label='Drago')

ax2.set_ylabel('Hurst')
ax2.set_title('Hurst Exponent (Drago)')
# lib
ax2.plot(lib_hurst_values.index, lib_hurst_values.values, color='cyan', lw=1, alpha=0.7, label='Python Hurst Lib')
# nikola
ax2.plot(nikola_values.index, nikola_values.values, color='orange', lw=1.5, label='Nikola (Research DFA)')
# Zones
ax2.axhline(0.5, color='gray', linestyle='-', alpha=0.5)
ax2.axhline(0.65, color='red', linestyle='--', alpha=0.5, label='Exhaustion/Trend')
ax2.axhline(0.35, color='green', linestyle='--', alpha=0.5, label='Mean Reversion')
ax2.set_ylim(0.3, 0.75) # Set limits to match DeMark view
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

ax3.plot(runs_z_values.index, runs_z_values.values, color='blue', lw=1, label='Runs Z-Score')
ax3.axhline(0, color='black', alpha=0.3)
# Color fills based on Significance Thresholds (approx +/- 1.96)
ax3.fill_between(runs_z_values.index, 1.96, runs_z_values.values, where=(runs_z_values.values > 1.96), color='green', alpha=0.3, label='Sig. Mean Rev')
ax3.fill_between(runs_z_values.index, -1.96, runs_z_values.values, where=(runs_z_values.values < -1.96), color='red', alpha=0.3, label='Sig. Trending')
ax3.set_ylabel('Z-Score')
ax3.legend(loc='upper right', fontsize='small')
ax3.grid(True, alpha=0.3)

# --- AX4: Runs Test P-Value (Significance) ---
# This answers "Is it significant?" explicitly
ax4.plot(runs_p_values.index, runs_p_values.values, color='purple', lw=1, label='P-Value')

# Red line at 0.05 (The standard alpha)
ax4.axhline(0.05, color='red', linestyle='--', linewidth=1, label='Significance (0.05)')

# Fill Green if Significant (Below 0.05), Red if Not
ax4.fill_between(runs_p_values.index, 0, runs_p_values.values, where=(runs_p_values.values <= 0.05), color='green', alpha=0.3, label='Significant')
ax4.fill_between(runs_p_values.index, 0, runs_p_values.values, where=(runs_p_values.values > 0.05), color='red', alpha=0.1, label='Noise')

ax4.set_ylabel('P-Value')
ax4.set_ylim(0, 0.2)   # Zoom in on the important area
ax4.invert_yaxis()     # Invert so "High Significance" (low p-value) is at the top
ax4.legend(loc='upper right', fontsize='small')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
