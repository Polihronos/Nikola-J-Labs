
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"
from hurst_drago import MarketBehaviorAnalyzer


class Plotter():
    def __init__(self, asset, interval, period, window_size, stats_window):
        self.asset = asset
        self.interval = interval
        self.period = period
        self.window_size = window_size
        self.stats_window = stats_window

        # Fetch and clean data
        df = yf.download(
            self.asset,
            period=self.period,
            interval=self.interval,
            progress=False,
            auto_adjust=False
        )

        # Handle yfinance multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs('Close', axis=1, level=0)
        elif 'Close' in df.columns:
            df = df['Close']

        # Ensure it's a clean Series of floats
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]

        self.df = df.astype(float).dropna()
    

    def rolling_hurst(self, analyzer_class):
        """Performs rolling Hurst exponent calculation on price data."""
        results = {
            'Price': [], 'Hc': [], 'P_Value': [], 'is_Significant': []
        }
        dates = []

        print(f"Running Analysis on {len(self.df)} candles with Window {self.window_size}...")

        for i in range(self.window_size, len(self.df)):
            price_chunk = self.df.iloc[i-self.window_size:i]
            current_date = self.df.index[i]
            current_price = self.df.iloc[i]

            returns_sample = np.log(price_chunk / price_chunk.shift(1)).dropna()
            analyzer = analyzer_class(returns_sample)

            try:
                result = analyzer.hurst_exponent(power_min=4)
                if result is not None:
                    dates.append(current_date)
                    results['Price'].append(current_price)
                    results['Hc'].append(result['Hc'])
                    results['P_Value'].append(result['stats']['p-value'])
                    results['is_Significant'].append(result['significant'])

            except ValueError:
                continue
            except Exception as e:
                print(f"Error at {current_date}: {e}")
                continue

        self.analysis_df = pd.DataFrame(results, index=dates)
        if self.analysis_df.empty:
            print("No results generated. Check Window Size or Data Source.")
        return self.analysis_df
    
    def global_statistics(self):
        """Calculates global mean, std, and static bounds for Hurst exponent."""
        if not hasattr(self, 'analysis_df') or self.analysis_df.empty:
            raise ValueError("Run rolling_hurst first to generate analysis_df.")

        hc_mean = self.analysis_df['Hc'].mean()
        hc_std = self.analysis_df['Hc'].std(ddof=1)
        z_score = 1.96

        bounds = {
            'mean': hc_mean,
            'std': hc_std,
            'upper': hc_mean + z_score * hc_std,
            'lower': hc_mean - z_score * hc_std
        }
        return bounds

    def correlation_statistics(self):
        """Calculates Pearson and Spearman correlations between Price and Hc."""
        if not hasattr(self, 'analysis_df') or self.analysis_df.empty:
            raise ValueError("Run rolling_hurst first to generate analysis_df.")

        pearson = self.analysis_df['Price'].corr(self.analysis_df['Hc'], method='pearson')
        spearman = self.analysis_df['Price'].corr(self.analysis_df['Hc'], method='spearman')

        return {'pearson': pearson, 'spearman': spearman}

    def plot_analysis(self):
        """Plots Price, Hurst Exponent, Statistical Significance, and Rolling Correlation."""
        if not hasattr(self, 'analysis_df') or self.analysis_df.empty:
            raise ValueError("Run rolling_hurst first to generate analysis_df.")

        df_corr = self.analysis_df.copy()
        bounds = self.global_statistics()
        correlations = self.correlation_statistics()

        hc_mean_global = bounds['mean']
        hc_std_global = bounds['std']
        hc_upper_global = bounds['upper']
        hc_lower_global = bounds['lower']
        pearson = correlations['pearson']
        spearman = correlations['spearman']

        sns.set_style("whitegrid")

        fig, axes = plt.subplots(
            4, 1, figsize=(14, 18), sharex=True,
            gridspec_kw={'height_ratios': [3, 2.5, 1.5, 2.5], 'hspace': 0.1}
        )

        # PANEL 1: PRICE ACTION
        axes[0].plot(df_corr.index, df_corr['Price'], color='#2c3e50', lw=1.2)
        axes[0].set_title(
            f"ASSET: {self.asset} | INTERVAL: {self.interval} | WINDOW: {self.window_size} | HISTORY: {self.period}", 
            fontsize=14, fontweight='bold', loc='left'
        )
        axes[0].set_ylabel("Price ($)", fontweight='bold')
        axes[0].tick_params(axis='y', labelsize=9)

        # PANEL 2: HURST EXPONENT
        axes[1].plot(df_corr.index, df_corr['Hc'], color='#2980b9', lw=1.5, label='Hurst Exponent (Hc)')
        axes[1].axhline(hc_mean_global, color='black', linestyle='--', lw=1.5, label=f'Center (Mean): {hc_mean_global:.4f}')
        axes[1].axhline(hc_upper_global, color='green', linestyle='-', lw=1.2, label=f'Upper Limit: {hc_upper_global:.4f}')
        axes[1].axhline(hc_lower_global, color='red', linestyle='-', lw=1.2, label=f'Lower Limit: {hc_lower_global:.4f}')
        axes[1].fill_between(df_corr.index, hc_upper_global, hc_lower_global, color='gray', alpha=0.1, label='Normal Range (95%)')
        axes[1].set_ylabel("Hurst Value", fontweight='bold')
        axes[1].set_title(f"Trend Strength (Global Stats | StdDev: {hc_std_global:.4f})", fontsize=11, loc='left')
        axes[1].legend(loc='upper right', frameon=True, fontsize=9, ncol=2)

        # PANEL 3: STATISTICAL SIGNIFICANCE
        axes[2].plot(df_corr.index, df_corr['P_Value'], color='#8e44ad', lw=1.2)
        axes[2].axhline(0.05, color='#c0392b', linestyle='-', linewidth=1.5, label='Significance Threshold (0.05)')
        axes[2].fill_between(df_corr.index, 0, 0.05, color='#27ae60', alpha=0.2, label='Statistically Valid Zone')
        axes[2].set_ylabel("P-Value", fontweight='bold')
        axes[2].set_ylim(0, 0.2)
        axes[2].set_title("Reliability of Calculation (Must be in Green Zone)", fontsize=11, loc='left')
        axes[2].legend(loc='upper right', frameon=True, fontsize=9)

        # PANEL 4: DYNAMIC CORRELATION
        corr_window = 90
        roll_corr = df_corr['Price'].rolling(corr_window).corr(df_corr['Hc'])
        axes[3].plot(df_corr.index, roll_corr, color='#34495e', lw=1)
        axes[3].axhline(0, color='black', linewidth=1)
        axes[3].fill_between(df_corr.index, 0, roll_corr, where=(roll_corr >= 0), color='#27ae60', alpha=0.4, label='Trend Strength ↑ as Price ↑')
        axes[3].fill_between(df_corr.index, 0, roll_corr, where=(roll_corr < 0), color='#c0392b', alpha=0.4, label='Trend Strength ↑ as Price ↓')
        axes[3].set_ylabel(f"Correlation ({corr_window})", fontweight='bold')
        axes[3].set_ylim(-1.1, 1.1)
        axes[3].set_title("Correlation: Price vs. Hurst (Is the trend Bullish or Bearish?)", fontsize=11, loc='left')

        # Text Box for Global Pearson/Spearman
        stats_text = (f"GLOBAL STATISTICS:\n"
                      f"Pearson (Linear): {pearson:.4f}\n"
                      f"Spearman (Rank):  {spearman:.4f}\n\n"
                      f"INTERPRETATION:\n"
                      f"Pearson > 0: Uptrends are stronger\n"
                      f"Pearson < 0: Downtrends are stronger")
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray')
        axes[3].text(0.015, 0.90, stats_text, transform=axes[3].transAxes, fontsize=9, verticalalignment='top', bbox=props)
        axes[3].legend(loc='lower right', frameon=True, fontsize=9)

        plt.tight_layout()
        plt.show()


    def plot_analysis_plotly(self, corr_window=90, html_file="hurst_analysis.html"):
        """Interactive Plotly version of Price, Hurst, P-Value, and Rolling Correlation."""
        if not hasattr(self, 'analysis_df') or self.analysis_df.empty:
            raise ValueError("Run rolling_hurst first to generate analysis_df.")

        df_corr = self.analysis_df.copy()
        bounds = self.global_statistics()
        correlations = self.correlation_statistics()

        hc_mean_global = bounds['mean']
        hc_std_global = bounds['std']
        hc_upper_global = bounds['upper']
        hc_lower_global = bounds['lower']
        pearson = correlations['pearson']
        spearman = correlations['spearman']

        roll_corr = df_corr['Price'].rolling(corr_window).corr(df_corr['Hc'])

        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.3, 0.25, 0.15, 0.25],
            subplot_titles=(
                f"<b>ASSET: {self.asset} | INTERVAL: {self.interval} | WINDOW: {self.window_size} | HISTORY: {self.period}</b>",
                f"<b>Trend Strength (Global Stats | StdDev: {hc_std_global:.4f})</b>",
                "<b>Reliability of Calculation (Must be in Green Zone)</b>",
                "<b>Correlation: Price vs. Hurst (Is the trend Bullish or Bearish?)</b>"
            )
        )

        # PANEL 1: PRICE ACTION
        fig.add_trace(go.Scatter(
            x=df_corr.index, y=df_corr['Price'],
            mode='lines', name='Price',
            line=dict(color='#2c3e50', width=1.5)
        ), row=1, col=1)

        # PANEL 2: HURST EXPONENT
        fig.add_hrect(y0=hc_lower_global, y1=hc_upper_global, fillcolor="gray", opacity=0.15, layer="below",
                      line_width=0, row=2, col=1, annotation_text="Normal Range (95%)", annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=[df_corr.index[0], df_corr.index[-1]],
            y=[hc_mean_global, hc_mean_global],
            mode='lines', name=f'Center (Mean): {hc_mean_global:.4f}',
            line=dict(color='black', width=1.5, dash='dash')
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[df_corr.index[0], df_corr.index[-1]],
            y=[hc_upper_global, hc_upper_global],
            mode='lines', name=f'Upper Limit: {hc_upper_global:.4f}',
            line=dict(color='green', width=1.2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[df_corr.index[0], df_corr.index[-1]],
            y=[hc_lower_global, hc_lower_global],
            mode='lines', name=f'Lower Limit: {hc_lower_global:.4f}',
            line=dict(color='red', width=1.2)
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df_corr.index, y=df_corr['Hc'],
            mode='lines', name='Hurst Exponent (Hc)',
            line=dict(color='#2980b9', width=1.5)
        ), row=2, col=1)

        # PANEL 3: P-Value / Statistical Significance
        fig.add_hrect(y0=0, y1=0.05, fillcolor="#27ae60", opacity=0.2, layer="below", line_width=0, row=3, col=1)
        fig.add_trace(go.Scatter(
            x=[df_corr.index[0], df_corr.index[-1]],
            y=[0.05, 0.05],
            mode='lines', name='Significance Threshold (0.05)',
            line=dict(color='#c0392b', width=1.5)
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df_corr.index, y=df_corr['P_Value'],
            mode='lines', name='P-Value',
            line=dict(color='#8e44ad', width=1.2)
        ), row=3, col=1)

        # PANEL 4: Rolling Correlation
        fig.add_trace(go.Scatter(
            x=df_corr.index, y=roll_corr.where(roll_corr >= 0, 0),
            mode='lines', line=dict(width=0),
            fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.4)',
            name='Trend Strength ↑ as Price ↑'
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df_corr.index, y=roll_corr.where(roll_corr < 0, 0),
            mode='lines', line=dict(width=0),
            fill='tozeroy', fillcolor='rgba(192, 57, 43, 0.4)',
            name='Trend Strength ↑ as Price ↓'
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df_corr.index, y=roll_corr,
            mode='lines', name=f'Corr ({corr_window})',
            line=dict(color='#34495e', width=1.2)
        ), row=4, col=1)
        fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)

        # Text box with global statistics
        stats_html = (
            f"<b>GLOBAL STATISTICS:</b><br>"
            f"Pearson (Linear): {pearson:.4f}<br>"
            f"Spearman (Rank): {spearman:.4f}<br><br>"
            f"<b>INTERPRETATION:</b><br>"
            f"Pearson > 0: Uptrends are stronger<br>"
            f"Pearson < 0: Downtrends are stronger"
        )
        fig.add_annotation(
            xref="x4 domain", yref="y4 domain",
            x=0.01, y=0.95,
            text=stats_html,
            showarrow=False,
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=11, family="Arial")
        )

        # Layout settings
        fig.update_layout(
            autosize=True,
            height=1300,
            template="plotly_white",
            showlegend=True,
            margin=dict(l=60, r=40, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            hoverlabel=dict(namelength=-1)
        )
        fig.update_xaxes(type="date", hoverformat="%Y-%m-%d %H:%M", showspikes=True, spikemode="across", spikesnap="cursor")
        fig.update_yaxes(showspikes=True)
        fig.update_yaxes(title_text="<b>Price ($)</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>Hurst Value</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>P-Value</b>", range=[0, 0.22], row=3, col=1)
        fig.update_yaxes(title_text="<b>Correlation</b>", range=[-1.1, 1.1], row=4, col=1)

        # Save and open
        fig.write_html(html_file, auto_open=True)