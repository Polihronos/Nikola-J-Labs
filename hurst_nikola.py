import numpy as np
import pandas as pd


class QuantitativeHurst:
    """
    Implements 'Gold Standard' Hurst calculations based on academic literature:
    1. Classic R/S (Mandelbrot/Van Ness 1968)
    2. Lo's Modified R/S (Andrew Lo 1991) - Corrects for short-term memory bias.
    3. DFA (Peng 1994 / Urquhart 2016) - Robust for non-stationary crypto data.
    """

    def __init__(self, series, kind='returns'):
        """
        :param series: Array-like time series (prices or returns).
        :param kind: 'price' or 'returns'. Internally converts to log-returns for calculation.
        """
        series = np.asarray(series, dtype=float).flatten()
        
        # Standardize to Log Returns for calculation consistency
        if kind == 'price':
            # Handle zeros in price to avoid log errors
            series[series == 0] = np.nan
            series = pd.Series(series).ffill().bfill().values
            self.returns = np.diff(np.log(series))
        else:
            self.returns = series
            
        # Drop NaNs created by diff or logging
        self.returns = self.returns[~np.isnan(self.returns)]
        self.N = len(self.returns)

    def get_classic_rs(self, min_window=10):
        """
        Calculates the Hurst Exponent using classic Rescaled Range (R/S) analysis.
        Reference: Mandelbrot & Van Ness (1968)
        """
        if self.N < min_window * 2: return None

        # Create logarithmic scales
        max_scale = self.N // 2
        scales = np.floor(np.logspace(np.log10(min_window), np.log10(max_scale), num=16)).astype(int)
        scales = np.unique(scales) # Remove duplicates
        
        rs_values = []
        
        for scale in scales:
            # Split data into chunks of size 'scale'
            num_chunks = self.N // scale
            
            # We must drop the tail to make even chunks
            # Note: This is crucial to avoid the 'Look-Ahead Bias' mentioned earlier
            reshaped_data = self.returns[:num_chunks*scale].reshape(num_chunks, scale)
            
            # Calculate means per chunk
            means = reshaped_data.mean(axis=1, keepdims=True)
            
            # Deviations
            z = reshaped_data - means
            
            # Cumulative deviations
            y = np.cumsum(z, axis=1)
            
            # Ranges
            r = np.max(y, axis=1) - np.min(y, axis=1)
            
            # Standard deviations (ddof=1 for sample std)
            s = np.std(reshaped_data, axis=1, ddof=1)
            
            # Handle division by zero
            s[s == 0] = 1e-9
            
            # Average R/S for this scale
            rs = np.mean(r / s)
            rs_values.append(rs)
            
        # Log-Log Regression
        poly = np.polyfit(np.log(scales), np.log(rs_values), 1)
        H = poly[0]
        
        return H

    def get_lo_modified_rs(self):
        """
        Calculates Andrew Lo's Modified R/S Statistic.
        This adjusts the denominator to account for short-term autocorrelation.
        
        Reference: Lo, A. W. (1991). "Long-term memory in stock market prices".
        """
        # 1. Standard deviations and mean
        mu = np.mean(self.returns)
        centered = self.returns - mu
        
        # Standard Variance (Short-term)
        sigma_sq = np.mean(centered**2)
        
        if sigma_sq == 0: return None

        # 2. Calculate Optimal Lag (q) based on Lo's formula for N
        # Lo suggests q ~ 4 * (N/100)^(1/4)
        q = int(4 * (self.N / 100)**0.25)
        if q >= self.N // 2: q = (self.N // 2) - 1
        
        # 3. Calculate Weighted Autocovariance Correction
        weighted_sum = 0
        for j in range(1, q + 1):
            # Autocovariance at lag j
            gamma_j = np.mean(centered[:-j] * centered[j:])
            
            # Bartlett weight
            weight = 1 - (j / (q + 1))
            weighted_sum += weight * gamma_j
            
        # 4. Modified Variance
        sigma_sq_mod = sigma_sq + 2 * weighted_sum
        
        if sigma_sq_mod <= 0: sigma_sq_mod = sigma_sq # Fallback if correction goes negative
        
        sigma_mod = np.sqrt(sigma_sq_mod)
        
        # 5. Calculate R (Range) on the FULL series (Lo uses global R/S statistic test)
        cum_dev = np.cumsum(centered)
        R = np.max(cum_dev) - np.min(cum_dev)
        
        # 6. Lo's V Statistic
        V = R / (sigma_mod * np.sqrt(self.N))
        
        # Lo doesn't produce an "exponent" H directly in the same way, 
        # but V can be mapped back to H approx via H = log(V)/log(N) + 0.5 for small N,
        # usually we use V to test Null Hypothesis. 
        # For compatibility, we return the V-stat and a significance boolean.
        
        # Lo's Confidence Interval (95%) for V is [0.809, 1.862]
        is_random_walk = (0.809 <= V <= 1.862)
        
        return {'V_stat': V, 'is_random': is_random_walk, 'lag_used': q}

    def get_dfa(self, min_window=10):
        """
        Detrended Fluctuation Analysis (DFA).
        Often preferred in Crypto/High-Frequency papers (Urquhart/Bariviera).
        Robust against non-stationary trends.
        """
        if self.N < min_window * 4: return None
        
        # 1. Integrate the series (Cumulative Sum of centered data)
        # Note: DFA works on the integrated path
        y = np.cumsum(self.returns - np.mean(self.returns))
        
        scales = np.floor(np.logspace(np.log10(min_window), np.log10(self.N // 4), num=16)).astype(int)
        scales = np.unique(scales)
        
        fluctuations = []
        
        for scale in scales:
            
            rms = []
            
            # 1. Forward Scan
            for i in range(0, self.N - scale + 1, scale):
                seg_y = y[i : i+scale]
                x = np.arange(scale)
                
                # Detrend: Fit a polynomial (Order 1 = Linear trend)
                coefs = np.polyfit(x, seg_y, 1)
                trend = np.polyval(coefs, x)
                rms.append(np.sqrt(np.mean((seg_y - trend)**2)))
            
            # 2. Backward Scan (Ensures tail data is used - Kantelhardt 2001)
            for i in range(self.N, scale - 1, -scale):
                seg_y = y[i-scale : i]
                x = np.arange(scale)
                
                # Detrend
                coefs = np.polyfit(x, seg_y, 1)
                trend = np.polyval(coefs, x)
                rms.append(np.sqrt(np.mean((seg_y - trend)**2)))
                
            fluctuations.append(np.mean(rms))
            
        # Log-Log Regression
        valid_idx = np.where(np.array(fluctuations) > 0)
        if len(valid_idx[0]) < 3: return None
        
        poly = np.polyfit(np.log(scales[valid_idx]), np.log(np.array(fluctuations)[valid_idx]), 1)
        H_dfa = poly[0]
        
        return H_dfa