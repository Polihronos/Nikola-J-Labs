import numpy as np
import math
import scipy.stats as st






class MarketBehaviorAnalyzer:

    def __init__(self, returns, alpha=0.05) -> None:

        # self.returns = returns
        self.returns = np.asarray(returns).flatten()

        self.alpha = alpha



    # def hurst_exponent(self, power_min=2, power_max=10, expected_hurst_value=0.5):
    def hurst_exponent(self, power_min=2, power_max=8, expected_hurst_value=0.5): # power_max = 8 because 2^8 = 256, which matches the sample size


        max_observations_count = 2**power_max

        # price_returns_sample = self.returns.iloc[-max_observations_count:]
        price_returns_sample = self.returns[-max_observations_count:]

        power = range(power_min, power_max + 1) #calc lag 256



        lags_list = []

        rescaled_ranges_list = []



        try:

            # if len(self.returns) <= max_observations_count:
            if len(self.returns) < max_observations_count:

                raise ValueError(f'\n  Length of returns df < max length for resample: {len(self.returns)}<{max_observations_count} \n Consider using power_max < {power_max} or increase df size')



            for i in power:

                lag = 2**i

                split_data = np.array_split(price_returns_sample, len(price_returns_sample) // lag)

                for subsample in split_data:

                    if len(subsample) == lag:

                        mean = subsample.mean()

                        deviate = (subsample - mean).cumsum()

                        difference = max(deviate) - min(deviate)

                        stdev = np.std(subsample)

                        if stdev != 0:

                            rescaled_range = difference / stdev

                            lags_list.append(lag)

                            rescaled_ranges_list.append(rescaled_range)



            if len(lags_list) <= 2:

                raise ValueError(f'\n  Number of lags < 2: {len(lags_list)} < 2 \n Consider using power_max > {power_max}')



            # Convert lists to NumPy arrays

            lags_array = np.array(lags_list)

            rescaled_ranges_array = np.array(rescaled_ranges_list)



            # Calculate means of rescaled ranges for each unique lag value

            unique_lags = np.unique(lags_array)

            log_lags = []

            log_rescaled_ranges = []



            for lag in unique_lags:

                indices = np.where(lags_array == lag)

                mean_rescaled_range = np.mean(rescaled_ranges_array[indices])

                if not np.isnan(mean_rescaled_range):

                    log_lags.append(math.log2(lag))

                    log_rescaled_ranges.append(math.log2(mean_rescaled_range))



            if len(log_lags) > 0:

                # Perform linear regression using scipy

                slope, intercept, r_value, p_value, std_err = st.linregress(log_lags, log_rescaled_ranges)

                hurst = slope

                tstat = (slope - expected_hurst_value) / std_err

                n = len(log_lags)

                pvalue = 2 * (1 - st.t.cdf(np.abs(tstat), df=n-2))



                is_significant = 1 if pvalue < self.alpha else 0

                analysis_result = {'isRandomWalk': 0, 'isMeanReverting': 0, 'isTrending': 0}

                if is_significant:

                    if hurst < expected_hurst_value:

                        analysis_result['isMeanReverting'] = 1

                    elif hurst > expected_hurst_value:

                        analysis_result['isTrending'] = 1

                    else:

                        analysis_result['isRandomWalk'] = 1

                else:

                    analysis_result['isRandomWalk'] = 1



                return {'Hc': hurst, 'stats': {'t-stat': tstat, 'p-value': pvalue}, 'analysis': analysis_result, 'significant': is_significant}

            else:

                raise ValueError(f'\n Not enough data for analysis')



        except Exception as e:

            print(f'There was an error: {e}')



    def runs_test(self):

        returns = self.returns[self.returns != 0]

        N = len(returns)

        signs = np.sign(returns)

        # runs = signs.diff().dropna()
        runs = np.diff(signs)

        observed_runs = np.count_nonzero(runs == 2) + np.count_nonzero(runs == -2) + 1

        positive_returns = np.count_nonzero(signs == 1)

        negative_returns = np.count_nonzero(signs == -1)

        expected_runs = 2 * positive_returns * negative_returns / N + 1

        # stdev_runs = math.sqrt(expected_runs * (expected_runs - 1) / (N - 1))
        numerator = 2 * positive_returns * negative_returns * (2 * positive_returns * negative_returns - N)
        denominator = (N ** 2) * (N - 1)
        stdev_runs = math.sqrt(numerator / denominator)

        z_stats = (observed_runs - expected_runs) / stdev_runs

        pvalue = 2 * (1 - st.norm.cdf(np.abs(z_stats)))



        is_significant = 1 if pvalue < self.alpha else 0

        analysis_result = {'isEfficient': 0, 'isMeanReverting': 0, 'isTrending': 0}

        if is_significant:

            if z_stats > 0:

                analysis_result['isMeanReverting'] = 1

            else:

                analysis_result['isTrending'] = 1

        else:

            analysis_result['isEfficient'] = 1



        return {'stats': {'z-stat': z_stats, 'p-value': pvalue}, 'analysis': analysis_result, 'significant': is_significant}



    def run_all_tests(self):

        hurst_results = self.hurst_exponent()

        runs_results = self.runs_test()

        return {'Hurst Exponent': hurst_results, 'Runs Test': runs_results}





