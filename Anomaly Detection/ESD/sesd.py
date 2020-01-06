import numpy as np
import scipy.stats as stats
from stldecompose import decompose


def calculate_zscore(ts, hybrid=False):
    if hybrid:
        median = np.ma.median(ts)
        mad = np.ma.median(np.abs(ts - median))
        return (ts - median) / mad
    else:
        return stats.zscore(ts, ddof=1)


def calculate_test_statistic(ts, test_statistics, hybrid=False):
    """
    Calculate the test statistic defined by being the top z-score in the time series.
    """
    corrected_ts = np.ma.array(ts, mask=False)
    for anomalous_index in test_statistics:
        corrected_ts.mask[anomalous_index] = True
    z_scores = abs(calculate_zscore(corrected_ts, hybrid=hybrid))
    max_idx = np.argmax(z_scores)
    return max_idx, z_scores[max_idx]


def calculate_critical_value(size, alpha):
    """
    Calculate the critical value with the formula 
    """
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)

    numerator = (size - 1) * t_dist
    denominator = np.sqrt(size ** 2 - size * 2 + size * t_dist ** 2)

    return numerator / denominator


def seasonal_esd(ts, seasonality=None, hybrid=False, max_anomalies=10, alpha=0.05):
    """
    Compute the Seasonal Extreme Studentized Deviate of a time series depending on the hybrid.
    """
    ts = np.array(ts)
    seasonal = seasonality or int(0.2 * len(ts))  # Seasonality is 20% of the ts if not given.
    decomposition = decompose(ts, period=seasonal)
    residual = ts - decomposition.seasonal - np.median(ts)
    outliers = esd(residual, max_anomalies=max_anomalies, alpha=alpha, hybrid=hybrid)
    return outliers


def esd(ts, max_anomalies=10, alpha=0.05, hybrid=False):
    """
    Compute the Extreme Studentized Deviate of a time series.
    """
    ts = np.copy(np.array(ts))
    test_statistics = []
    total_anomalies = 0
    for curr in range(max_anomalies):
        test_idx, test_val = calculate_test_statistic(ts, test_statistics, hybrid=hybrid)
        critical_value = calculate_critical_value(len(ts) - len(test_statistics), alpha)
        if test_val > critical_value:
            total_anomalies = curr
        test_statistics.append(test_idx)
    anomalous_indices = test_statistics[:total_anomalies + 1]
    return anomalous_indices