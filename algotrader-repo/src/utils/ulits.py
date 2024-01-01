from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp


def get_summary_statistics(
        returns_df: pd.DataFrame, 
        reference_df: pd.DataFrame, 
        filename: str
    ) -> pd.DataFrame:

    stats_df = pd.DataFrame()
    for col in returns_df.columns:
        column = returns_df[col]
        ref_column = reference_df[col]
        stat_values, ref_stat_values, p_value = calculate_values(column, ref_column)
        min_val, q1, median_val, mean_val, q3, max_val, n = stat_values
        ref_min_val, ref_q1, ref_median_val, ref_mean_val, ref_q3, ref_max_val, ref_n = ref_stat_values
        stats_df[col] = [
            min_val, q1, median_val, mean_val, q3, max_val, n, p_value, 
            ref_min_val, ref_q1, ref_median_val, ref_mean_val, ref_q3, ref_max_val, ref_n
        ]

    stats_df = stats_df.transpose()
    stats_df.columns = [
        "min", "q1", "median", "mean", "q3", "max", "n", "p_value", 
        "ref_min_val", "ref_q1", "ref_median_val", "ref_mean_val", "ref_q3", "ref_max_val", "ref_n",
    ]
    stats_df.to_excel(filename)

    return stats_df

def calculate_values(column: pd.Series, ref_column: pd.Series) -> tuple[float]:
    def _calc_values(col: pd.Series) -> tuple[float]:
        min_val = non_zero_returns.min()
        q1 = non_zero_returns.quantile(0.25)
        median_val = non_zero_returns.median()
        mean_val = non_zero_returns.mean()
        q3 = non_zero_returns.quantile(0.75)
        max_val = non_zero_returns.max()
        n = non_zero_returns.count()
        return min_val, q1, median_val, mean_val, q3, max_val, n

    non_zero_returns = column[column != 0.0]
    expected_returns = ref_column[ref_column != 0.0]
    _, p_value = ks_2samp(non_zero_returns, expected_returns)
    stat_values = _calc_values(non_zero_returns)
    ref_stat_values = _calc_values(expected_returns)
     
    return stat_values, ref_stat_values, p_value

