"""
Pearson Correlation Matrix utility for Breyers Survey Dashboard.
Calculates correlation coefficients and p-values for each pair.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def run_correlation(
    df: pd.DataFrame,
    columns: list,
) -> Dict[str, Any]:
    """
    Calculate Pearson correlation matrix with p-values.

    Args:
        df: DataFrame with survey data
        columns: List of numeric column names to correlate

    Returns:
        Dictionary with corr_matrix (DataFrame), pvalue_matrix (DataFrame),
        and n_obs (int).
    """
    data = df[columns].dropna()
    n_obs = len(data)

    if n_obs < 3:
        return {
            "error": f"Insufficient data after dropping NA: n={n_obs}",
            "corr_matrix": None,
            "pvalue_matrix": None,
            "n_obs": n_obs,
        }

    n = len(columns)
    corr_vals = np.ones((n, n))
    p_vals = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_vals[i, j] = 1.0
                p_vals[i, j] = np.nan
            else:
                r, p = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])
                corr_vals[i, j] = r
                p_vals[i, j] = p

    corr_matrix = pd.DataFrame(corr_vals, index=columns, columns=columns).round(3)
    pvalue_matrix = pd.DataFrame(p_vals, index=columns, columns=columns).round(4)

    return {
        "corr_matrix": corr_matrix,
        "pvalue_matrix": pvalue_matrix,
        "n_obs": n_obs,
        "error": None,
    }
