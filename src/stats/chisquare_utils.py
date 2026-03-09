"""
Chi-Square Test of Independence utility for Breyers Survey Dashboard.
Tests association between two categorical variables.
"""

import pandas as pd
from scipy import stats
from typing import Dict, Any


def run_chisquare(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
) -> Dict[str, Any]:
    """
    Run a chi-square test of independence between two categorical variables.

    Args:
        df: DataFrame with survey data
        row_col: Column name for the row variable
        col_col: Column name for the column variable

    Returns:
        Dictionary with chi2_statistic, p_value, dof, expected_freq,
        and observed_freq crosstab.
    """
    data = df[[row_col, col_col]].dropna()

    if len(data) < 5:
        return {
            "error": f"Insufficient data after dropping NA: n={len(data)}",
            "chi2_statistic": None,
            "p_value": None,
            "dof": None,
            "expected_freq": None,
            "observed_freq": None,
            "n_obs": len(data),
        }

    observed = pd.crosstab(data[row_col], data[col_col])

    chi2, p_val, dof, expected = stats.chi2_contingency(observed)

    expected_df = pd.DataFrame(
        expected,
        index=observed.index,
        columns=observed.columns,
    ).round(2)

    return {
        "chi2_statistic": round(float(chi2), 3),
        "p_value": round(float(p_val), 4),
        "dof": int(dof),
        "expected_freq": expected_df,
        "observed_freq": observed,
        "n_obs": len(data),
        "error": None,
    }
