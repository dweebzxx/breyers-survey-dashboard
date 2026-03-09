"""
Independent Samples T-Test utility for Breyers Survey Dashboard.
Compares means between two concept groups.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any


def run_ttest(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    group1_label: str,
    group2_label: str
) -> Dict[str, Any]:
    """
    Run an independent samples t-test comparing two groups on a numeric metric.

    Args:
        df: DataFrame with survey data
        metric_col: Column name for the numeric metric (e.g., 'Q11_Appeal')
        group_col: Column name for the grouping variable (e.g., 'ConceptLabel')
        group1_label: Label value for group 1
        group2_label: Label value for group 2

    Returns:
        Dictionary with t-test results including means, t-statistic, p-value,
        95% CI for mean difference, and sample sizes.
    """
    g1 = df.loc[df[group_col] == group1_label, metric_col].dropna()
    g2 = df.loc[df[group_col] == group2_label, metric_col].dropna()

    n1 = len(g1)
    n2 = len(g2)

    if n1 < 2 or n2 < 2:
        return {
            "error": f"Insufficient data: Group 1 n={n1}, Group 2 n={n2}. Need at least 2 per group.",
            "group1_mean": None,
            "group2_mean": None,
            "t_statistic": None,
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "n1": n1,
            "n2": n2,
        }

    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

    mean1 = float(g1.mean())
    mean2 = float(g2.mean())
    mean_diff = mean1 - mean2

    # 95% CI for mean difference using Welch's degrees of freedom
    se1 = float(g1.std(ddof=1)) / np.sqrt(n1)
    se2 = float(g2.std(ddof=1)) / np.sqrt(n2)
    se_diff = np.sqrt(se1**2 + se2**2)

    # Welch-Satterthwaite degrees of freedom
    dof = (se1**2 + se2**2)**2 / (
        (se1**2)**2 / (n1 - 1) + (se2**2)**2 / (n2 - 1)
    )
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {
        "group1_mean": round(mean1, 2),
        "group2_mean": round(mean2, 2),
        "t_statistic": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "ci_low": round(float(ci_low), 3),
        "ci_high": round(float(ci_high), 3),
        "n1": n1,
        "n2": n2,
        "mean_diff": round(float(mean_diff), 3),
        "error": None,
    }
