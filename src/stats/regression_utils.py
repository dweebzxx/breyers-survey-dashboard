"""
Linear and Logistic Regression utilities for Breyers Survey Dashboard.
Supports dynamic dummy coding for ClaimCell.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any, List


# Short labels for ClaimCell reference group dropdown
CLAIM_CELL_LABELS = {
    1: "Low Sugar (Cell 1)",
    2: "High Protein (Cell 2)",
    3: "Both (Cell 3)",
}


def _build_regression_data(
    df: pd.DataFrame,
    dv_col: str,
    iv_cols: List[str],
    reference_cell: int = 1,
) -> pd.DataFrame:
    """
    Build the regression design matrix with ClaimCell dummy coding.

    Args:
        df: DataFrame with survey data
        dv_col: Dependent variable column name
        iv_cols: List of IV column names (excluding ClaimCell)
        reference_cell: ClaimCell code to use as reference category (1, 2, or 3)

    Returns:
        DataFrame with DV + IVs + ClaimCell dummies, dropna applied
    """
    dummy_names = {
        1: {2: "ClaimCell_HighProtein", 3: "ClaimCell_Both"},
        2: {1: "ClaimCell_LowSugar", 3: "ClaimCell_Both"},
        3: {1: "ClaimCell_LowSugar", 2: "ClaimCell_HighProtein"},
    }
    cells_to_dummy = {k: v for k, v in dummy_names[reference_cell].items()}

    data = df[[dv_col] + iv_cols + ["ClaimCell"]].dropna().copy()

    for cell_code, dummy_col in cells_to_dummy.items():
        data[dummy_col] = (data["ClaimCell"] == cell_code).astype(int)

    return data, list(cells_to_dummy.values())


def run_linear_regression(
    df: pd.DataFrame,
    dv_col: str = "Q12_PurchaseIntent",
    iv_cols: List[str] = None,
    reference_cell: int = 1,
) -> Dict[str, Any]:
    """
    Run OLS linear regression predicting Purchase Intent.

    Args:
        df: DataFrame with survey data
        dv_col: Dependent variable column (default: Q12_PurchaseIntent)
        iv_cols: IVs to include (default: Q8 attribute importance ratings)
        reference_cell: ClaimCell reference group (default: 1 = Low Sugar)

    Returns:
        Dictionary with coefficients, std_errors, p_values, conf_int,
        r_squared, and model_summary.
    """
    if iv_cols is None:
        iv_cols = [f"Q8_AttrImportance_{i}" for i in range(1, 8)]

    data, dummy_cols = _build_regression_data(df, dv_col, iv_cols, reference_cell)

    predictor_cols = iv_cols + dummy_cols
    X = sm.add_constant(data[predictor_cols])
    y = data[dv_col]

    model = sm.OLS(y, X).fit()

    conf_int = model.conf_int()

    results = {
        "model_type": "OLS",
        "n_obs": int(model.nobs),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic": round(float(model.fvalue), 3),
        "f_pvalue": round(float(model.f_pvalue), 4),
        "coefficients": {
            name: round(float(val), 4)
            for name, val in model.params.items()
        },
        "std_errors": {
            name: round(float(val), 4)
            for name, val in model.bse.items()
        },
        "p_values": {
            name: round(float(val), 4)
            for name, val in model.pvalues.items()
        },
        "conf_int_low": {
            name: round(float(conf_int.loc[name, 0]), 4)
            for name in conf_int.index
        },
        "conf_int_high": {
            name: round(float(conf_int.loc[name, 1]), 4)
            for name in conf_int.index
        },
        "model_summary": model.summary(),
        "predictor_cols": predictor_cols,
        "reference_cell": reference_cell,
        "error": None,
    }
    return results


def run_logistic_regression(
    df: pd.DataFrame,
    dv_col: str = "Top2Box_PI",
    iv_cols: List[str] = None,
    reference_cell: int = 1,
) -> Dict[str, Any]:
    """
    Run logistic regression predicting Top-2-Box Purchase Intent.

    Args:
        df: DataFrame with survey data
        dv_col: Binary DV column (default: Top2Box_PI)
        iv_cols: IVs to include (default: Q8 attribute importance ratings)
        reference_cell: ClaimCell reference group (default: 1 = Low Sugar)

    Returns:
        Dictionary with coefficients, odds ratios, p_values, conf_int,
        pseudo_r_squared, and model_summary.
    """
    if iv_cols is None:
        iv_cols = [f"Q8_AttrImportance_{i}" for i in range(1, 8)]

    data, dummy_cols = _build_regression_data(df, dv_col, iv_cols, reference_cell)

    predictor_cols = iv_cols + dummy_cols
    X = sm.add_constant(data[predictor_cols])
    y = data[dv_col]

    model = sm.Logit(y, X).fit(disp=0)

    conf_int = model.conf_int()
    odds_ratios = np.exp(model.params)
    ci_or_low = np.exp(conf_int[0])
    ci_or_high = np.exp(conf_int[1])

    results = {
        "model_type": "Logit",
        "n_obs": int(model.nobs),
        "pseudo_r_squared": round(float(model.prsquared), 4),
        "llr_pvalue": round(float(model.llr_pvalue), 4),
        "coefficients": {
            name: round(float(val), 4)
            for name, val in model.params.items()
        },
        "odds_ratios": {
            name: round(float(val), 4)
            for name, val in odds_ratios.items()
        },
        "std_errors": {
            name: round(float(val), 4)
            for name, val in model.bse.items()
        },
        "p_values": {
            name: round(float(val), 4)
            for name, val in model.pvalues.items()
        },
        "conf_int_low": {
            name: round(float(conf_int.loc[name, 0]), 4)
            for name in conf_int.index
        },
        "conf_int_high": {
            name: round(float(conf_int.loc[name, 1]), 4)
            for name in conf_int.index
        },
        "or_ci_low": {
            name: round(float(ci_or_low[name]), 4)
            for name in ci_or_low.index
        },
        "or_ci_high": {
            name: round(float(ci_or_high[name]), 4)
            for name in ci_or_high.index
        },
        "model_summary": model.summary(),
        "predictor_cols": predictor_cols,
        "reference_cell": reference_cell,
        "error": None,
    }
    return results
