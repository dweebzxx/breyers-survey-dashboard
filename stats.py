"""Statistical analysis helpers for the Breyers survey dashboard."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def chi_square_test(df: pd.DataFrame, var1: str, var2: str) -> dict:
    """
    Perform Pearson chi-square test of independence between var1 and var2.

    Returns dict with keys: stat, p_value, dof, n, contingency_table, expected.
    Returns error dict on failure.
    """
    try:
        subset = df[[var1, var2]].dropna()
        n = len(subset)
        if n < 5:
            return {"error": f"Insufficient data (n={n})"}

        ct = pd.crosstab(subset[var1], subset[var2])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {"error": "Contingency table requires at least 2 rows and 2 columns"}

        chi2, p, dof, expected = stats.chi2_contingency(ct)
        return {
            "stat": float(chi2),
            "p_value": float(p),
            "dof": int(dof),
            "n": int(n),
            "contingency_table": ct,
            "expected": pd.DataFrame(expected, index=ct.index, columns=ct.columns),
        }
    except Exception as exc:
        return {"error": str(exc)}


def t_test_independent(
    df: pd.DataFrame, group_var: str, value_var: str, groups=None
) -> dict:
    """
    Two-group independent-samples t-test.

    groups: list of exactly 2 group values; if None the first two unique values are used.
    Returns dict with keys: t_stat, p_value, means, stds, ns, groups.
    Returns error dict on failure.
    """
    try:
        subset = df[[group_var, value_var]].dropna()
        if groups is None:
            groups = sorted(subset[group_var].unique())[:2]
        if len(groups) < 2:
            return {"error": "Need at least 2 groups"}

        g1 = subset.loc[subset[group_var] == groups[0], value_var]
        g2 = subset.loc[subset[group_var] == groups[1], value_var]
        if len(g1) < 2 or len(g2) < 2:
            return {"error": f"Insufficient data in groups: n={len(g1)}, n={len(g2)}"}

        t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "means": {str(groups[0]): float(g1.mean()), str(groups[1]): float(g2.mean())},
            "stds": {str(groups[0]): float(g1.std()), str(groups[1]): float(g2.std())},
            "ns": {str(groups[0]): int(len(g1)), str(groups[1]): int(len(g2))},
            "groups": [str(g) for g in groups],
        }
    except Exception as exc:
        return {"error": str(exc)}


def correlation_analysis(df: pd.DataFrame, vars_list: list) -> dict:
    """
    Compute pairwise Pearson correlations and p-values for vars_list.

    Returns dict with keys: corr_matrix (DataFrame), p_matrix (DataFrame), n.
    Returns error dict on failure.
    """
    try:
        subset = df[vars_list].dropna()
        n = len(subset)
        if n < 3:
            return {"error": f"Insufficient data (n={n})"}

        k = len(vars_list)
        r_data = np.ones((k, k))
        p_data = np.ones((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                r, p = pearsonr(subset.iloc[:, i], subset.iloc[:, j])
                r_data[i, j] = r_data[j, i] = r
                p_data[i, j] = p_data[j, i] = p
        corr_matrix = pd.DataFrame(r_data, index=vars_list, columns=vars_list)
        p_matrix = pd.DataFrame(p_data, index=vars_list, columns=vars_list)
        return {"corr_matrix": corr_matrix, "p_matrix": p_matrix, "n": int(n)}
    except Exception as exc:
        return {"error": str(exc)}


def linear_regression(
    df: pd.DataFrame, outcome_var: str, predictor_vars: list
) -> dict:
    """
    OLS linear regression: outcome_var ~ predictor_vars.

    Returns dict with keys: coef, p_values, r_squared, adj_r_squared, n, model_summary.
    Returns error dict on failure.
    """
    try:
        cols = [outcome_var] + predictor_vars
        subset = df[cols].dropna()
        n = len(subset)
        if n < len(predictor_vars) + 2:
            return {"error": f"Insufficient data (n={n})"}

        X = sm.add_constant(subset[predictor_vars])
        y = subset[outcome_var]
        model = sm.OLS(y, X).fit()
        return {
            "coef": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "n": int(n),
            "model_summary": model.summary().as_text(),
        }
    except Exception as exc:
        return {"error": str(exc)}


def logistic_regression(
    df: pd.DataFrame, outcome_var: str, predictor_vars: list
) -> dict:
    """
    Logistic regression: outcome_var (binary) ~ predictor_vars.

    Uses statsmodels Logit for coefficients / p-values / pseudo-R2.
    Returns dict with keys: coef, p_values, odds_ratios, pseudo_r2, n.
    Returns error dict on failure.
    """
    try:
        cols = [outcome_var] + predictor_vars
        subset = df[cols].dropna()
        n = len(subset)
        unique_y = subset[outcome_var].unique()
        if n < len(predictor_vars) + 5 or len(unique_y) < 2:
            return {"error": f"Insufficient data or non-binary outcome (n={n})"}

        X = sm.add_constant(subset[predictor_vars].astype(float))
        y = subset[outcome_var].astype(float)
        model = sm.Logit(y, X).fit(disp=0)
        odds_ratios = np.exp(model.params).to_dict()
        return {
            "coef": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "odds_ratios": odds_ratios,
            "pseudo_r2": float(model.prsquared),
            "n": int(n),
        }
    except Exception as exc:
        return {"error": str(exc)}
