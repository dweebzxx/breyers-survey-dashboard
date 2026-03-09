import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, pearsonr
import statsmodels.api as sm

def perform_chi_square(df, col1, col2):
    """Chi-square test of independence."""
    clean_df = df.dropna(subset=[col1, col2])
    n = len(clean_df)
    if n < 5:
        return {"error": f"Insufficient data (n={n}) for Chi-square."}

    contingency_table = pd.crosstab(clean_df[col1], clean_df[col2])

    # Check if table is valid
    if contingency_table.size == 0 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return {"error": "Variables must have at least two categories present in the filtered data."}

    chi2, p, dof, _ = chi2_contingency(contingency_table)

    return {
        "n": n,
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "variables": f"{col1} vs {col2}"
    }

def perform_ttest(df, num_col, cat_col, group1, group2):
    """Independent samples t-test."""
    clean_df = df.dropna(subset=[num_col, cat_col])
    g1_data = clean_df[clean_df[cat_col] == group1][num_col]
    g2_data = clean_df[clean_df[cat_col] == group2][num_col]

    n1, n2 = len(g1_data), len(g2_data)
    if n1 < 2 or n2 < 2:
         return {"error": f"Insufficient data in groups (n1={n1}, n2={n2}) for T-test."}

    t_stat, p = ttest_ind(g1_data, g2_data, equal_var=False)

    return {
        "n": n1 + n2,
        "group1": group1,
        "group2": group2,
        "n1": n1,
        "n2": n2,
        "t_stat": t_stat,
        "p_value": p,
        "mean1": g1_data.mean(),
        "mean2": g2_data.mean(),
        "variables": f"{num_col} grouped by {cat_col}"
    }

def perform_correlation(df, col1, col2):
    """Pearson correlation coefficient."""
    clean_df = df.dropna(subset=[col1, col2])
    n = len(clean_df)
    if n < 3:
        return {"error": f"Insufficient data (n={n}) for Correlation."}

    # Ensure numeric
    try:
        x = pd.to_numeric(clean_df[col1])
        y = pd.to_numeric(clean_df[col2])
    except Exception:
        return {"error": "Both variables must be numeric for Correlation."}

    # Constant check
    if x.nunique() == 1 or y.nunique() == 1:
        return {"error": "One or both variables are constant; correlation cannot be computed."}

    r, p = pearsonr(x, y)

    return {
        "n": n,
        "r": r,
        "p_value": p,
        "variables": f"{col1} vs {col2}"
    }

def perform_linear_regression(df, target, predictors):
    """Linear regression using statsmodels."""
    cols = [target] + predictors
    clean_df = df.dropna(subset=cols)
    n = len(clean_df)

    if n < len(predictors) + 2:
        return {"error": f"Insufficient data (n={n}) for Linear Regression with {len(predictors)} predictors."}

    try:
        y = pd.to_numeric(clean_df[target])
        X = clean_df[predictors].apply(pd.to_numeric)
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        return {
            "n": n,
            "r_squared": model.rsquared,
            "f_pvalue": model.f_pvalue,
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "variables": f"Target: {target}, Predictors: {', '.join(predictors)}"
        }
    except Exception as e:
        return {"error": f"Error fitting model: {str(e)}"}

def perform_logistic_regression(df, target, target_positive_value, predictors):
    """Logistic regression using statsmodels."""
    cols = [target] + predictors
    clean_df = df.dropna(subset=cols).copy()
    n = len(clean_df)

    if n < len(predictors) + 2:
        return {"error": f"Insufficient data (n={n}) for Logistic Regression."}

    try:
        # Create binary target
        clean_df['__binary_target'] = (clean_df[target] == target_positive_value).astype(int)

        y = clean_df['__binary_target']
        X = clean_df[predictors].apply(pd.to_numeric)
        X = sm.add_constant(X)

        # Check for single class
        if y.nunique() < 2:
            return {"error": "Target variable has only one class in the filtered data."}

        model = sm.Logit(y, X).fit(disp=0)
        return {
            "n": n,
            "pseudo_r_squared": model.prsquared,
            "params": model.params.to_dict(),
            "pvalues": model.pvalues.to_dict(),
            "variables": f"Target: {target} (Positive='{target_positive_value}'), Predictors: {', '.join(predictors)}"
        }
    except Exception as e:
        return {"error": f"Error fitting model: {str(e)}. (Consider perfect separation or singular matrix issues)."}
