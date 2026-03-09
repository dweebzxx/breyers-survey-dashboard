"""
Tab 3: Driver Analysis
Linear and Logistic Regression panels with dynamic ClaimCell dummy coding.
"""

import streamlit as st
import pandas as pd
from src.stats.regression_utils import (
    run_linear_regression,
    run_logistic_regression,
    CLAIM_CELL_LABELS,
)
from src.label_mappings import ATTR_IMPORTANCE


# Map attribute column names to human-readable labels
ATTR_LABELS = {
    f"Q8_AttrImportance_{i}": ATTR_IMPORTANCE[i]
    for i in range(1, 8)
}


def _coeff_table(results: dict, label_map: dict, is_logit: bool = False) -> pd.DataFrame:
    """Build a styled coefficient DataFrame for display."""
    rows = []
    for name in results["coefficients"]:
        display_name = label_map.get(name, name)
        p = results["p_values"][name]
        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"

        row = {
            "Predictor": display_name,
            "Coeff (β)": results["coefficients"][name],
            "Std Error": results["std_errors"][name],
            "p-value": p,
            "Sig.": stars,
            "CI Low (95%)": results["conf_int_low"][name],
            "CI High (95%)": results["conf_int_high"][name],
        }
        if is_logit:
            row["Odds Ratio"] = results["odds_ratios"][name]
            row["OR CI Low"] = results["or_ci_low"][name]
            row["OR CI High"] = results["or_ci_high"][name]
        rows.append(row)

    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> None:
    """Render the Driver Analysis tab."""
    st.header("Driver Analysis")

    # ── Reference group selector ────────────────────────────────────────────
    st.subheader("ClaimCell Reference Group")
    ref_label_options = {v: k for k, v in CLAIM_CELL_LABELS.items()}
    ref_choice = st.selectbox(
        "Select reference group for ClaimCell dummy coding",
        options=list(CLAIM_CELL_LABELS.values()),
        index=0,
        key="regression_ref_group",
    )
    reference_cell = ref_label_options[ref_choice]

    st.divider()

    # Build a combined label map for display
    label_map = {**ATTR_LABELS, "const": "Intercept"}
    dummy_display = {
        "ClaimCell_LowSugar": "ClaimCell: Low Sugar",
        "ClaimCell_HighProtein": "ClaimCell: High Protein",
        "ClaimCell_Both": "ClaimCell: Both",
    }
    label_map.update(dummy_display)

    # ── Linear Regression Panel ─────────────────────────────────────────────
    st.subheader("Linear Regression (OLS)")
    st.markdown("**Dependent Variable:** Q12 Purchase Intent")

    try:
        ols = run_linear_regression(df, reference_cell=reference_cell)

        col_r2, col_adj, col_f, col_n = st.columns(4)
        col_r2.metric("R²", f"{ols['r_squared']:.4f}")
        col_adj.metric("Adj. R²", f"{ols['adj_r_squared']:.4f}")
        col_f.metric("F-statistic", f"{ols['f_statistic']:.3f}")
        col_n.metric("N", ols["n_obs"])

        ols_df = _coeff_table(ols, label_map, is_logit=False)
        st.dataframe(ols_df, use_container_width=True, hide_index=True)
        st.caption("* p<.05  ** p<.01  *** p<.001")

        with st.expander("Full OLS Model Summary"):
            st.text(str(ols["model_summary"]))

    except Exception as e:
        st.error(f"OLS regression failed: {e}")

    st.divider()

    # ── Logistic Regression Panel ───────────────────────────────────────────
    st.subheader("Logistic Regression (Logit)")
    st.markdown("**Dependent Variable:** Top 2 Box Purchase Intent (binary)")

    try:
        logit = run_logistic_regression(df, reference_cell=reference_cell)

        col_pr2, col_llr, col_n2 = st.columns(3)
        col_pr2.metric("Pseudo R² (McFadden)", f"{logit['pseudo_r_squared']:.4f}")
        col_llr.metric("LLR p-value", f"{logit['llr_pvalue']:.4f}")
        col_n2.metric("N", logit["n_obs"])

        logit_df = _coeff_table(logit, label_map, is_logit=True)
        st.dataframe(logit_df, use_container_width=True, hide_index=True)
        st.caption("* p<.05  ** p<.01  *** p<.001 | Odds Ratio = exp(β)")

        with st.expander("Full Logit Model Summary"):
            st.text(str(logit["model_summary"]))

    except Exception as e:
        st.error(f"Logistic regression failed: {e}")
