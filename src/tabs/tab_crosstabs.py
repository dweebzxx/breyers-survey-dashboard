"""
Tab 4: Crosstabs Tool
Cross-tabulation with chi-square test between two categorical variables.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.stats.chisquare_utils import run_chisquare
from src.label_mappings import (
    DIET_FOCUS,
    REPLACEMENT,
    AGE,
    HOUSEHOLD_TYPE,
    CONCEPT_LABEL,
)


# Available variables for crosstab analysis
CROSSTAB_VARS = {
    "Q21_DietFocus": "Diet Focus (Q21)",
    "Q13_Replacement": "Replacement Behavior (Q13)",
    "Q23_Age": "Age (Q23)",
    "Q22_HouseholdType": "Household Type (Q22)",
    "ClaimCell": "Concept Cell (ClaimCell)",
}

# Label maps for display
VAR_LABEL_MAPS = {
    "Q21_DietFocus": DIET_FOCUS,
    "Q13_Replacement": REPLACEMENT,
    "Q23_Age": AGE,
    "Q22_HouseholdType": HOUSEHOLD_TYPE,
    "ClaimCell": CONCEPT_LABEL,
}


def _apply_labels(series: pd.Series, col: str) -> pd.Series:
    """Map numeric codes to human-readable labels for a variable."""
    label_map = VAR_LABEL_MAPS.get(col)
    if label_map:
        return series.map(label_map).fillna(series.astype(str))
    return series.astype(str)


def render(df: pd.DataFrame, question_text: dict = None) -> None:
    """Render the Crosstabs Tool tab."""
    st.header("Crosstabs Tool")

    var_options = list(CROSSTAB_VARS.keys())
    var_labels = list(CROSSTAB_VARS.values())

    col_row, col_col = st.columns(2)
    with col_row:
        row_idx = st.selectbox(
            "Row Variable",
            options=range(len(var_options)),
            format_func=lambda i: var_labels[i],
            index=0,
            key="crosstab_row",
        )
        row_col = var_options[row_idx]

    with col_col:
        col_idx = st.selectbox(
            "Column Variable",
            options=range(len(var_options)),
            format_func=lambda i: var_labels[i],
            index=1,
            key="crosstab_col",
        )
        col_col_var = var_options[col_idx]

    if row_col == col_col_var:
        st.error("⚠️ Please select two different variables.")
        return

    st.divider()

    # Run chi-square
    result = run_chisquare(df, row_col, col_col_var)

    if result.get("error"):
        st.warning(f"⚠️ {result['error']}")
        return

    # Display observed crosstab with labels
    observed = result["observed_freq"].copy()
    observed.index = _apply_labels(pd.Series(observed.index), row_col).values
    observed.columns = _apply_labels(pd.Series(observed.columns), col_col_var).values

    st.subheader(f"Crosstab: {CROSSTAB_VARS[row_col]} × {CROSSTAB_VARS[col_col_var]}")
    st.dataframe(observed, use_container_width=True)
    st.caption(f"Observed counts. N = {result['n_obs']}")

    st.divider()

    # ── Stat-Check Card ─────────────────────────────────────────────────────
    p = result["p_value"]
    sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
    sig_color = "green" if p < 0.05 else "red"

    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:14px; background:#f9f9f9;">
        <b>📊 Stat-Check: Chi-Square Test of Independence</b>
        <table style="width:100%; margin-top:8px;">
          <tr>
            <td><b>χ² statistic</b></td><td>{result['chi2_statistic']:.3f}</td>
            <td><b>Degrees of Freedom</b></td><td>{result['dof']}</td>
          </tr>
          <tr>
            <td><b>p-value</b></td><td>{result['p_value']:.4f}</td>
            <td><b>N</b></td><td>{result['n_obs']}</td>
          </tr>
          <tr>
            <td colspan="4"><span style="color:{sig_color}"><b>{sig} (α = .05)</b></span></td>
          </tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show Expected Frequencies"):
        expected = result["expected_freq"].copy()
        expected.index = _apply_labels(pd.Series(expected.index), row_col).values
        expected.columns = _apply_labels(pd.Series(expected.columns), col_col_var).values
        st.dataframe(expected, use_container_width=True)
