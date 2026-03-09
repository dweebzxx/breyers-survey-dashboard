"""
Tab 5: Correlation Analysis
Pearson correlation heatmap for Q8 attribute importance ratings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.stats.correlation_utils import run_correlation
from src.label_mappings import ATTR_IMPORTANCE


# Attribute column names
ATTR_COLS = [f"Q8_AttrImportance_{i}" for i in range(1, 8)]

# Human-readable axis labels
ATTR_DISPLAY = [ATTR_IMPORTANCE[i] for i in range(1, 8)]


def _sig_stars(p: float) -> str:
    import math
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def render(df: pd.DataFrame) -> None:
    """Render the Correlation Analysis tab."""
    st.header("Correlation Analysis")
    st.markdown("Pearson correlation coefficients for Q8 Attribute Importance ratings.")

    result = run_correlation(df, ATTR_COLS)

    if result.get("error"):
        st.warning(f"⚠️ {result['error']}")
        return

    corr = result["corr_matrix"].copy()
    pvals = result["pvalue_matrix"].copy()

    # Rename axes for display
    col_rename = {col: ATTR_DISPLAY[i] for i, col in enumerate(ATTR_COLS)}
    corr_display = corr.rename(index=col_rename, columns=col_rename)
    pval_display = pvals.rename(index=col_rename, columns=col_rename)

    # ── Heatmap ─────────────────────────────────────────────────────────────
    n = len(ATTR_DISPLAY)

    # Build annotation text: r value + significance stars
    annot_text = []
    for i in range(n):
        row_texts = []
        for j in range(n):
            r_val = corr_display.iloc[i, j]
            p_val = pval_display.iloc[i, j]
            stars = _sig_stars(p_val) if i != j else ""
            row_texts.append(f"{r_val:.2f}{stars}")
        annot_text.append(row_texts)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_display.values,
            x=ATTR_DISPLAY,
            y=ATTR_DISPLAY,
            text=annot_text,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(
        title=f"Pearson Correlation Matrix (N = {result['n_obs']})",
        xaxis_title="Attribute",
        yaxis_title="Attribute",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pearson correlation coefficients. * p<.05, ** p<.01, *** p<.001"
    )

    st.divider()

    # ── Stat-Check: p-value matrix ──────────────────────────────────────────
    st.subheader("📊 Stat-Check: P-Value Matrix")
    with st.expander("Show p-value matrix (expand)"):
        # Format p-values with stars
        import math
        pval_styled = pval_display.copy().astype(object)
        for i in range(n):
            for j in range(n):
                p = pval_display.iloc[i, j]
                if isinstance(p, float) and math.isnan(p):
                    pval_styled.iloc[i, j] = "—"
                else:
                    pval_styled.iloc[i, j] = f"{p:.4f} {_sig_stars(p)}"
        st.dataframe(pval_styled, use_container_width=True)
        st.caption("* p<.05  ** p<.01  *** p<.001")
