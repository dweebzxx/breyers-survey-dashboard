"""
Tab 2: Concept Performance
Compares Appeal and Purchase Intent across concept cells with pairwise t-test.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from src.label_mappings import SCALE_FOOTNOTES
from src.stats.ttest_utils import run_ttest


# Short labels used in the UI
CONCEPT_SHORT = {
    "with low or zero added sugar": "Low Sugar",
    "with higher protein": "High Protein",
    "with higher protein and low or zero added sugar": "Both",
}


def _concept_mean_bar(df: pd.DataFrame, metric_col: str, title: str) -> None:
    """Render a bar chart of mean metric by concept cell (sorted descending)."""
    means = (
        df.groupby("ConceptLabel")[metric_col]
        .mean()
        .reset_index()
    )
    means.columns = ["ConceptLabel", "Mean"]
    means["Concept"] = means["ConceptLabel"].map(lambda x: CONCEPT_SHORT.get(x, x))
    means = means.sort_values("Mean", ascending=False)

    fig = px.bar(
        means,
        x="Concept",
        y="Mean",
        text=means["Mean"].round(2),
        title=title,
        color="Concept",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 5.5])
    st.plotly_chart(fig, use_container_width=True)


def _stat_check_card(result: dict, group_a: str, group_b: str, metric_label: str) -> None:
    """Render a Stat-Check card with t-test results."""
    if result.get("error"):
        st.warning(f"⚠️ {result['error']}")
        return

    p = result["p_value"]
    sig = "✅ Significant" if p < 0.05 else "❌ Not significant"
    sig_color = "green" if p < 0.05 else "red"

    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:14px; background:#f9f9f9;">
        <b>📊 Stat-Check: {metric_label} — {group_a} vs {group_b}</b>
        <table style="width:100%; margin-top:8px;">
          <tr>
            <td><b>Mean ({group_a})</b></td><td>{result['group1_mean']:.2f} (n={result['n1']})</td>
            <td><b>Mean ({group_b})</b></td><td>{result['group2_mean']:.2f} (n={result['n2']})</td>
          </tr>
          <tr>
            <td><b>t-statistic</b></td><td>{result['t_statistic']:.3f}</td>
            <td><b>p-value</b></td><td>{result['p_value']:.4f}</td>
          </tr>
          <tr>
            <td><b>95% CI (diff)</b></td>
            <td colspan="3">[{result['ci_low']:.3f}, {result['ci_high']:.3f}]</td>
          </tr>
          <tr>
            <td colspan="4"><span style="color:{sig_color}"><b>{sig} (α = .05)</b></span></td>
          </tr>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(df: pd.DataFrame, question_text: dict = None) -> None:
    """Render the Concept Performance tab."""
    st.header("Concept Performance")

    # Build list of available concepts from data
    all_labels = sorted(df["ConceptLabel"].dropna().unique().tolist())
    short_options = [CONCEPT_SHORT.get(lbl, lbl) for lbl in all_labels]

    # ── Pairwise T-test Selector ────────────────────────────────────────────
    st.subheader("Pairwise Comparison Selector")
    col_a, col_b = st.columns(2)

    with col_a:
        group_a_short = st.selectbox(
            "Group A",
            options=short_options,
            index=0,
            key="concept_group_a",
        )
    with col_b:
        default_b_idx = 1 if len(short_options) > 1 else 0
        group_b_short = st.selectbox(
            "Group B",
            options=short_options,
            index=default_b_idx,
            key="concept_group_b",
        )

    # Validate same-group selection
    if group_a_short == group_b_short:
        st.error("⚠️ Please select two different concept cells to compare.")
        # Still render bar charts below
        group_a_label = None
        group_b_label = None
    else:
        reverse_short = {v: k for k, v in CONCEPT_SHORT.items()}
        group_a_label = reverse_short.get(group_a_short, group_a_short)
        group_b_label = reverse_short.get(group_b_short, group_b_short)

    st.divider()

    # ── Appeal (Q11) ────────────────────────────────────────────────────────
    st.subheader("Appeal (Q11)")
    _concept_mean_bar(df, "Q11_Appeal", "Mean Appeal Score by Concept Cell")
    st.caption(SCALE_FOOTNOTES.get("Q11_Appeal", ""))

    if group_a_label and group_b_label:
        result_appeal = run_ttest(df, "Q11_Appeal", "ConceptLabel", group_a_label, group_b_label)
        _stat_check_card(result_appeal, group_a_short, group_b_short, "Appeal")

    st.divider()

    # ── Purchase Intent (Q12) ───────────────────────────────────────────────
    st.subheader("Purchase Intent (Q12)")
    _concept_mean_bar(df, "Q12_PurchaseIntent", "Mean Purchase Intent by Concept Cell")
    st.caption(SCALE_FOOTNOTES.get("Q12_PurchaseIntent", ""))

    if group_a_label and group_b_label:
        result_pi = run_ttest(df, "Q12_PurchaseIntent", "ConceptLabel", group_a_label, group_b_label)
        _stat_check_card(result_pi, group_a_short, group_b_short, "Purchase Intent")
