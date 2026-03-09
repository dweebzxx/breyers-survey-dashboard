"""
Tab 1: Executive Overview
Displays sample size, ClaimCell distribution, and Q4 purchase frequency.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from src.label_mappings import CONCEPT_LABEL, PURCHASE_FREQ, SCALE_FOOTNOTES


def render(df: pd.DataFrame, question_text: dict = None) -> None:
    """Render the Executive Overview tab."""
    st.header("Executive Overview")

    # ── Sample size metric cards ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Respondents", f"N = {len(df)}")
    with col2:
        t2b = df["Top2Box_PI"].mean() * 100 if "Top2Box_PI" in df.columns else 0
        st.metric("Top 2 Box Purchase Intent", f"{t2b:.1f}%")
    with col3:
        n_concepts = df["ConceptLabel"].nunique() if "ConceptLabel" in df.columns else 0
        st.metric("Concept Cells", n_concepts)

    st.divider()

    # ── ClaimCell distribution ──────────────────────────────────────────────
    st.subheader("Concept Cell Distribution")

    if "ClaimCell" in df.columns and "ConceptLabel" in df.columns:
        # Build a count table with short concept labels
        short_labels = {
            "with low or zero added sugar": "Low Sugar",
            "with higher protein": "High Protein",
            "with higher protein and low or zero added sugar": "Both",
        }
        concept_counts = (
            df.groupby("ConceptLabel")
            .size()
            .reset_index(name="Count")
        )
        concept_counts["Concept"] = concept_counts["ConceptLabel"].map(
            lambda x: short_labels.get(x, x)
        )
        concept_counts = concept_counts.sort_values("Count", ascending=False)

        fig_concept = px.bar(
            concept_counts,
            x="Concept",
            y="Count",
            color="Concept",
            text="Count",
            title="Distribution by Concept Cell",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_concept.update_traces(textposition="outside")
        fig_concept.update_layout(showlegend=False, xaxis_title="Concept Cell")
        st.plotly_chart(fig_concept, use_container_width=True)
    else:
        st.warning("ConceptLabel or ClaimCell column not found in data.")

    st.divider()

    # ── Q4 Purchase Frequency ───────────────────────────────────────────────
    st.subheader("Purchase Frequency (Q4)")

    if "Q4_PurchaseFreq" in df.columns:
        freq_counts = (
            df["Q4_PurchaseFreq"]
            .map(PURCHASE_FREQ)
            .value_counts()
            .reset_index()
        )
        freq_counts.columns = ["Purchase Frequency", "Count"]
        freq_counts = freq_counts.sort_values("Count", ascending=False)

        fig_freq = px.bar(
            freq_counts,
            x="Purchase Frequency",
            y="Count",
            text="Count",
            title="How often do you buy ice cream?",
            color_discrete_sequence=["#4C78A8"],
        )
        fig_freq.update_traces(textposition="outside")
        fig_freq.update_layout(xaxis_title="Purchase Frequency")
        st.plotly_chart(fig_freq, use_container_width=True)
        st.caption("Q4: How often do you buy ice cream?")
    else:
        st.warning("Q4_PurchaseFreq column not found in data.")

    # =================================================================
    # ACKNOWLEDGMENTS FOOTER (Required by course AI policy)
    # =================================================================
    st.markdown("---")
    st.caption("**Acknowledgments:** Dashboard architecture, data processing, and Streamlit code generation were assisted by an AI coding agent.")
