"""
Tab 6: Price Sensitivity
Line chart of mean likelihood across 5 price points, with optional concept breakout.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.label_mappings import PRICE_POINTS, SCALE_FOOTNOTES


# Price columns in order
PRICE_COLS = list(PRICE_POINTS.keys())
PRICE_DISPLAY = list(PRICE_POINTS.values())

CONCEPT_SHORT = {
    "with low or zero added sugar": "Low Sugar",
    "with higher protein": "High Protein",
    "with higher protein and low or zero added sugar": "Both",
}


def render(df: pd.DataFrame, question_text: dict = None) -> None:
    """Render the Price Sensitivity tab."""
    st.header("Price Sensitivity")

    breakout = st.checkbox(
        "Break out by Concept Cell",
        value=False,
        key="price_breakout",
    )

    st.divider()

    missing = [c for c in PRICE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing price columns: {missing}")
        return

    if breakout:
        # Build long-format data with concept labels
        records = []
        for concept_label, group in df.groupby("ConceptLabel"):
            short = CONCEPT_SHORT.get(concept_label, concept_label)
            for col, display in zip(PRICE_COLS, PRICE_DISPLAY):
                mean_val = group[col].mean()
                records.append(
                    {"Price": display, "Mean Likelihood": mean_val, "Concept": short}
                )
        plot_df = pd.DataFrame(records)

        fig = px.line(
            plot_df,
            x="Price",
            y="Mean Likelihood",
            color="Concept",
            markers=True,
            title="Mean Purchase Likelihood by Price Point and Concept Cell",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        # Overall means
        means = [df[col].mean() for col in PRICE_COLS]
        plot_df = pd.DataFrame({"Price": PRICE_DISPLAY, "Mean Likelihood": means})

        fig = px.line(
            plot_df,
            x="Price",
            y="Mean Likelihood",
            markers=True,
            title="Mean Purchase Likelihood by Price Point (Overall)",
            color_discrete_sequence=["#4C78A8"],
        )

    fig.update_layout(
        yaxis_range=[1, 5],
        xaxis_title="Price Point",
        yaxis_title="Mean Likelihood (1–5)",
    )
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(SCALE_FOOTNOTES.get("Q17_Price", "Scale: 1 = Very unlikely, 5 = Very likely"))
    st.caption("Q17a–Q17e: How likely would you be to purchase at each price point?")
