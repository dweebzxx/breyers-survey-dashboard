"""
Tab 7: Raw Data Viewer
Searchable, filterable data table with human-readable labels and CSV download.
"""

import streamlit as st
import pandas as pd
from src.label_mappings import (
    CONCEPT_LABEL,
    PURCHASE_FREQ,
    APPEAL,
    PURCHASE_INTENT,
    DIET_FOCUS,
    HOUSEHOLD_TYPE,
    AGE,
    INCOME,
    REPLACEMENT,
    TRADEOFF,
    ACTIVE_SEEKING,
    PURCHASE_LOCATION,
)


# Columns to display with their human-readable names
DISPLAY_COLUMNS = {
    "ClaimCell": "Concept Cell",
    "ConceptLabel": "Concept Label",
    "Q4_PurchaseFreq": "Purchase Frequency",
    "Q8_AttrImportance_1": "Attr: Taste",
    "Q8_AttrImportance_2": "Attr: Price",
    "Q8_AttrImportance_3": "Attr: Brand Reputation",
    "Q8_AttrImportance_4": "Attr: Low/Zero Sugar",
    "Q8_AttrImportance_5": "Attr: High Protein",
    "Q8_AttrImportance_6": "Attr: Clean Ingredients",
    "Q8_AttrImportance_7": "Attr: Low Calorie",
    "Q9_Tradeoff": "Tradeoff Preference",
    "Q10_ActiveSeeking": "Active Seeking",
    "Q11_Appeal": "Appeal (Q11)",
    "Q12_PurchaseIntent": "Purchase Intent (Q12)",
    "Top2Box_PI": "Top 2 Box PI",
    "Q13_Replacement": "Replacement Behavior",
    "Q16_PurchaseLocation": "Purchase Location",
    "Q17a_Price399": "Likelihood @ $3.99",
    "Q17b_Price499": "Likelihood @ $4.99",
    "Q17c_Price599": "Likelihood @ $5.99",
    "Q17d_Price699": "Likelihood @ $6.99",
    "Q17e_Price799": "Likelihood @ $7.99",
    "Q21_DietFocus": "Diet Focus",
    "Q22_HouseholdType": "Household Type",
    "Q23_Age": "Age Group",
    "Q24_Income": "Income",
}

# Value-label mappings for columns
VALUE_LABEL_MAPS = {
    "ClaimCell": CONCEPT_LABEL,
    "Q4_PurchaseFreq": PURCHASE_FREQ,
    "Q9_Tradeoff": TRADEOFF,
    "Q10_ActiveSeeking": ACTIVE_SEEKING,
    "Q11_Appeal": APPEAL,
    "Q12_PurchaseIntent": PURCHASE_INTENT,
    "Q13_Replacement": REPLACEMENT,
    "Q16_PurchaseLocation": PURCHASE_LOCATION,
    "Q21_DietFocus": DIET_FOCUS,
    "Q22_HouseholdType": HOUSEHOLD_TYPE,
    "Q23_Age": AGE,
}


def render(df: pd.DataFrame) -> None:
    """Render the Raw Data Viewer tab."""
    st.header("Raw Data Viewer")
    st.markdown(f"Showing **{len(df)}** respondents (after filters applied).")

    # Select columns that exist in the dataframe
    available_cols = [c for c in DISPLAY_COLUMNS.keys() if c in df.columns]
    display_df = df[available_cols].copy()

    # Apply human-readable labels
    for col, mapping in VALUE_LABEL_MAPS.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].map(mapping).fillna(display_df[col])

    # Rename columns
    rename_map = {c: DISPLAY_COLUMNS[c] for c in available_cols}
    display_df = display_df.rename(columns=rename_map)

    # Display table
    st.dataframe(display_df, use_container_width=True, height=500)

    # Download button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name="breyers_survey_filtered.csv",
        mime="text/csv",
    )
