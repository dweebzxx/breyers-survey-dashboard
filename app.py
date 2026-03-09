"""
Breyers Better For You Survey Dashboard
Marketing Research Course (MKTG6051)

Main Streamlit application entry point.
"""

import streamlit as st
from src.data_loader import load_data, filter_data
from src.label_mappings import (
    CONCEPT_LABEL, DIET_FOCUS, AGE,
    PURCHASE_FREQ, SCALE_FOOTNOTES
)

# Import all tab modules
from src.tabs.tab_overview import render as render_overview
from src.tabs.tab_concept import render as render_concept
from src.tabs.tab_regression import render as render_regression
from src.tabs.tab_crosstabs import render as render_crosstabs
from src.tabs.tab_correlation import render as render_correlation
from src.tabs.tab_price import render as render_price
from src.tabs.tab_rawdata import render as render_rawdata

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Breyers Survey Dashboard",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DATA LOADING (Cached)
# =============================================================================
@st.cache_data
def get_data():
    return load_data()

df, question_text = get_data()

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================
with st.sidebar:
    st.title("🍦 Filters")
    st.markdown("---")

    # Concept Cell Filter
    st.subheader("Concept Cell")
    concept_options = df['ConceptLabel'].unique().tolist()
    selected_concepts = st.multiselect(
        "Select Concept(s)",
        options=concept_options,
        default=concept_options,
        key="concept_filter",
        help="Filter by product concept shown to respondents"
    )

    # Diet Focus Filter (Q21)
    st.subheader("Diet Focus (Q21)")
    diet_options = {v: k for k, v in DIET_FOCUS.items()}  # Label -> Code
    diet_labels = list(diet_options.keys())
    selected_diet_labels = st.multiselect(
        "Select Diet Focus",
        options=diet_labels,
        default=diet_labels,
        key="diet_filter",
        help="Filter by respondent's dietary focus"
    )
    selected_diet_codes = [diet_options[label] for label in selected_diet_labels]

    # Age Filter (Q23)
    st.subheader("Age Group (Q23)")
    age_options = {v: k for k, v in AGE.items()}  # Label -> Code
    age_labels = list(age_options.keys())
    selected_age_labels = st.multiselect(
        "Select Age Group(s)",
        options=age_labels,
        default=age_labels,
        key="age_filter",
        help="Filter by respondent age group"
    )
    selected_age_codes = [age_options[label] for label in selected_age_labels]

    st.markdown("---")

    # Apply filters
    filtered_df = filter_data(
        df,
        concept_labels=selected_concepts if selected_concepts else None,
        diet_focus=selected_diet_codes if selected_diet_codes else None,
        age_groups=selected_age_codes if selected_age_codes else None
    )

    # Filtered Sample Size
    st.metric("Filtered Sample Size", f"N = {len(filtered_df)}")

    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches current filters")
    elif len(filtered_df) < 30:
        st.warning(f"⚠️ Small sample size (N={len(filtered_df)}). Statistical results may be unreliable.")

    st.markdown("---")
    st.caption("MKTG6051 - Marketing Research")
    st.caption("Breyers Better For You Survey")

# =============================================================================
# MAIN CONTENT - HEADER
# =============================================================================
st.title("🍦 Breyers Better For You Survey Dashboard")
st.markdown("**Marketing Research Course (MKTG6051)**")

# Check for empty data
if len(filtered_df) == 0:
    st.error("No data matches the current filter selection. Please adjust your filters in the sidebar.")
    st.stop()

# =============================================================================
# TAB NAVIGATION
# =============================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Overview",
    "🎯 Concept Performance",
    "📈 Driver Analysis",
    "📋 Crosstabs",
    "🔗 Correlation",
    "💰 Price Sensitivity",
    "📁 Raw Data"
])

# =============================================================================
# RENDER TABS
# =============================================================================
with tab1:
    render_overview(filtered_df, question_text)

with tab2:
    render_concept(filtered_df, question_text)

with tab3:
    render_regression(filtered_df, question_text)

with tab4:
    render_crosstabs(filtered_df, question_text)

with tab5:
    render_correlation(filtered_df, question_text)

with tab6:
    render_price(filtered_df, question_text)

with tab7:
    render_rawdata(filtered_df, question_text)
