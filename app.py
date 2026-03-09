"""
Breyers Better For You Survey Dashboard
Marketing Research Course (MKTG6051)

Main Streamlit application entry point.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Breyers Survey Dashboard",
    page_icon="🍦",
    layout="wide"
)

st.title("🍦 Breyers Better For You Survey Dashboard")
st.markdown("**Marketing Research Course (MKTG6051)**")

# Load data
try:
    from src.data_loader import load_data, filter_data
    from src.label_mappings import DIET_FOCUS, AGE

    df_full, question_text = load_data()

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)
    st.stop()

# ── Sidebar: Global Filters ─────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Global Filters")

    # Concept Cell filter
    concept_options = sorted(df_full["ConceptLabel"].dropna().unique().tolist())
    short_labels = {
        "with low or zero added sugar": "Low Sugar",
        "with higher protein": "High Protein",
        "with higher protein and low or zero added sugar": "Both",
    }
    concept_display = [short_labels.get(c, c) for c in concept_options]
    selected_concept_display = st.multiselect(
        "Concept Cell",
        options=concept_display,
        default=concept_display,
        key="filter_concept",
    )
    reverse_short = {v: k for k, v in short_labels.items()}
    selected_concepts = [reverse_short.get(d, d) for d in selected_concept_display]

    # Diet Focus filter
    diet_options = sorted(df_full["Q21_DietFocus"].dropna().unique().tolist())
    diet_display = {k: v for k, v in DIET_FOCUS.items() if k in diet_options}
    selected_diet_display = st.multiselect(
        "Diet Focus",
        options=list(diet_display.values()),
        default=list(diet_display.values()),
        key="filter_diet",
    )
    reverse_diet = {v: k for k, v in diet_display.items()}
    selected_diet = [reverse_diet[d] for d in selected_diet_display if d in reverse_diet]

    # Age filter
    age_options = sorted(df_full["Q23_Age"].dropna().unique().tolist())
    age_display = {k: v for k, v in AGE.items() if k in age_options}
    selected_age_display = st.multiselect(
        "Age Group",
        options=list(age_display.values()),
        default=list(age_display.values()),
        key="filter_age",
    )
    reverse_age = {v: k for k, v in age_display.items()}
    selected_age = [reverse_age[d] for d in selected_age_display if d in reverse_age]

    st.divider()
    st.caption(f"Total respondents (unfiltered): N = {len(df_full)}")

# Apply filters
df = filter_data(
    df_full,
    concept_labels=selected_concepts if len(selected_concepts) < len(concept_options) else None,
    diet_focus=selected_diet if len(selected_diet) < len(diet_options) else None,
    age_groups=selected_age if len(selected_age) < len(age_options) else None,
)

st.caption(f"**Filtered sample: N = {len(df)}**")

if len(df) == 0:
    st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
    st.stop()

# ── 7-Tab Navigation ─────────────────────────────────────────────────────────
from src.tabs import (
    tab_overview,
    tab_concept,
    tab_regression,
    tab_crosstabs,
    tab_correlation,
    tab_price,
    tab_rawdata,
)

tabs = st.tabs([
    "1️⃣ Executive Overview",
    "2️⃣ Concept Performance",
    "3️⃣ Driver Analysis",
    "4️⃣ Crosstabs",
    "5️⃣ Correlation Analysis",
    "6️⃣ Price Sensitivity",
    "7️⃣ Raw Data",
])

with tabs[0]:
    tab_overview.render(df)

with tabs[1]:
    tab_concept.render(df)

with tabs[2]:
    tab_regression.render(df)

with tabs[3]:
    tab_crosstabs.render(df)

with tabs[4]:
    tab_correlation.render(df)

with tabs[5]:
    tab_price.render(df)

with tabs[6]:
    tab_rawdata.render(df)
