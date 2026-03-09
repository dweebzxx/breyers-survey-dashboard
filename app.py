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

st.info("""
    **Phase 1 Complete:** Data loader and label mappings are ready.
    
    Phase 2 will add the statistical utilities and tab modules.
""")

# Test data loading
try:
    from src.data_loader import load_data, get_question_text
    from src.label_mappings import CONCEPT_LABEL, PURCHASE_INTENT, TOP2BOX
    
    df, question_text = load_data()
    
    st.success(f"✅ Data loaded successfully! N = {len(df)} respondents")
    
    # Show sample statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Respondents", len(df))
    
    with col2:
        t2b_rate = df['Top2Box_PI'].mean() * 100
        st.metric("Top 2 Box Rate", f"{t2b_rate:.1f}%")
    
    with col3:
        concepts = df['ConceptLabel'].nunique()
        st.metric("Concept Cells", concepts)
    
    # Show concept distribution
    st.subheader("Concept Cell Distribution")
    concept_counts = df['ConceptLabel'].value_counts()
    st.bar_chart(concept_counts)
    
    # Show sample of question text dictionary
    st.subheader("Sample Question Text Mapping")
    sample_questions = {
        "Q11_Appeal": get_question_text(question_text, "Q11_Appeal"),
        "Q12_PurchaseIntent": get_question_text(question_text, "Q12_PurchaseIntent"),
        "Q21_DietFocus": get_question_text(question_text, "Q21_DietFocus")
    }
    for col, text in sample_questions.items():
        st.markdown(f"**{col}:** {text}")
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)
