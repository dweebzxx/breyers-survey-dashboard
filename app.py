import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_data
from stats_utils import (perform_chi_square, perform_ttest, perform_correlation,
                         perform_linear_regression, perform_logistic_regression)

st.set_page_config(page_title="Breyers Better For You Survey Dashboard", layout="wide")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

st.title("Breyers Better For You Survey Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Global Filters")

# Concept Exposure
concept_options = df['ClaimCell'].dropna().unique().tolist()
selected_concepts = st.sidebar.multiselect("Concept Exposure", concept_options, default=concept_options)

# Purchase Frequency
freq_options = df['Q4_PurchaseFreq'].dropna().unique().tolist()
selected_freq = st.sidebar.multiselect("Purchase Frequency", freq_options, default=freq_options)

# Age
age_options = sorted(df['Q23_Age'].dropna().unique().tolist())
selected_age = st.sidebar.multiselect("Age", age_options, default=age_options)

# Income
income_options = df['Q24_Income'].dropna().unique().tolist()
selected_income = st.sidebar.multiselect("Income", income_options, default=income_options)

# Diet Focus
diet_options = df['Q21_DietFocus'].dropna().unique().tolist()
selected_diet = st.sidebar.multiselect("Diet Focus", diet_options, default=diet_options)

# Apply filters
filtered_df = df[
    (df['ClaimCell'].isin(selected_concepts)) &
    (df['Q4_PurchaseFreq'].isin(selected_freq)) &
    (df['Q23_Age'].isin(selected_age)) &
    (df['Q24_Income'].isin(selected_income)) &
    (df['Q21_DietFocus'].isin(selected_diet))
]

st.sidebar.markdown("---")
st.sidebar.metric("Sample Size (n)", len(filtered_df))

if len(filtered_df) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# --- Tabs ---
tabs = st.tabs([
    "Overview",
    "Concept Performance",
    "Attributes and Tradeoff",
    "Price Sensitivity",
    "Demographics",
    "Statistical Analysis",
    "Raw Data"
])

def plot_bar(df, col, title, orientation='h'):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'Count']
    fig = px.bar(counts, x='Count' if orientation=='h' else col,
                 y=col if orientation=='h' else 'Count',
                 title=title, orientation=orientation)
    return fig

# To be implemented in later steps
with tabs[0]:
    st.header("Overview")
    st.metric("Total Respondents (Filtered)", len(filtered_df))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_bar(filtered_df, 'ClaimCell', 'Concept Exposure Counts', 'v'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_bar(filtered_df, 'Q4_PurchaseFreq', 'Purchase Frequency Counts', 'h'), use_container_width=True)

with tabs[1]:
    st.header("Concept Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_bar(filtered_df, 'Q11_Appeal', 'Concept Appeal (Q11)'), use_container_width=True)
        st.plotly_chart(plot_bar(filtered_df, 'Q13_Replacement', 'Replacement vs Addition (Q13)'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_bar(filtered_df, 'Q12_PurchaseIntent', 'Purchase Intent (Q12)'), use_container_width=True)
        st.plotly_chart(plot_bar(filtered_df, 'Q14_InterestComparison', 'Interest vs Regular Ice Cream (Q14)'), use_container_width=True)

with tabs[2]:
    st.header("Attributes and Tradeoff")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_bar(filtered_df, 'Q9_Tradeoff', 'Tradeoff Preference (Q9)'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_bar(filtered_df, 'Q10_ActiveSeeking', 'Active Seeking Behavior (Q10)'), use_container_width=True)

    st.subheader("Attribute Importance (Q8)")
    attr_cols = [f'Q8_AttrImportance_{i}' for i in range(1, 8)]
    attr_names = ['Taste', 'Price', 'Brand reputation', 'Low/zero sugar', 'High protein', 'Short/clean ingredient list', 'Low calorie content']

    # Calculate top 2 box (Very + Extremely important)
    top2_scores = []
    for c, name in zip(attr_cols, attr_names):
        if c in filtered_df.columns:
            counts = filtered_df[c].value_counts()
            top2 = counts.get('Very important', 0) + counts.get('Extremely important', 0)
            valid_len = len(filtered_df[c].dropna())
            top2_pct = top2 / valid_len * 100 if valid_len > 0 else 0
            top2_scores.append({'Attribute': name, 'Top 2 Box %': top2_pct})

    if top2_scores:
        top2_df = pd.DataFrame(top2_scores).sort_values('Top 2 Box %', ascending=True)
        fig = px.bar(top2_df, x='Top 2 Box %', y='Attribute', title='Top 2 Box Importance % (Very/Extremely)', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.header("Price Sensitivity")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price Too Expensive (Q18)")
        if 'Q18_PriceTooExpensive' in filtered_df.columns:
            clean_price = filtered_df['Q18_PriceTooExpensive'].dropna()
            if not clean_price.empty:
                fig = px.histogram(clean_price, x='Q18_PriceTooExpensive', nbins=20, title='Distribution of "Too Expensive" Price')
                st.plotly_chart(fig, use_container_width=True)
                st.metric('Median "Too Expensive" Price', f"${clean_price.median():.2f}")
            else:
                st.write("No numeric price data available for current filters.")

    with col2:
        st.subheader("Likelihood to Buy at Price Points (Q17)")
        price_cols = ['Q17a_Price399', 'Q17b_Price499', 'Q17c_Price599', 'Q17d_Price699', 'Q17e_Price799']
        price_labels = ['$3.99', '$4.99', '$5.99', '$6.99', '$7.99']

        top2_price = []
        for c, label in zip(price_cols, price_labels):
            if c in filtered_df.columns:
                counts = filtered_df[c].value_counts()
                top2 = counts.get('Likely', 0) + counts.get('Very likely', 0)
                valid_len = len(filtered_df[c].dropna())
                top2_pct = top2 / valid_len * 100 if valid_len > 0 else 0
                top2_price.append({'Price Point': label, 'Likely/Very Likely %': top2_pct})

        if top2_price:
            p_df = pd.DataFrame(top2_price)
            fig = px.line(p_df, x='Price Point', y='Likely/Very Likely %', markers=True, title='Demand Curve (Top 2 Box)')
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
with tabs[4]:
    st.header("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_bar(filtered_df, 'Q23_Age', 'Age Distribution (Q23)', 'v'), use_container_width=True)
        st.plotly_chart(plot_bar(filtered_df, 'Q21_DietFocus', 'Diet Focus (Q21)', 'h'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_bar(filtered_df, 'Q24_Income', 'Income Distribution (Q24)', 'h'), use_container_width=True)
        st.plotly_chart(plot_bar(filtered_df, 'Q22_HouseholdType', 'Household Type (Q22)', 'h'), use_container_width=True)

with tabs[5]:
    st.header("Statistical Analysis")
    st.write("Statistical outputs generated from the filtered dataset. Assumptions and limitations apply based on the available filtered subset.")

    st.subheader("1. Chi-square Test of Independence")
    st.write("Testing relationship between Concept Exposure (ClaimCell) and Purchase Intent (Q12).")
    chi_res = perform_chi_square(filtered_df, 'ClaimCell', 'Q12_PurchaseIntent')
    if "error" in chi_res:
        st.error(chi_res["error"])
    else:
        st.write(f"**Variables**: {chi_res['variables']}")
        st.write(f"**Sample Size (n)**: {chi_res['n']}")
        st.write(f"**Chi-square statistic**: {chi_res['chi2']:.4f}")
        st.write(f"**p-value**: {chi_res['p_value']:.4e}")
        st.write(f"**Degrees of Freedom**: {chi_res['dof']}")

    st.markdown("---")
    st.subheader("2. Independent Samples T-test")
    st.write("Comparing 'Too Expensive' Price (Q18) between those who actively seek low sugar vs high protein (Q10).")
    group1_label = 'I actively look for low/zero sugar ice cream'
    group2_label = 'I actively look for high-protein ice cream'
    ttest_res = perform_ttest(filtered_df, 'Q18_PriceTooExpensive', 'Q10_ActiveSeeking', group1_label, group2_label)
    if "error" in ttest_res:
        st.error(ttest_res["error"])
    else:
        st.write(f"**Variables**: {ttest_res['variables']}")
        st.write(f"**Sample Size (n)**: {ttest_res['n']} (n1={ttest_res['n1']}, n2={ttest_res['n2']})")
        st.write(f"**Mean '{group1_label}'**: ${ttest_res['mean1']:.2f}")
        st.write(f"**Mean '{group2_label}'**: ${ttest_res['mean2']:.2f}")
        st.write(f"**T-statistic**: {ttest_res['t_stat']:.4f}")
        st.write(f"**p-value**: {ttest_res['p_value']:.4e}")

    st.markdown("---")
    st.subheader("3. Correlation Analysis")
    st.write("Pearson correlation between Purchase Intent (Q12) numeric mapping and Concept Appeal (Q11) numeric mapping.")
    corr_res = perform_correlation(filtered_df, 'Q12_PurchaseIntent_Num', 'Q11_Appeal_Num')
    if "error" in corr_res:
        st.error(corr_res["error"])
    else:
        st.write(f"**Variables**: {corr_res['variables']}")
        st.write(f"**Sample Size (n)**: {corr_res['n']}")
        st.write(f"**Pearson r**: {corr_res['r']:.4f}")
        st.write(f"**p-value**: {corr_res['p_value']:.4e}")

    st.markdown("---")
    st.subheader("4. Linear Regression")
    st.write("Predicting 'Too Expensive' price threshold (Q18) based on Purchase Intent (Q12_Num) and Appeal (Q11_Num).")
    lin_res = perform_linear_regression(filtered_df, 'Q18_PriceTooExpensive', ['Q12_PurchaseIntent_Num', 'Q11_Appeal_Num'])
    if "error" in lin_res:
        st.error(lin_res["error"])
    else:
        st.write(f"**Variables**: {lin_res['variables']}")
        st.write(f"**Sample Size (n)**: {lin_res['n']}")
        st.write(f"**R-squared**: {lin_res['r_squared']:.4f}")
        st.write(f"**F-test p-value**: {lin_res['f_pvalue']:.4e}")

        st.write("**Coefficients**:")
        params_df = pd.DataFrame({
            "Coefficient": lin_res["params"],
            "p-value": lin_res["pvalues"]
        })
        st.dataframe(params_df.style.format("{:.4f}"))

    st.markdown("---")
    st.subheader("5. Logistic Regression")
    st.write("Predicting likelihood of Top Box Purchase Intent ('Definitely would buy') based on Concept Appeal (Q11_Num).")
    log_res = perform_logistic_regression(filtered_df, 'Q12_PurchaseIntent', 'Definitely would buy', ['Q11_Appeal_Num'])
    if "error" in log_res:
        st.error(log_res["error"])
    else:
        st.write(f"**Variables**: {log_res['variables']}")
        st.write(f"**Sample Size (n)**: {log_res['n']}")
        st.write(f"**Pseudo R-squared**: {log_res['pseudo_r_squared']:.4f}")

        st.write("**Coefficients**:")
        lparams_df = pd.DataFrame({
            "Coefficient": log_res["params"],
            "p-value": log_res["pvalues"]
        })
        st.dataframe(lparams_df.style.format("{:.4f}"))

with tabs[6]:
    st.header("Raw Data")
    st.write("This table shows the dataset after applying the global filters.")
    # Convert all columns to string to avoid PyArrow serialization issues
    st.dataframe(filtered_df.astype(str))
