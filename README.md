# 🍦 Breyers Better For You Survey Dashboard

Interactive marketing research dashboard for analyzing the Breyers "Better For You" ice cream concept survey.

## Course Information
- **Course:** MKTG6051 - Marketing Research
- **Sample Size:** N = 169 respondents
- **Concepts Tested:** 3 (Low Sugar, High Protein, Both)

## Features

### 📊 Executive Overview
- Sample demographics and distribution
- Concept cell allocation
- Purchase frequency breakdown

### 🎯 Concept Performance
- Appeal and Purchase Intent comparisons across concepts
- Pairwise T-tests with p-values and 95% Confidence Intervals
- Interactive concept selector for comparisons

### 📈 Driver Analysis
- Linear Regression (DV: Purchase Intent Q12)
- Logistic Regression (DV: Top 2 Box Purchase Intent)
- Dynamic dummy coding with selectable reference group
- Full coefficient tables with R², p-values, and 95% CI
- Automatic handling of single-concept filter scenarios

### 📋 Crosstabs
- Chi-square tests of independence
- Customizable row/column variable selection
- Observed and expected frequency tables

### 🔗 Correlation Analysis
- Pearson correlation matrix for attribute importance (Q8)
- Significance indicators (* p<.05, ** p<.01, *** p<.001)
- Interactive heatmap visualization

### 💰 Price Sensitivity
- Purchase likelihood across 5 price points ($3.99 - $7.99)
- Optional breakout by concept cell

### 📁 Raw Data
- Searchable, filterable data table
- Human-readable labels (not raw codes)
- CSV export functionality

## Statistical Rigor

All analyses adhere to academic statistical standards:
- **P-values** displayed for all hypothesis tests
- **95% Confidence Intervals** for T-tests and regression coefficients
- **R²** (or Pseudo R²) for all regression models
- **Scale footnotes** on all visualizations
- **1-2 decimal formatting** for all statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/dweebzxx/breyers-survey-dashboard.git
cd breyers-survey-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## Deployment

This app is configured for Streamlit Community Cloud deployment.

1. Fork this repository
2. Connect to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Deploy from your forked repository

## Project Structure

```
breyers-survey-dashboard/
├── app.py                    # Main application entry point
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
├── data/
│   ├── breyers-survey-data-cleaned.csv
│   └── breyers-survey-key.md
└── src/
    ├── data_loader.py        # Data loading and preprocessing
    ├── label_mappings.py     # Value label dictionaries
    ├── stats/
    │   ├── ttest_utils.py
    │   ├── regression_utils.py
    │   ├── chisquare_utils.py
    │   └── correlation_utils.py
    └── tabs/
        ├── tab_overview.py
        ├── tab_concept.py
        ├── tab_regression.py
        ├── tab_crosstabs.py
        ├── tab_correlation.py
        ├── tab_price.py
        └── tab_rawdata.py
```

## Acknowledgments

**Dashboard architecture, data processing, and Streamlit code generation were assisted by an AI coding agent.**

This project was developed for MKTG6051 - Marketing Research in compliance with course AI usage policies.

## License

Academic use only - MKTG6051 Course Project
