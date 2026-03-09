# Breyers Better For You Survey Dashboard

This is an interactive dashboard to visualize the results of the Breyers Better For You survey. The dashboard is built using [Streamlit](https://streamlit.io/) and presents data, statistical outputs, and charts in a neutral and descriptive manner without narrative insights or conclusions.

## Requirements

Ensure you have Python installed, then install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

To start the Streamlit server and run the dashboard locally, execute the following command in the project's root directory:

```bash
streamlit run app.py
```

The dashboard will be available in your browser at `http://localhost:8501`.

## Files

- `app.py`: Main entry point for the Streamlit dashboard.
- `data_loader.py`: Handles loading and preprocessing the raw survey data (`breyers-survey-data-cleaned.csv`).
- `stats_utils.py`: Contains functions for computing statistical analyses like Chi-square, T-test, Correlation, and Regression.
- `requirements.txt`: Dependencies to run the application.

## Data Mapping Notes

Variables and their levels are mapped exactly as described in `breyers-survey-key.md`. Missing or skipped questions are explicitly handled as NaN and generally excluded from metric denominators.

- Categorical values have been mapped to their respective text labels for easier reading on the dashboard.
- Multi-select fields (e.g., `Q6_BrandsBought`) have been expanded/mapped to list corresponding text labels.
- Pricing inputs (`Q18_PriceTooExpensive`) have been forced to numeric values, ignoring invalid text entries.
