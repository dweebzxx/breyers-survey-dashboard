# Breyers Survey Analysis Dashboard

An interactive Python Dash dashboard for exploring and analysing the Breyers Better-For-You ice cream concept survey results (n = 169 respondents).

## Prerequisites

- Python 3.9 or later
- pip

## Installation

```bash
git clone <repo-url>
cd breyers-survey-dashboard

pip install -r requirements.txt
```

## Running the App

**Development server (hot-reload):**
```bash
python app.py
```
The dashboard will be available at [http://localhost:8050](http://localhost:8050).

**Production with Gunicorn:**
```bash
gunicorn app:server -b 0.0.0.0:8050 --workers 2
```

## Project Structure

```
breyers-survey-dashboard/
├── app.py                          # Main Dash application
├── data.py                         # Data loading & preprocessing
├── stats.py                        # Statistical analysis helpers
├── requirements.txt
├── breyers-survey-data-cleaned.csv # Survey data (not committed to public repos)
└── README.md
```

## Dashboard Tabs

| Tab | Contents |
|---|---|
| **Overview** | KPI summary cards, grouped bar charts for Appeal / Purchase Intent / Interest Comparison by Claim Cell, mean-score summary table |
| **Concept Performance** | Per-concept bar charts (Appeal, Purchase Intent, Replacement, Location), chi-square and pairwise t-test results |
| **Attributes & Tradeoff** | Mean attribute importance bar chart, Q9 tradeoff, Q10 active seeking, attribute correlation heatmap, correlations with Q11/Q12, OLS regression |
| **Price Sensitivity** | Mean likelihood by price point (line chart), "too expensive" histogram, club-store / online delivery charts, logistic & linear regression |
| **Demographics** | Age, income, diet focus, household type, purchase frequency distributions, decision-role pie, cross-tab heatmaps, chi-square tests |
| **Raw Data** | Full sortable / filterable DataTable with all columns, row count, CSV export |

## Global Filters (sidebar)

All tabs respond to a set of sidebar filters:
- Claim Cell (1 = High Protein, 2 = Low Sugar, 3 = Both)
- Age group (Q23)
- Household income (Q24)
- Diet focus (Q21)
- Active seeking (Q10)
- Household type (Q22)
- Purchase frequency (Q4)

A **Filtered n** counter updates dynamically to show how many respondents match the current filter combination.

## Data Handling Notes

### Two-row header
The CSV contains two header rows:
- **Row 0** — machine-readable column names (used as DataFrame column headers)
- **Row 1** — full question text (skipped via `skiprows=[1]`)

### Multi-select column (Q6_BrandsBought)
Responses are stored as comma-separated integers (e.g. `"1,2,5"`). `data.py` expands these into binary indicator columns:
`Q6_Brand_Breyers`, `Q6_Brand_BenJerrys`, `Q6_Brand_HaloTop`, `Q6_Brand_Enlightened`, `Q6_Brand_Nicks`, `Q6_Brand_StoreBrand`, `Q6_Brand_LocalRegional`.

### Open-ended numeric (Q18_PriceTooExpensive)
Parsed as float; capped at the 99th percentile to reduce outlier distortion.

### Claim Cell assignment
`ClaimCell` values `1`, `2`, `3` map to `High Protein`, `Low Sugar`, and `Both` respectively. Respondents were randomly assigned to one concept cell.

## Statistical Methods

| Function | Method |
|---|---|
| `chi_square_test` | Pearson χ² test of independence (scipy) |
| `t_test_independent` | Welch's independent-samples t-test (scipy) |
| `correlation_analysis` | Pearson r with p-values (scipy) |
| `linear_regression` | OLS (statsmodels) |
| `logistic_regression` | Logit (statsmodels) |

All statistics display variable names, n, test statistic, p-value, and degrees of freedom without interpretation.
