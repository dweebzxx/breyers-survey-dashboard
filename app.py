"""Breyers Survey Analysis Dashboard — main Dash application."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html
from scipy.stats import pearsonr

from data import (
    ATTR_LABELS,
    Q6_BRAND_COL_MAP,
    Q6_BRAND_MAP,
    apply_labels,
    get_label_maps,
    load_data,
)
from stats import (
    chi_square_test,
    correlation_analysis,
    linear_regression,
    logistic_regression,
    t_test_independent,
)

# ── Bootstrap colour palette mapped to ClaimCell labels ──────────────────────
CLAIM_COLORS = {"High Protein": "#2c7bb6", "Low Sugar": "#d7191c", "Both": "#1a9641"}
CLAIM_ORDER = ["High Protein", "Low Sugar", "Both"]
HIGH_INTENT_THRESHOLD = 4  # Q12_PurchaseIntent >= this value is classified as high intent

# ── Load data once at startup ─────────────────────────────────────────────────
DF_RAW = load_data()
LABEL_MAPS = get_label_maps()

# ── Dash app ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Breyers Survey Dashboard",
)
server = app.server  # expose for gunicorn


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _opts(label_map: dict) -> list:
    """Build dropdown options list from a label mapping dict."""
    return [{"label": v, "value": k} for k, v in sorted(label_map.items())]


def _empty_fig(msg: str = "Insufficient data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            dict(
                text=msg,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font={"size": 16, "color": "#888"},
            )
        ],
    )
    return fig


def _apply_filters(
    df: pd.DataFrame,
    claim_cell=None,
    age=None,
    income=None,
    diet=None,
    seeking=None,
    household=None,
    freq=None,
) -> pd.DataFrame:
    """Apply global sidebar filters to df and return the filtered copy."""
    mask = pd.Series(True, index=df.index)
    pairs = [
        ("ClaimCell", claim_cell),
        ("Q23_Age", age),
        ("Q24_Income", income),
        ("Q21_DietFocus", diet),
        ("Q10_ActiveSeeking", seeking),
        ("Q22_HouseholdType", household),
        ("Q4_PurchaseFreq", freq),
    ]
    for col, val in pairs:
        if val and val != "all":
            mask &= df[col] == int(val)
    return df[mask].copy()


def _pct_count_bar(
    df: pd.DataFrame,
    col: str,
    title: str,
    label_map: dict = None,
    color_col: str = None,
    color_map: dict = None,
) -> go.Figure:
    """Bar chart of value counts with % labels for a single column."""
    s = df[col].dropna()
    n_total = len(s)
    if n_total == 0:
        return _empty_fig()
    counts = s.value_counts().sort_index()
    pcts = (counts / n_total * 100).round(1)
    x_labels = (
        [label_map.get(int(k), str(k)) for k in counts.index]
        if label_map else [str(k) for k in counts.index]
    )
    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=counts.values,
            text=[f"{p}%" for p in pcts.values],
            textposition="auto",
            marker_color=(
                [color_map.get(lb, "#888") for lb in x_labels]
                if color_map else "#3498db"
            ),
        )
    )
    fig.update_layout(
        title=f"{title} (n={n_total})",
        xaxis_title=col,
        yaxis_title="Count",
        template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig


def _grouped_bar_by_claim(
    df: pd.DataFrame,
    col: str,
    title: str,
    label_map: dict = None,
) -> go.Figure:
    """Grouped bar: col distribution broken out by ClaimCell."""
    sub = df[["ClaimCell_Label", col]].dropna()
    n_total = len(sub)
    if n_total < 2:
        return _empty_fig()

    all_vals = sorted(sub[col].unique())
    fig = go.Figure()
    for claim in CLAIM_ORDER:
        grp = sub[sub["ClaimCell_Label"] == claim]
        if grp.empty:
            continue
        cnts = grp[col].value_counts()
        y_vals = [cnts.get(v, 0) for v in all_vals]
        x_labs = (
            [label_map.get(int(v), str(v)) for v in all_vals]
            if label_map else [str(v) for v in all_vals]
        )
        fig.add_trace(
            go.Bar(
                name=claim,
                x=x_labs,
                y=y_vals,
                marker_color=CLAIM_COLORS.get(claim, "#888"),
            )
        )
    fig.update_layout(
        title=f"{title} (n={n_total})",
        barmode="group",
        xaxis_title=col,
        yaxis_title="Count",
        template="plotly_white",
        legend_title="Claim Cell",
        margin=dict(t=50, b=40),
    )
    return fig


def _stat_card(content: str, title: str = "") -> dbc.Card:
    """Render a statistical result as a monospace card."""
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted mb-2") if title else None,
            html.Pre(content, style={"fontSize": "0.75rem", "whiteSpace": "pre-wrap"}),
        ]),
        className="mb-3 border-0 bg-light",
    )


def _fmt_chi2(result: dict, var1: str, var2: str) -> str:
    if "error" in result:
        return f"Chi-square {var1} × {var2}: {result['error']}"
    return (
        f"Chi-square: {var1} × {var2}\n"
        f"  χ²({result['dof']}) = {result['stat']:.3f},  "
        f"p = {result['p_value']:.4f},  n = {result['n']}"
    )


def _fmt_ttest(result: dict, g_var: str, v_var: str) -> str:
    if "error" in result:
        return f"t-test {g_var} ~ {v_var}: {result['error']}"
    lines = [f"t-test (Welch's): {v_var} by {g_var}"]
    for g in result["groups"]:
        lines.append(
            f"  {g_var}={g}: M={result['means'][g]:.2f}, SD={result['stds'][g]:.2f}, n={result['ns'][g]}"
        )
    lines.append(f"  t = {result['t_stat']:.3f},  p = {result['p_value']:.4f}")
    return "\n".join(lines)


def _fmt_regression(result: dict, outcome: str, predictors: list) -> str:
    if "error" in result:
        return f"Regression {outcome}: {result['error']}"
    lines = [
        f"OLS: {outcome} ~ {' + '.join(predictors)}",
        f"  n = {result['n']},  R² = {result['r_squared']:.3f},  adj-R² = {result['adj_r_squared']:.3f}",
        f"  {'Variable':<30} {'Coef':>8} {'p':>8}",
        "  " + "-" * 50,
    ]
    for var, coef in result["coef"].items():
        pval = result["p_values"].get(var, np.nan)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        lines.append(f"  {var:<30} {coef:>8.3f} {pval:>8.4f} {sig}")
    return "\n".join(lines)


def _fmt_logistic(result: dict, outcome: str, predictors: list) -> str:
    if "error" in result:
        return f"Logistic {outcome}: {result['error']}"
    lines = [
        f"Logit: {outcome} ~ {' + '.join(predictors)}",
        f"  n = {result['n']},  McFadden pseudo-R² = {result['pseudo_r2']:.3f}",
        f"  {'Variable':<30} {'Coef':>8} {'OR':>8} {'p':>8}",
        "  " + "-" * 58,
    ]
    for var, coef in result["coef"].items():
        pval = result["p_values"].get(var, np.nan)
        OR = result["odds_ratios"].get(var, np.nan)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        lines.append(f"  {var:<30} {coef:>8.3f} {OR:>8.3f} {pval:>8.4f} {sig}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> dbc.Col:
    """Left sidebar with global filters."""
    def _dd(fid, label, opts, multi=False):
        return html.Div([
            html.Label(label, className="fw-semibold small mb-1"),
            dcc.Dropdown(
                id=fid,
                options=[{"label": "All", "value": "all"}] + opts,
                value="all",
                clearable=False,
                multi=multi,
                className="mb-2",
            ),
        ])

    return dbc.Col(
        [
            html.H5("Filters", className="mb-3 fw-bold"),
            _dd("f-claim", "Claim Cell", _opts(LABEL_MAPS["ClaimCell"])),
            _dd("f-age", "Age", _opts(LABEL_MAPS["Q23_Age"])),
            _dd("f-income", "Income", _opts(LABEL_MAPS["Q24_Income"])),
            _dd("f-diet", "Diet Focus", _opts(LABEL_MAPS["Q21_DietFocus"])),
            _dd("f-seeking", "Active Seeking", _opts(LABEL_MAPS["Q10_ActiveSeeking"])),
            _dd("f-household", "Household Type", _opts(LABEL_MAPS["Q22_HouseholdType"])),
            _dd("f-freq", "Purchase Freq.", _opts(LABEL_MAPS["Q4_PurchaseFreq"])),
            html.Hr(),
            html.Div(id="filter-n", className="text-muted small"),
        ],
        width=3,
        className="bg-light border-end p-3",
        style={"minHeight": "100vh"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab layouts
# ─────────────────────────────────────────────────────────────────────────────

def _tab_overview():
    return dbc.Tab(
        label="Overview",
        tab_id="tab-overview",
        children=[
            dbc.Row(id="overview-kpis", className="mb-3 mt-3 g-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="ov-appeal-bar"), md=4),
                dbc.Col(dcc.Graph(id="ov-intent-bar"), md=4),
                dbc.Col(dcc.Graph(id="ov-interest-bar"), md=4),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("Mean Scores by Claim Cell", className="fw-bold"),
                    html.Div(id="ov-summary-table"),
                ], md=12),
            ]),
        ],
    )


def _tab_concept():
    return dbc.Tab(
        label="Concept Performance",
        tab_id="tab-concept",
        children=[
            dbc.Row(dbc.Col(
                html.Div([
                    html.Label("Filter by Claim Cell:", className="fw-semibold small me-2"),
                    dcc.Dropdown(
                        id="cp-claim-filter",
                        options=[{"label": "All", "value": "all"}]
                        + [{"label": v, "value": k} for k, v in LABEL_MAPS["ClaimCell"].items()],
                        value="all",
                        clearable=False,
                        style={"width": "200px", "display": "inline-block"},
                    ),
                ], className="d-flex align-items-center mt-3 mb-3"),
                width=12,
            )),
            dbc.Row([
                dbc.Col(dcc.Graph(id="cp-appeal"), md=6),
                dbc.Col(dcc.Graph(id="cp-intent"), md=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="cp-replacement"), md=6),
                dbc.Col(dcc.Graph(id="cp-replaced"), md=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="cp-interest"), md=6),
                dbc.Col(dcc.Graph(id="cp-location"), md=6),
            ], className="mb-2"),
            html.Hr(),
            dbc.Row([
                dbc.Col([html.Div(id="cp-chi2-stats")], md=6),
                dbc.Col([html.Div(id="cp-ttest-stats")], md=6),
            ]),
        ],
    )


def _tab_attributes():
    return dbc.Tab(
        label="Attributes & Tradeoff",
        tab_id="tab-attributes",
        children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="at-attr-bar"), md=6),
                dbc.Col(dcc.Graph(id="at-tradeoff-bar"), md=3),
                dbc.Col(dcc.Graph(id="at-seeking-bar"), md=3),
            ], className="mt-3 mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="at-corr-heatmap"), md=6),
                dbc.Col(html.Div(id="at-corr-table"), md=6),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(html.Div(id="at-reg-output"), md=12),
            ]),
        ],
    )


def _tab_price():
    return dbc.Tab(
        label="Price Sensitivity",
        tab_id="tab-price",
        children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="pr-price-line"), md=6),
                dbc.Col(dcc.Graph(id="pr-tooexp-hist"), md=6),
            ], className="mt-3 mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="pr-club-bar"), md=6),
                dbc.Col(dcc.Graph(id="pr-online-bar"), md=6),
            ], className="mb-2"),
            html.Hr(),
            dbc.Row([
                dbc.Col(html.Div(id="pr-logistic-output"), md=6),
                dbc.Col(html.Div(id="pr-reg-output"), md=6),
            ]),
        ],
    )


def _tab_demo():
    return dbc.Tab(
        label="Demographics",
        tab_id="tab-demo",
        children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="dm-age-bar"), md=4),
                dbc.Col(dcc.Graph(id="dm-income-bar"), md=4),
                dbc.Col(dcc.Graph(id="dm-role-pie"), md=4),
            ], className="mt-3 mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="dm-diet-bar"), md=4),
                dbc.Col(dcc.Graph(id="dm-household-bar"), md=4),
                dbc.Col(dcc.Graph(id="dm-freq-bar"), md=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="dm-heatmap-age"), md=6),
                dbc.Col(dcc.Graph(id="dm-heatmap-diet"), md=6),
            ], className="mb-2"),
            html.Hr(),
            dbc.Row([
                dbc.Col(html.Div(id="dm-chi2-age"), md=6),
                dbc.Col(html.Div(id="dm-chi2-diet"), md=6),
            ]),
        ],
    )


def _tab_rawdata():
    return dbc.Tab(
        label="Raw Data",
        tab_id="tab-rawdata",
        children=[
            dbc.Row(dbc.Col([
                dbc.Row([
                    dbc.Col(html.Div(id="rd-rowcount", className="text-muted small mt-3"), md=9),
                    dbc.Col(
                        dbc.Button(
                            "Export CSV",
                            id="rd-export-btn",
                            color="secondary",
                            size="sm",
                            className="mt-3",
                        ),
                        md=3, className="text-end",
                    ),
                ]),
                dcc.Download(id="rd-download"),
                html.Div(id="rd-table-container", className="mt-2"),
            ], width=12)),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# App layout
# ─────────────────────────────────────────────────────────────────────────────

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H3("Breyers Survey Analysis Dashboard", className="mb-0 fw-bold text-white"),
                    html.Small("Interactive survey results explorer", className="text-white-50"),
                ]),
                className="py-3 px-4 bg-primary",
            ),
            className="mb-0",
        ),
        dbc.Row(
            [
                _sidebar(),
                dbc.Col(
                    dbc.Tabs(
                        [
                            _tab_overview(),
                            _tab_concept(),
                            _tab_attributes(),
                            _tab_price(),
                            _tab_demo(),
                            _tab_rawdata(),
                        ],
                        id="main-tabs",
                        active_tab="tab-overview",
                        className="mt-0",
                    ),
                    width=9,
                    className="ps-3",
                ),
            ],
            className="g-0",
        ),
    ],
    fluid=True,
    className="px-0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Filter-n indicator
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("filter-n", "children"),
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
    ],
)
def update_filter_n(claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)
    return f"Filtered n: {len(dff)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Overview callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("overview-kpis", "children"),
        Output("ov-appeal-bar", "figure"),
        Output("ov-intent-bar", "figure"),
        Output("ov-interest-bar", "figure"),
        Output("ov-summary-table", "children"),
    ],
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
    ],
)
def update_overview(claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)
    n = len(dff)

    # KPI cards
    mean_appeal = dff["Q11_Appeal"].mean()
    mean_intent = dff["Q12_PurchaseIntent"].mean()
    mean_interest = dff["Q14_InterestComparison"].mean()

    claim_counts = dff["ClaimCell_Label"].value_counts()

    def _kpi(label, value, color="primary"):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5(value, className=f"card-title text-{color} mb-0"),
                    html.Small(label, className="text-muted"),
                ]),
                className="text-center shadow-sm",
            ),
            md=2,
        )

    kpis = [_kpi("Total n", str(n))]
    for label in CLAIM_ORDER:
        cnt = claim_counts.get(label, 0)
        pct = f"{cnt/n*100:.0f}%" if n > 0 else "—"
        kpis.append(_kpi(label, f"{cnt} ({pct})", "secondary"))
    kpis.append(_kpi("Mean Appeal", f"{mean_appeal:.2f}" if n > 0 else "—", "info"))
    kpis.append(_kpi("Mean Intent", f"{mean_intent:.2f}" if n > 0 else "—", "success"))
    kpis.append(_kpi("Mean Interest Comp.", f"{mean_interest:.2f}" if n > 0 else "—", "warning"))

    appeal_map = LABEL_MAPS["Q11_Appeal"]
    intent_map = LABEL_MAPS["Q12_PurchaseIntent"]
    interest_map = LABEL_MAPS["Q14_InterestComparison"]

    fig_appeal = _grouped_bar_by_claim(dff, "Q11_Appeal", "Q11 Appeal Rating Distribution by Claim Cell", appeal_map)
    fig_intent = _grouped_bar_by_claim(dff, "Q12_PurchaseIntent", "Q12 Purchase Intent Distribution by Claim Cell", intent_map)
    fig_interest = _grouped_bar_by_claim(dff, "Q14_InterestComparison", "Q14 Interest Comparison Distribution by Claim Cell", interest_map)

    # Summary table
    summary_rows = []
    for claim in CLAIM_ORDER:
        grp = dff[dff["ClaimCell_Label"] == claim]
        ng = len(grp)
        summary_rows.append({
            "Claim Cell": claim,
            "n": ng,
            "Mean Appeal": f"{grp['Q11_Appeal'].mean():.2f}" if ng > 0 else "—",
            "Mean Purchase Intent": f"{grp['Q12_PurchaseIntent'].mean():.2f}" if ng > 0 else "—",
            "Mean Interest Comparison": f"{grp['Q14_InterestComparison'].mean():.2f}" if ng > 0 else "—",
        })
    smry_df = pd.DataFrame(summary_rows)
    table = dbc.Table.from_dataframe(smry_df, striped=True, bordered=True, hover=True, size="sm")

    return kpis, fig_appeal, fig_intent, fig_interest, table


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Concept Performance callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("cp-appeal", "figure"),
        Output("cp-intent", "figure"),
        Output("cp-replacement", "figure"),
        Output("cp-replaced", "figure"),
        Output("cp-interest", "figure"),
        Output("cp-location", "figure"),
        Output("cp-chi2-stats", "children"),
        Output("cp-ttest-stats", "children"),
    ],
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
        Input("cp-claim-filter", "value"),
    ],
)
def update_concept(claim, age, income, diet, seeking, household, freq, local_claim):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)

    # Apply local ClaimCell filter
    if local_claim and local_claim != "all":
        dff = dff[dff["ClaimCell"] == int(local_claim)].copy()

    n = len(dff)

    def _bar(col, title, lmap):
        return _pct_count_bar(dff, col, title, label_map=lmap)

    fig_appeal = _bar("Q11_Appeal", "Q11 Appeal Rating Distribution", LABEL_MAPS["Q11_Appeal"])
    fig_intent = _bar("Q12_PurchaseIntent", "Q12 Purchase Intent Distribution", LABEL_MAPS["Q12_PurchaseIntent"])
    fig_repl = _bar("Q13_Replacement", "Q13 Replacement Behaviour Distribution", LABEL_MAPS["Q13_Replacement"])
    fig_replaced = _bar("Q13A_WhatReplaced", "Q13A What Would Be Replaced Distribution", LABEL_MAPS["Q13A_WhatReplaced"])
    fig_interest = _bar("Q14_InterestComparison", "Q14 Interest Comparison Distribution", LABEL_MAPS["Q14_InterestComparison"])
    fig_location = _bar("Q16_PurchaseLocation", "Q16 Preferred Purchase Location Distribution", LABEL_MAPS["Q16_PurchaseLocation"])

    # Stats — use the full globally-filtered df (not local-claim-filtered)
    dff_global = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)

    chi2 = chi_square_test(dff_global, "ClaimCell", "Q11_Appeal")
    chi2_text = _fmt_chi2(chi2, "ClaimCell", "Q11_Appeal")

    # Pairwise t-tests: 1v2, 1v3, 2v3
    pairs = [(1, 2), (1, 3), (2, 3)]
    ttest_lines = ["Pairwise Welch's t-tests: Q12_PurchaseIntent by ClaimCell\n"]
    for g1, g2 in pairs:
        res = t_test_independent(dff_global, "ClaimCell", "Q12_PurchaseIntent", groups=[g1, g2])
        lbl1 = LABEL_MAPS["ClaimCell"].get(g1, str(g1))
        lbl2 = LABEL_MAPS["ClaimCell"].get(g2, str(g2))
        if "error" in res:
            ttest_lines.append(f"  {lbl1} vs {lbl2}: {res['error']}")
        else:
            ttest_lines.append(
                f"  {lbl1} (M={res['means'][str(g1)]:.2f}, n={res['ns'][str(g1)]})"
                f"  vs  {lbl2} (M={res['means'][str(g2)]:.2f}, n={res['ns'][str(g2)]})"
                f"  →  t={res['t_stat']:.3f}, p={res['p_value']:.4f}"
            )

    chi2_card = _stat_card(chi2_text, "Chi-Square: ClaimCell × Q11_Appeal")
    ttest_card = _stat_card("\n".join(ttest_lines), "Pairwise t-tests: Q12_PurchaseIntent")

    return (
        fig_appeal, fig_intent, fig_repl, fig_replaced,
        fig_interest, fig_location,
        chi2_card, ttest_card,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Attributes & Tradeoff callbacks
# ─────────────────────────────────────────────────────────────────────────────

ATTR_COLS = [f"Q8_AttrImportance_{i}" for i in range(1, 8)]
PRICE_COLS = ["Q17a_Price399", "Q17b_Price499", "Q17c_Price599", "Q17d_Price699", "Q17e_Price799"]


@app.callback(
    [
        Output("at-attr-bar", "figure"),
        Output("at-tradeoff-bar", "figure"),
        Output("at-seeking-bar", "figure"),
        Output("at-corr-heatmap", "figure"),
        Output("at-corr-table", "children"),
        Output("at-reg-output", "children"),
    ],
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
    ],
)
def update_attributes(claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)
    n = len(dff)

    # Horizontal bar — mean attribute importance
    if n == 0:
        fig_attr = _empty_fig()
    else:
        means = {ATTR_LABELS[c]: dff[c].mean() for c in ATTR_COLS if c in dff.columns}
        means_s = pd.Series(means).sort_values()
        fig_attr = go.Figure(
            go.Bar(
                x=means_s.values,
                y=means_s.index,
                orientation="h",
                marker_color="#2980b9",
                text=[f"{v:.2f}" for v in means_s.values],
                textposition="auto",
            )
        )
        fig_attr.update_layout(
            title=f"Mean Attribute Importance Scores (n={n})",
            xaxis_title="Mean (1–5)",
            yaxis_title="",
            xaxis_range=[1, 5],
            template="plotly_white",
            margin=dict(t=50, b=40),
        )

    fig_tradeoff = _pct_count_bar(dff, "Q9_Tradeoff", "Q9 Tradeoff Preference Distribution", LABEL_MAPS["Q9_Tradeoff"])
    fig_seeking = _pct_count_bar(dff, "Q10_ActiveSeeking", "Q10 Active Seeking Distribution", LABEL_MAPS["Q10_ActiveSeeking"])

    # Correlation heatmap of Q8 attributes
    corr_res = correlation_analysis(dff, ATTR_COLS)
    if "error" in corr_res:
        fig_heatmap = _empty_fig(corr_res["error"])
    else:
        short_labels = [ATTR_LABELS[c] for c in ATTR_COLS]
        cmat = corr_res["corr_matrix"].values
        fig_heatmap = go.Figure(
            go.Heatmap(
                z=cmat,
                x=short_labels,
                y=short_labels,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in cmat],
                texttemplate="%{text}",
            )
        )
        fig_heatmap.update_layout(
            title=f"Q8 Attribute Importance Correlation Matrix (n={corr_res['n']})",
            template="plotly_white",
            margin=dict(t=50, b=40),
        )

    # Correlation table: each attribute vs Q11, Q12
    target_cols = ["Q11_Appeal", "Q12_PurchaseIntent"]
    corr_rows = []
    for acol in ATTR_COLS:
        row = {"Attribute": ATTR_LABELS[acol]}
        for tcol in target_cols:
            sub = dff[[acol, tcol]].dropna()
            if len(sub) >= 3:
                r, p = pearsonr(sub[acol], sub[tcol])
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                row[f"r ({tcol})"] = f"{r:.3f}{sig}"
                row[f"p ({tcol})"] = f"{p:.4f}"
            else:
                row[f"r ({tcol})"] = "—"
                row[f"p ({tcol})"] = "—"
        corr_rows.append(row)
    corr_tbl_df = pd.DataFrame(corr_rows)
    corr_tbl = html.Div([
        html.H6(f"Attribute Correlations with Q11 & Q12 (n≈{n})", className="fw-bold"),
        dbc.Table.from_dataframe(corr_tbl_df, striped=True, bordered=True, hover=True, size="sm"),
    ])

    # Linear regression Q12 ~ all Q8 attributes
    reg_res = linear_regression(dff, "Q12_PurchaseIntent", ATTR_COLS)
    reg_card = _stat_card(_fmt_regression(reg_res, "Q12_PurchaseIntent", ATTR_COLS),
                          "OLS: Q12_PurchaseIntent ~ Q8 Attributes")

    return fig_attr, fig_tradeoff, fig_seeking, fig_heatmap, corr_tbl, reg_card


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Price Sensitivity callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("pr-price-line", "figure"),
        Output("pr-tooexp-hist", "figure"),
        Output("pr-club-bar", "figure"),
        Output("pr-online-bar", "figure"),
        Output("pr-logistic-output", "children"),
        Output("pr-reg-output", "children"),
    ],
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
    ],
)
def update_price(claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)
    n = len(dff)

    price_points = ["$3.99", "$4.99", "$5.99", "$6.99", "$7.99"]
    price_cols_ordered = ["Q17a_Price399", "Q17b_Price499", "Q17c_Price599", "Q17d_Price699", "Q17e_Price799"]

    # Line chart: mean likelihood by ClaimCell
    if n == 0:
        fig_line = _empty_fig()
    else:
        fig_line = go.Figure()
        for claim_label in CLAIM_ORDER:
            grp = dff[dff["ClaimCell_Label"] == claim_label]
            if grp.empty:
                continue
            means = [grp[c].mean() for c in price_cols_ordered]
            fig_line.add_trace(
                go.Scatter(
                    x=price_points,
                    y=means,
                    mode="lines+markers",
                    name=claim_label,
                    line=dict(color=CLAIM_COLORS.get(claim_label, "#888")),
                )
            )
        fig_line.update_layout(
            title=f"Mean Purchase Likelihood by Price Point and Claim Cell (n={n})",
            xaxis_title="Price Point",
            yaxis_title="Mean Likelihood (1–5)",
            yaxis_range=[1, 5],
            template="plotly_white",
            legend_title="Claim Cell",
            margin=dict(t=50, b=40),
        )

    # Histogram: Q18 price too expensive
    q18 = dff["Q18_PriceTooExpensive"].dropna()
    if len(q18) == 0:
        fig_hist = _empty_fig()
    else:
        median_val = q18.median()
        cap_val = q18.quantile(0.99)
        fig_hist = go.Figure(
            go.Histogram(
                x=q18,
                nbinsx=20,
                marker_color="#e67e22",
                opacity=0.8,
            )
        )
        fig_hist.add_vline(
            x=median_val, line_dash="dash", line_color="red",
            annotation_text=f"Median: ${median_val:.2f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            title=f"Q18 Price Too Expensive Distribution (n={len(q18)}, capped at 99th pct=${cap_val:.2f})",
            xaxis_title="Price ($)",
            yaxis_title="Count",
            template="plotly_white",
            margin=dict(t=50, b=40),
        )

    fig_club = _pct_count_bar(dff, "Q19_ClubStore4Pack", "Q19 Club Store 4-Pack Interest Distribution", LABEL_MAPS["Q19_ClubStore4Pack"])
    fig_online = _pct_count_bar(dff, "Q20_OnlineDelivery", "Q20 Online Delivery Likelihood Distribution", LABEL_MAPS["Q20_OnlineDelivery"])

    # Logistic regression: high intent (Q12>=4) ~ Q17 price columns
    dff2 = dff.copy()
    dff2["HighIntent"] = (dff2["Q12_PurchaseIntent"] >= HIGH_INTENT_THRESHOLD).astype(float)
    logit_res = logistic_regression(dff2, "HighIntent", price_cols_ordered)
    logit_card = _stat_card(
        _fmt_logistic(logit_res, "HighIntent (Q12>=4)", price_cols_ordered),
        "Logit: High Purchase Intent ~ Price Sensitivity Scores",
    )

    # Linear regression: Q17a ~ Q23_Age + Q24_Income + Q21_DietFocus
    reg_res = linear_regression(dff, "Q17a_Price399", ["Q23_Age", "Q24_Income", "Q21_DietFocus"])
    reg_card = _stat_card(
        _fmt_regression(reg_res, "Q17a_Price399", ["Q23_Age", "Q24_Income", "Q21_DietFocus"]),
        "OLS: Q17a_Price399 ~ Q23_Age + Q24_Income + Q21_DietFocus",
    )

    return fig_line, fig_hist, fig_club, fig_online, logit_card, reg_card


# ─────────────────────────────────────────────────────────────────────────────
# Tab 5 — Demographics callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _crosstab_heatmap(df, row_var, col_var, row_map, col_map, title):
    sub = df[[row_var, col_var]].dropna()
    n = len(sub)
    if n < 2:
        return _empty_fig()
    ct = pd.crosstab(sub[row_var], sub[col_var], normalize="index") * 100
    ct.index = [row_map.get(int(i), str(i)) for i in ct.index]
    ct.columns = [col_map.get(int(c), str(c)) for c in ct.columns]
    fig = go.Figure(
        go.Heatmap(
            z=ct.values,
            x=ct.columns.tolist(),
            y=ct.index.tolist(),
            colorscale="Blues",
            text=[[f"{v:.1f}%" for v in row] for row in ct.values],
            texttemplate="%{text}",
            colorbar=dict(title="% within row"),
        )
    )
    fig.update_layout(
        title=f"{title} (n={n})",
        template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig


@app.callback(
    [
        Output("dm-age-bar", "figure"),
        Output("dm-income-bar", "figure"),
        Output("dm-role-pie", "figure"),
        Output("dm-diet-bar", "figure"),
        Output("dm-household-bar", "figure"),
        Output("dm-freq-bar", "figure"),
        Output("dm-heatmap-age", "figure"),
        Output("dm-heatmap-diet", "figure"),
        Output("dm-chi2-age", "children"),
        Output("dm-chi2-diet", "children"),
    ],
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
    ],
)
def update_demographics(claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)

    fig_age = _pct_count_bar(dff, "Q23_Age", "Q23 Age Distribution", LABEL_MAPS["Q23_Age"])
    fig_income = _pct_count_bar(dff, "Q24_Income", "Q24 Income Distribution", LABEL_MAPS["Q24_Income"])
    fig_diet = _pct_count_bar(dff, "Q21_DietFocus", "Q21 Diet Focus Distribution", LABEL_MAPS["Q21_DietFocus"])
    fig_household = _pct_count_bar(dff, "Q22_HouseholdType", "Q22 Household Type Distribution", LABEL_MAPS["Q22_HouseholdType"])
    fig_freq = _pct_count_bar(dff, "Q4_PurchaseFreq", "Q4 Purchase Frequency Distribution", LABEL_MAPS["Q4_PurchaseFreq"])

    # Pie: Q3_DecisionRole
    role_counts = dff["Q3_DecisionRole"].dropna().value_counts()
    n_role = role_counts.sum()
    if n_role == 0:
        fig_role = _empty_fig()
    else:
        labels = [LABEL_MAPS["Q3_DecisionRole"].get(int(k), str(k)) for k in role_counts.index]
        fig_role = go.Figure(
            go.Pie(
                labels=labels,
                values=role_counts.values,
                hole=0.3,
                textinfo="label+percent",
            )
        )
        fig_role.update_layout(
            title=f"Q3 Decision Role (n={n_role})",
            template="plotly_white",
            margin=dict(t=50, b=20),
        )

    # Heatmaps
    fig_hm_age = _crosstab_heatmap(
        dff, "ClaimCell", "Q23_Age",
        LABEL_MAPS["ClaimCell"], LABEL_MAPS["Q23_Age"],
        "ClaimCell × Q23 Age (% within Claim Cell)",
    )
    fig_hm_diet = _crosstab_heatmap(
        dff, "ClaimCell", "Q21_DietFocus",
        LABEL_MAPS["ClaimCell"], LABEL_MAPS["Q21_DietFocus"],
        "ClaimCell × Q21 Diet Focus (% within Claim Cell)",
    )

    # Chi-square stats
    chi2_age = chi_square_test(dff, "ClaimCell", "Q23_Age")
    chi2_diet = chi_square_test(dff, "ClaimCell", "Q21_DietFocus")

    chi2_age_card = _stat_card(_fmt_chi2(chi2_age, "ClaimCell", "Q23_Age"), "Chi-Square: ClaimCell × Q23_Age")
    chi2_diet_card = _stat_card(_fmt_chi2(chi2_diet, "ClaimCell", "Q21_DietFocus"), "Chi-Square: ClaimCell × Q21_DietFocus")

    return (
        fig_age, fig_income, fig_role,
        fig_diet, fig_household, fig_freq,
        fig_hm_age, fig_hm_diet,
        chi2_age_card, chi2_diet_card,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 6 — Raw Data callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    [
        Output("rd-table-container", "children"),
        Output("rd-rowcount", "children"),
    ],
    [
        Input("f-claim", "value"),
        Input("f-age", "value"),
        Input("f-income", "value"),
        Input("f-diet", "value"),
        Input("f-seeking", "value"),
        Input("f-household", "value"),
        Input("f-freq", "value"),
    ],
)
def update_rawdata(claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)

    # Build display columns: keep original survey + brand binary
    brand_cols = list(Q6_BRAND_COL_MAP.values())
    exclude_cols = set(brand_cols) | {"ClaimCell_Label"}
    display_cols = [c for c in DF_RAW.columns if c not in exclude_cols]
    display_df = dff[display_cols].copy()

    # Tooltip headers mapping col → question label
    tooltip_map = {
        "Q1_Consent": "Consent to participate",
        "Q2_PurchaseRecent": "Purchased ice cream last 6 months?",
        "Q3_DecisionRole": "Decision role in household",
        "Q4_PurchaseFreq": "Purchase frequency",
        "Q5_UsualChannel": "Usual purchase channel",
        "Q6_BrandsBought": "Brands purchased (multi-select)",
        "Q7_BrandMostOften": "Brand purchased most often",
        "Q8_AttrImportance_1": "Attr importance: Taste",
        "Q8_AttrImportance_2": "Attr importance: Price",
        "Q8_AttrImportance_3": "Attr importance: Brand reputation",
        "Q8_AttrImportance_4": "Attr importance: Low/zero sugar",
        "Q8_AttrImportance_5": "Attr importance: High protein",
        "Q8_AttrImportance_6": "Attr importance: Clean ingredients",
        "Q8_AttrImportance_7": "Attr importance: Low calorie",
        "Q9_Tradeoff": "Tradeoff preference",
        "Q10_ActiveSeeking": "Actively seeking claim type",
        "Q11_Appeal": "Concept appeal rating",
        "Q12_PurchaseIntent": "Purchase intent",
        "Q13_Replacement": "Replacement behaviour",
        "Q13A_WhatReplaced": "What would be replaced",
        "Q14_InterestComparison": "Interest vs usual ice cream",
        "Q15_AttentionCheck_1": "Attention check",
        "Q16_PurchaseLocation": "Preferred purchase location",
        "Q17a_Price399": "Likelihood at $3.99",
        "Q17b_Price499": "Likelihood at $4.99",
        "Q17c_Price599": "Likelihood at $5.99",
        "Q17d_Price699": "Likelihood at $6.99",
        "Q17e_Price799": "Likelihood at $7.99",
        "Q18_PriceTooExpensive": "Price considered too expensive ($)",
        "Q19_ClubStore4Pack": "Club store 4-pack interest",
        "Q20_OnlineDelivery": "Online delivery likelihood",
        "Q21_DietFocus": "Diet focus",
        "Q22_HouseholdType": "Household type",
        "Q23_Age": "Age group",
        "Q24_Income": "Household income",
        "ClaimCell": "Claim cell assignment (1=High Protein, 2=Low Sugar, 3=Both)",
        "ConceptLabel": "Concept label text",
    }

    columns = [
        {
            "name": c,
            "id": c,
            "deletable": False,
            "selectable": False,
        }
        for c in display_df.columns
    ]

    tooltip_header = {
        c: {"value": tooltip_map.get(c, c), "use_with": "header"}
        for c in display_df.columns
    }

    table = dash_table.DataTable(
        data=display_df.astype(str).replace("nan", "").to_dict("records"),
        columns=columns,
        page_size=20,
        sort_action="native",
        filter_action="native",
        tooltip_header=tooltip_header,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": "0.75rem", "padding": "4px 8px", "maxWidth": "150px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#fdfdfd"}
        ],
    )

    row_count = f"Showing {len(dff)} rows"
    return table, row_count


@app.callback(
    Output("rd-download", "data"),
    Input("rd-export-btn", "n_clicks"),
    [
        State("f-claim", "value"),
        State("f-age", "value"),
        State("f-income", "value"),
        State("f-diet", "value"),
        State("f-seeking", "value"),
        State("f-household", "value"),
        State("f-freq", "value"),
    ],
    prevent_initial_call=True,
)
def export_csv(n_clicks, claim, age, income, diet, seeking, household, freq):
    dff = _apply_filters(DF_RAW, claim, age, income, diet, seeking, household, freq)
    return dcc.send_data_frame(dff.to_csv, "breyers_survey_filtered.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
