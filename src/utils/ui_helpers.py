"""
UI helper functions for consistent styling across the dashboard.
"""

import streamlit as st
from typing import Dict, Optional, Any
from src.label_mappings import SCALE_FOOTNOTES


def render_stat_card(title: str, stats: Dict[str, Any], significant: Optional[bool] = None) -> None:
    """
    Render a consistent Stat-Check card below charts.

    Args:
        title: Card title (e.g., "T-Test Results")
        stats: Dictionary of stat name -> value
        significant: True/False/None for significance indicator
    """
    with st.container():
        st.markdown(f"**📈 {title}**")

        # Create columns for stats
        cols = st.columns(len(stats))
        for i, (name, value) in enumerate(stats.items()):
            with cols[i]:
                if isinstance(value, float):
                    st.metric(name, f"{value:.2f}")
                else:
                    st.metric(name, str(value))

        # Significance indicator
        if significant is not None:
            if significant:
                st.success("✅ Statistically Significant (p < .05)")
            else:
                st.info("ℹ️ Not Statistically Significant (p ≥ .05)")


def add_scale_footnote(variable: str) -> None:
    """
    Add appropriate scale footnote below a chart.

    Args:
        variable: Variable name to look up in SCALE_FOOTNOTES
    """
    if variable in SCALE_FOOTNOTES:
        st.caption(f"_{SCALE_FOOTNOTES[variable]}_")


def format_p_value(p: float) -> str:
    """
    Format p-value with significance stars.

    Args:
        p: P-value

    Returns:
        Formatted string with significance indicator
    """
    if p < 0.001:
        return f"{p:.3f}***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def format_ci(ci_low: float, ci_high: float) -> str:
    """
    Format confidence interval.

    Args:
        ci_low: Lower bound
        ci_high: Upper bound

    Returns:
        Formatted CI string
    """
    return f"[{ci_low:.2f}, {ci_high:.2f}]"
