"""views/quality_view.py — Quality & anomaly command center."""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards


def render(
    quality_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    quality_score: int,
    rows: int,
    columns: int,
    missing_cells: int,
    duplicate_rows: int,
    total_outlier_values: int,
    health: dict | None = None,
    anomalies: dict | None = None,
) -> None:

    cards.section_header(
        "Quality & Anomaly Command Center",
        "Detect missing data, duplicates, anomalies, health dimensions, and structural quality risks.",
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        h_score = health.get("health_score", quality_score) if health else quality_score
        st.metric("Quality & Health Score", f"{h_score}/100")
    with q2:
        pct = (missing_cells / (rows * columns) * 100) if rows * columns else 0
        st.metric("Missing Data", f"{pct:.2f}%")
    with q3:
        pct = (duplicate_rows / rows * 100) if rows else 0
        st.metric("Duplicate Data", f"{pct:.2f}%")
    with q4:
        anom_cnt = anomalies.get("total_anomalies", total_outlier_values) if anomalies else total_outlier_values
        st.metric("Potential Anomalies", f"{anom_cnt:,}")

    st.divider()

    # Health Dimensions Breakdown if available
    if health and isinstance(health, dict) and health.get("dimensions"):
        st.markdown("#### 🛡️ Health Dimensions Assessment")
        dims = health["dimensions"]
        d_cols = st.columns(min(len(dims), 6))
        for col, (d_name, d_val) in zip(d_cols, dims.items()):
            with col:
                st.metric(
                    label=d_name.replace("_", " ").title(),
                    value=f"{d_val.get('score', 0)}/100" if isinstance(d_val, dict) else f"{d_val}/100",
                )
        st.divider()

    st.markdown("#### Quality Findings")
    if not quality_df.empty:
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
    else:
        cards.finding_card("All clear", "No major quality issues detected.", "success")

    st.markdown("#### Numerical Anomalies & Investigation")
    if anomalies and isinstance(anomalies, dict) and anomalies.get("anomalies"):
        anom_items = anomalies.get("anomalies", [])
        for a in anom_items[:5]:
            col_n = a.get("column", "")
            cnt = a.get("anomaly_count", a.get("count", 0))
            expl = a.get("explanation", a.get("description", "Unusual numerical values detected."))
            cards.finding_card(
                title=f"Anomaly in '{col_n}' ({cnt} records)",
                body=expl,
                kind="warning",
            )

    if not anomaly_df.empty:
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    elif not (anomalies and anomalies.get("anomalies")):
        cards.finding_card("All clear", "No numerical anomalies detected.", "success")

