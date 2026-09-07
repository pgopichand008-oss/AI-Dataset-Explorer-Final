"""
views/profile_view.py — Deep Data Profiling & Statistical Explorer.

Provides:
- Schema Explorer (Column attributes, data types, inferred roles)
- Comprehensive Descriptive Statistics (Count, Mean, Median, Std, Var, Min, Max, Range, Q1, Q3, IQR, Skewness, Kurtosis)
- Distribution Lab (Skewness classification & shape analysis)
- Outlier Analysis (IQR bounds lower/upper & outlier counts)
- Missing Value Intelligence (Completeness & null patterns)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st

from components import cards


def render(
    df: pd.DataFrame,
    column_intelligence_df: pd.DataFrame | None = None,
    numerical_summary: pd.DataFrame | None = None,
    categorical_summary_df: pd.DataFrame | None = None,
) -> None:
    cards.section_header(
        "📊 Profile & Descriptive Intelligence",
        "Explore schema definitions, rigorous statistical metrics, distributions, and missingness patterns.",
    )

    if df.empty:
        st.warning("No data available for profiling.")
        return

    # Tabs inside Profile View
    p_tab1, p_tab2, p_tab3, p_tab4, p_tab5 = st.tabs([
        "📋 Schema Explorer",
        "🔢 Descriptive Statistics",
        "📈 Distribution Lab",
        "🎯 Outlier Analysis",
        "❓ Missing Value Intelligence",
    ])

    # ----------------------------------------------------
    # TAB 1: SCHEMA EXPLORER
    # ----------------------------------------------------
    with p_tab1:
        st.markdown("#### Schema & Attribute Roles")
        if column_intelligence_df is not None and not column_intelligence_df.empty:
            st.dataframe(column_intelligence_df, use_container_width=True, hide_index=True)
        else:
            schema_data = []
            for col in df.columns:
                series = df[col]
                schema_data.append({
                    "Attribute": col,
                    "Data Type": str(series.dtype),
                    "Non-Null Count": int(series.count()),
                    "Null Count": int(series.isna().sum()),
                    "Missing %": round((series.isna().mean() * 100), 2),
                    "Unique Count": int(series.nunique(dropna=True)),
                    "Unique %": round((series.nunique(dropna=True) / max(len(df), 1) * 100), 2),
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # TAB 2: DESCRIPTIVE STATISTICS
    # ----------------------------------------------------
    with p_tab2:
        st.markdown("#### Comprehensive Numerical Statistics")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            stats_rows = []
            for col in numeric_cols:
                series = df[col].dropna()
                if series.empty:
                    continue
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                mean_val = float(series.mean())
                median_val = float(series.median())
                std_val = float(series.std()) if len(series) > 1 else 0.0
                var_val = float(series.var()) if len(series) > 1 else 0.0
                min_val = float(series.min())
                max_val = float(series.max())
                rng = max_val - min_val
                skew_val = float(series.skew()) if len(series) > 2 else 0.0
                kurt_val = float(series.kurtosis()) if len(series) > 3 else 0.0

                stats_rows.append({
                    "Attribute": col,
                    "Count": int(len(series)),
                    "Mean": round(mean_val, 4),
                    "Median": round(median_val, 4),
                    "Std Dev": round(std_val, 4),
                    "Variance": round(var_val, 4),
                    "Min": round(min_val, 4),
                    "Max": round(max_val, 4),
                    "Range": round(rng, 4),
                    "Q1 (25%)": round(q1, 4),
                    "Q3 (75%)": round(q3, 4),
                    "IQR": round(iqr, 4),
                    "Skewness": round(skew_val, 4),
                    "Kurtosis": round(kurt_val, 4),
                })
            stats_df = pd.DataFrame(stats_rows)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numerical columns available for descriptive statistics.")

        if categorical_summary_df is not None and not categorical_summary_df.empty:
            st.divider()
            st.markdown("#### Categorical Attribute Summary")
            st.dataframe(categorical_summary_df, use_container_width=True, hide_index=True)

    # ----------------------------------------------------
    # TAB 3: DISTRIBUTION LAB
    # ----------------------------------------------------
    with p_tab3:
        st.markdown("#### Skewness & Shape Classification")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            dist_rows = []
            for col in numeric_cols:
                series = df[col].dropna()
                if series.empty:
                    continue
                skew_val = float(series.skew()) if len(series) > 2 else 0.0
                abs_skew = abs(skew_val)

                if abs_skew < 0.5:
                    classification = "🟢 Symmetric (Normal-like)"
                elif abs_skew < 1.0:
                    classification = "🟡 Moderately Skewed"
                else:
                    classification = "🔴 Highly Skewed / Heavy Tailed"

                kurt_val = float(series.kurtosis()) if len(series) > 3 else 0.0
                if kurt_val > 0.5:
                    tail_desc = "Leptokurtic (Heavy tails)"
                elif kurt_val < -0.5:
                    tail_desc = "Platykurtic (Light tails)"
                else:
                    tail_desc = "Mesokurtic (Normal-like tails)"

                dist_rows.append({
                    "Attribute": col,
                    "Skewness": round(skew_val, 4),
                    "Classification": classification,
                    "Kurtosis": round(kurt_val, 4),
                    "Tail Type": tail_desc,
                })
            dist_df = pd.DataFrame(dist_rows)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            selected_col = st.selectbox("Select Attribute for Distribution Preview", list(numeric_cols), key="dist_lab_col")
            if selected_col:
                series = df[selected_col].dropna()
                c_hist, c_box = st.columns(2)
                with c_hist:
                    st.markdown(f"**Histogram: `{selected_col}`**")
                    st.bar_chart(np.histogram(series, bins=20)[0])
                with c_box:
                    st.markdown(f"**Summary Quantiles: `{selected_col}`**")
                    q_df = pd.DataFrame([{
                        "Min": round(float(series.min()), 3),
                        "25% (Q1)": round(float(series.quantile(0.25)), 3),
                        "50% (Median)": round(float(series.median()), 3),
                        "75% (Q3)": round(float(series.quantile(0.75)), 3),
                        "Max": round(float(series.max()), 3),
                    }])
                    st.dataframe(q_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numerical columns available for distribution analysis.")

    # ----------------------------------------------------
    # TAB 4: OUTLIER ANALYSIS
    # ----------------------------------------------------
    with p_tab4:
        st.markdown("#### IQR Method Outlier Detection")
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            outlier_rows = []
            for col in numeric_cols:
                series = df[col].dropna()
                if series.empty:
                    continue
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                below_cnt = int((series < lower_bound).sum())
                above_cnt = int((series > upper_bound).sum())
                tot_outliers = below_cnt + above_cnt
                outlier_pct = round((tot_outliers / len(series) * 100), 2)

                outlier_rows.append({
                    "Attribute": col,
                    "Q1": round(q1, 4),
                    "Q3": round(q3, 4),
                    "IQR": round(iqr, 4),
                    "Lower Bound (Q1-1.5*IQR)": round(lower_bound, 4),
                    "Upper Bound (Q3+1.5*IQR)": round(upper_bound, 4),
                    "Outliers Below": below_cnt,
                    "Outliers Above": above_cnt,
                    "Total Outliers": tot_outliers,
                    "Outlier %": outlier_pct,
                })
            outlier_df = pd.DataFrame(outlier_rows)
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        else:
            st.info("No numerical columns available for outlier detection.")

    # ----------------------------------------------------
    # TAB 5: MISSING VALUE INTELLIGENCE
    # ----------------------------------------------------
    with p_tab5:
        st.markdown("#### Completeness & Null Distribution")
        tot_cells = len(df) * len(df.columns)
        tot_missing = int(df.isna().sum().sum())
        completeness = round(((tot_cells - tot_missing) / max(tot_cells, 1) * 100), 2)

        m1, m2, m3 = st.columns(3)
        m1.metric("Dataset Completeness", f"{completeness}%")
        m2.metric("Total Missing Cells", f"{tot_missing:,}")
        m3.metric("Columns with Missing Data", f"{(df.isna().sum() > 0).sum()} of {len(df.columns)}")

        st.divider()

        missing_data = []
        for col in df.columns:
            m_cnt = int(df[col].isna().sum())
            if m_cnt > 0:
                missing_data.append({
                    "Attribute": col,
                    "Missing Count": m_cnt,
                    "Missing %": round((m_cnt / len(df) * 100), 2),
                    "Status": "⚠️ High Risk" if (m_cnt / len(df)) > 0.2 else "🟡 Moderate",
                })

        if missing_data:
            st.dataframe(pd.DataFrame(missing_data), use_container_width=True, hide_index=True)
        else:
            cards.finding_card("100% Complete", "No missing values detected in any column.", "success")
