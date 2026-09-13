"""
views/profile_view.py — Deep Data Profiling & Statistical Explorer.

Provides:
- Schema Explorer (Column attributes, data types, inferred roles)
- AI Data Dictionary (Semantic roles, meanings, quality concerns, analytical/ML usefulness)
- Comprehensive Descriptive Statistics
- Distribution Lab (Skewness classification & shape analysis)
- Outlier Analysis (IQR bounds and outlier counts)
- Missing Value Intelligence (Completeness & null patterns)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

from components import cards


# ============================================================
# GENERIC FIELD EXTRACTION
# ============================================================

def _extract_field(
    d: dict,
    keys: list,
    default: str = "Not available",
) -> str:

    for key in keys:

        if key in d and d[key] is not None:

            value = str(d[key]).strip()

            if value != "":
                return value

    return default


# ============================================================
# MAIN PROFILE VIEW
# ============================================================

def render(
    df: pd.DataFrame,
    column_intelligence_df: pd.DataFrame | None = None,
    numerical_summary: pd.DataFrame | None = None,
    categorical_summary_df: pd.DataFrame | None = None,
    data_dictionary: dict | None = None,
) -> None:

    cards.section_header(
        "📊 Profile & Descriptive Intelligence",
        "Explore schema definitions, AI Data Dictionary, rigorous statistical metrics, distributions, and missingness patterns.",
    )

    # ========================================================
    # EMPTY DATASET CHECK
    # ========================================================

    if df.empty:

        st.warning(
            "No data available for profiling."
        )

        return

    # ========================================================
    # PROFILE TABS
    # ========================================================

    (
        p_tab1,
        p_tab2,
        p_tab3,
        p_tab4,
        p_tab5,
        p_tab6,
    ) = st.tabs(
        [
            "📋 Schema Explorer",
            "🔢 Descriptive Statistics",
            "📈 Distribution Lab",
            "🎯 Outlier Analysis",
            "❓ Missing Value Intelligence",
            "🤖 AI Data Dictionary",
        ]
    )

    # ========================================================
    # TAB 1: SCHEMA EXPLORER
    # ========================================================

    with p_tab1:

        st.markdown(
            "#### Schema & Attribute Roles"
        )

        if (
            column_intelligence_df is not None
            and not column_intelligence_df.empty
        ):

            st.dataframe(
                column_intelligence_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            schema_data = []

            for col in df.columns:

                series = df[col]

                schema_data.append(
                    {
                        "Attribute": col,

                        "Data Type": str(
                            series.dtype
                        ),

                        "Non-Null Count": int(
                            series.count()
                        ),

                        "Null Count": int(
                            series.isna().sum()
                        ),

                        "Missing %": round(
                            series.isna().mean()
                            * 100,
                            2,
                        ),

                        "Unique Count": int(
                            series.nunique(
                                dropna=True
                            )
                        ),

                        "Unique %": round(
                            series.nunique(
                                dropna=True
                            )
                            / max(
                                len(df),
                                1,
                            )
                            * 100,
                            2,
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(schema_data),
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # TAB 2: DESCRIPTIVE STATISTICS
    # ========================================================

    with p_tab2:

        st.markdown(
            "#### Comprehensive Numerical Statistics"
        )

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            stats_rows = []

            for col in numeric_cols:

                series = df[col].dropna()

                if series.empty:
                    continue

                q1 = float(
                    series.quantile(0.25)
                )

                q3 = float(
                    series.quantile(0.75)
                )

                iqr = q3 - q1

                mean_val = float(
                    series.mean()
                )

                median_val = float(
                    series.median()
                )

                std_val = (
                    float(series.std())
                    if len(series) > 1
                    else 0.0
                )

                var_val = (
                    float(series.var())
                    if len(series) > 1
                    else 0.0
                )

                min_val = float(
                    series.min()
                )

                max_val = float(
                    series.max()
                )

                rng = max_val - min_val

                skew_val = (
                    float(series.skew())
                    if len(series) > 2
                    else 0.0
                )

                kurt_val = (
                    float(series.kurtosis())
                    if len(series) > 3
                    else 0.0
                )

                stats_rows.append(
                    {
                        "Attribute": col,
                        "Count": int(
                            len(series)
                        ),
                        "Mean": round(
                            mean_val,
                            4,
                        ),
                        "Median": round(
                            median_val,
                            4,
                        ),
                        "Std Dev": round(
                            std_val,
                            4,
                        ),
                        "Variance": round(
                            var_val,
                            4,
                        ),
                        "Min": round(
                            min_val,
                            4,
                        ),
                        "Max": round(
                            max_val,
                            4,
                        ),
                        "Range": round(
                            rng,
                            4,
                        ),
                        "Q1 (25%)": round(
                            q1,
                            4,
                        ),
                        "Q3 (75%)": round(
                            q3,
                            4,
                        ),
                        "IQR": round(
                            iqr,
                            4,
                        ),
                        "Skewness": round(
                            skew_val,
                            4,
                        ),
                        "Kurtosis": round(
                            kurt_val,
                            4,
                        ),
                    }
                )

            stats_df = pd.DataFrame(
                stats_rows
            )

            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No numerical columns available "
                "for descriptive statistics."
            )

        if (
            categorical_summary_df is not None
            and not categorical_summary_df.empty
        ):

            st.divider()

            st.markdown(
                "#### Categorical Attribute Summary"
            )

            st.dataframe(
                categorical_summary_df,
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # TAB 3: DISTRIBUTION LAB
    # ========================================================

    with p_tab3:

        st.markdown(
            "#### Skewness & Shape Classification"
        )

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            dist_rows = []

            for col in numeric_cols:

                series = df[col].dropna()

                if series.empty:
                    continue

                skew_val = (
                    float(series.skew())
                    if len(series) > 2
                    else 0.0
                )

                abs_skew = abs(
                    skew_val
                )

                if abs_skew < 0.5:

                    classification = (
                        "🟢 Symmetric (Normal-like)"
                    )

                elif abs_skew < 1.0:

                    classification = (
                        "🟡 Moderately Skewed"
                    )

                else:

                    classification = (
                        "🔴 Highly Skewed / Heavy Tailed"
                    )

                kurt_val = (
                    float(series.kurtosis())
                    if len(series) > 3
                    else 0.0
                )

                if kurt_val > 0.5:

                    tail_desc = (
                        "Leptokurtic (Heavy tails)"
                    )

                elif kurt_val < -0.5:

                    tail_desc = (
                        "Platykurtic (Light tails)"
                    )

                else:

                    tail_desc = (
                        "Mesokurtic (Normal-like tails)"
                    )

                dist_rows.append(
                    {
                        "Attribute": col,
                        "Skewness": round(
                            skew_val,
                            4,
                        ),
                        "Classification": classification,
                        "Kurtosis": round(
                            kurt_val,
                            4,
                        ),
                        "Tail Type": tail_desc,
                    }
                )

            dist_df = pd.DataFrame(
                dist_rows
            )

            st.dataframe(
                dist_df,
                use_container_width=True,
                hide_index=True,
            )

            selected_col = st.selectbox(
                "Select Attribute for Distribution Preview",
                list(numeric_cols),
                key="dist_lab_col",
            )

            if selected_col:

                series = df[
                    selected_col
                ].dropna()

                c_hist, c_box = (
                    st.columns(2)
                )

                with c_hist:

                    st.markdown(
                        f"**Histogram: `{selected_col}`**"
                    )

                    st.bar_chart(
                        np.histogram(
                            series,
                            bins=20,
                        )[0]
                    )

                with c_box:

                    st.markdown(
                        f"**Summary Quantiles: `{selected_col}`**"
                    )

                    q_df = pd.DataFrame(
                        [
                            {
                                "Min": round(
                                    float(
                                        series.min()
                                    ),
                                    3,
                                ),

                                "25% (Q1)": round(
                                    float(
                                        series.quantile(
                                            0.25
                                        )
                                    ),
                                    3,
                                ),

                                "50% (Median)": round(
                                    float(
                                        series.median()
                                    ),
                                    3,
                                ),

                                "75% (Q3)": round(
                                    float(
                                        series.quantile(
                                            0.75
                                        )
                                    ),
                                    3,
                                ),

                                "Max": round(
                                    float(
                                        series.max()
                                    ),
                                    3,
                                ),
                            }
                        ]
                    )

                    st.dataframe(
                        q_df,
                        use_container_width=True,
                        hide_index=True,
                    )

        else:

            st.info(
                "No numerical columns available "
                "for distribution analysis."
            )

    # ========================================================
    # TAB 4: OUTLIER ANALYSIS
    # ========================================================

    with p_tab4:

        st.markdown(
            "#### IQR Method Outlier Detection"
        )

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns

        if len(numeric_cols) > 0:

            outlier_rows = []

            for col in numeric_cols:

                series = df[col].dropna()

                if series.empty:
                    continue

                q1 = float(
                    series.quantile(0.25)
                )

                q3 = float(
                    series.quantile(0.75)
                )

                iqr = q3 - q1

                lower_bound = (
                    q1 - 1.5 * iqr
                )

                upper_bound = (
                    q3 + 1.5 * iqr
                )

                below_cnt = int(
                    (
                        series
                        < lower_bound
                    ).sum()
                )

                above_cnt = int(
                    (
                        series
                        > upper_bound
                    ).sum()
                )

                total_outliers = (
                    below_cnt
                    + above_cnt
                )

                outlier_pct = round(
                    total_outliers
                    / len(series)
                    * 100,
                    2,
                )

                outlier_rows.append(
                    {
                        "Attribute": col,
                        "Q1": round(
                            q1,
                            4,
                        ),
                        "Q3": round(
                            q3,
                            4,
                        ),
                        "IQR": round(
                            iqr,
                            4,
                        ),
                        "Lower Bound (Q1-1.5*IQR)": round(
                            lower_bound,
                            4,
                        ),
                        "Upper Bound (Q3+1.5*IQR)": round(
                            upper_bound,
                            4,
                        ),
                        "Outliers Below": below_cnt,
                        "Outliers Above": above_cnt,
                        "Total Outliers": total_outliers,
                        "Outlier %": outlier_pct,
                    }
                )

            outlier_df = pd.DataFrame(
                outlier_rows
            )

            st.dataframe(
                outlier_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "No numerical columns available "
                "for outlier detection."
            )

    # ========================================================
    # TAB 5: MISSING VALUE INTELLIGENCE
    # ========================================================

    with p_tab5:

        st.markdown(
            "#### Completeness & Null Distribution"
        )

        total_cells = (
            len(df)
            * len(df.columns)
        )

        total_missing = int(
            df.isna().sum().sum()
        )

        completeness = round(
            (
                (
                    total_cells
                    - total_missing
                )
                / max(
                    total_cells,
                    1,
                )
                * 100
            ),
            2,
        )

        m1, m2, m3 = (
            st.columns(3)
        )

        m1.metric(
            "Dataset Completeness",
            f"{completeness}%",
        )

        m2.metric(
            "Total Missing Cells",
            f"{total_missing:,}",
        )

        m3.metric(
            "Columns with Missing Data",
            f"{(df.isna().sum() > 0).sum()} "
            f"of {len(df.columns)}",
        )

        st.divider()

        missing_data = []

        for col in df.columns:

            missing_count = int(
                df[col].isna().sum()
            )

            if missing_count > 0:

                missing_data.append(
                    {
                        "Attribute": col,
                        "Missing Count": missing_count,
                        "Missing %": round(
                            missing_count
                            / len(df)
                            * 100,
                            2,
                        ),
                        "Status": (
                            "⚠️ High Risk"
                            if (
                                missing_count
                                / len(df)
                            )
                            > 0.2
                            else "🟡 Moderate"
                        ),
                    }
                )

        if missing_data:

            st.dataframe(
                pd.DataFrame(
                    missing_data
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            cards.finding_card(
                "100% Complete",
                "No missing values detected in any column.",
                "success",
            )

    # ========================================================
    # TAB 6: AI DATA DICTIONARY
    # ========================================================

    with p_tab6:

        st.markdown(
            "#### 🤖 AI Data Dictionary"
        )

        st.caption(
            "Factual column-level intelligence, semantic "
            "attribute roles, and analytical guidance."
        )

        # ----------------------------------------------------
        # VALIDATE DICTIONARY PAYLOAD
        # ----------------------------------------------------

        if (
            not data_dictionary
            or not isinstance(
                data_dictionary,
                dict,
            )
            or not isinstance(
                data_dictionary.get(
                    "columns"
                ),
                list,
            )
            or not data_dictionary.get(
                "columns"
            )
        ):

            st.info(
                "AI Data Dictionary is not available "
                "for this dataset."
            )

        else:

            cols_dict_list = (
                data_dictionary.get(
                    "columns",
                    [],
                )
            )

            dict_rows = []

            # =================================================
            # DICTIONARY TABLE
            # =================================================

            for index, c in enumerate(
                cols_dict_list
            ):

                if not isinstance(
                    c,
                    dict,
                ):
                    continue

                # ---------------------------------------------
                # COLUMN NAME
                # ---------------------------------------------

                name = _extract_field(
                    c,
                    [
                        "column_name",
                        "column",
                        "name",
                        "field",
                        "attribute",
                    ],
                )

                # ---------------------------------------------
                # DETECTED DATA TYPE
                # ---------------------------------------------

                det_type = _extract_field(
                    c,
                    [
                        "detected_type",
                        "data_type",
                        "type",
                    ],
                )

                # ---------------------------------------------
                # GENERIC FALLBACK
                # ---------------------------------------------

                # If the engine payload does not contain
                # a usable name, use the actual dataframe
                # column at the same position.

                if (
                    name
                    == "Not available"
                    and index
                    < len(df.columns)
                ):

                    name = str(
                        df.columns[index]
                    )

                # If the engine payload does not contain
                # a usable detected type, use the actual
                # pandas datatype.

                if (
                    det_type
                    == "Not available"
                    and index
                    < len(df.columns)
                ):

                    det_type = str(
                        df[
                            df.columns[index]
                        ].dtype
                    )

                # ---------------------------------------------
                # SEMANTIC ROLE
                # ---------------------------------------------

                role = _extract_field(
                    c,
                    [
                        "semantic_role",
                        "analytical_role",
                        "role",
                    ],
                )

                # ---------------------------------------------
                # CONFIDENCE
                # ---------------------------------------------

                confidence = c.get(
                    "confidence"
                )

                if confidence is not None:

                    try:

                        confidence_value = (
                            float(
                                confidence
                            )
                        )

                        if (
                            confidence_value
                            <= 1.0
                        ):

                            confidence_string = (
                                f"{confidence_value * 100:.0f}%"
                            )

                        else:

                            confidence_string = (
                                f"{confidence_value:.0f}%"
                            )

                    except Exception:

                        confidence_string = str(
                            confidence
                        )

                else:

                    confidence_string = (
                        "Not available"
                    )

                # ---------------------------------------------
                # POSSIBLE MEANING
                # ---------------------------------------------

                meaning = c.get(
                    "possible_meaning",
                    "Not available",
                )

                # ---------------------------------------------
                # QUALITY CONCERNS
                # ---------------------------------------------

                concerns = c.get(
                    "quality_concerns",
                    [],
                )

                if isinstance(
                    concerns,
                    list,
                ):

                    concerns_string = (
                        ", ".join(
                            str(item)
                            for item in concerns
                        )
                        if concerns
                        else "None detected"
                    )

                elif isinstance(
                    concerns,
                    str,
                ):

                    concerns_string = (
                        concerns
                        if concerns.strip()
                        else "None detected"
                    )

                else:

                    concerns_string = (
                        "Not available"
                    )

                # ---------------------------------------------
                # ANALYTICAL USEFULNESS
                # ---------------------------------------------

                analytical = c.get(
                    "analytical_usefulness",
                    "Not available",
                )

                # ---------------------------------------------
                # ML USEFULNESS
                # ---------------------------------------------

                ml_use = c.get(
                    "ml_usefulness",
                    "Not available",
                )

                # ---------------------------------------------
                # FINAL TABLE ROW
                # ---------------------------------------------

                dict_rows.append(
                    {
                        "Column Name": name,
                        "Detected Data Type": det_type,
                        "Semantic Role": role,
                        "Confidence": confidence_string,
                        "Possible Meaning": meaning,
                        "Quality Concerns": concerns_string,
                        "Analytical Usefulness": analytical,
                        "ML Usefulness": ml_use,
                    }
                )

            # =================================================
            # DISPLAY DICTIONARY TABLE
            # =================================================

            if dict_rows:

                dict_df = pd.DataFrame(
                    dict_rows
                )

                st.dataframe(
                    dict_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # =============================================
                # DEEP COLUMN INTELLIGENCE
                # =============================================

                st.markdown(
                    "##### 🔍 Deep Column Intelligence Explorer"
                )

                for index, c in enumerate(
                    cols_dict_list
                ):

                    if not isinstance(
                        c,
                        dict,
                    ):
                        continue

                    # -----------------------------------------
                    # COLUMN NAME
                    # -----------------------------------------

                    c_name = _extract_field(
                        c,
                        [
                            "column_name",
                            "column",
                            "name",
                            "field",
                            "attribute",
                        ],
                    )

                    # -----------------------------------------
                    # DETECTED TYPE
                    # -----------------------------------------

                    c_type = _extract_field(
                        c,
                        [
                            "detected_type",
                            "data_type",
                            "type",
                        ],
                    )

                    # -----------------------------------------
                    # GENERIC FALLBACK
                    # -----------------------------------------

                    if (
                        c_name
                        == "Not available"
                        and index
                        < len(df.columns)
                    ):

                        c_name = str(
                            df.columns[index]
                        )

                    if (
                        c_type
                        == "Not available"
                        and index
                        < len(df.columns)
                    ):

                        c_type = str(
                            df[
                                df.columns[index]
                            ].dtype
                        )

                    # -----------------------------------------
                    # SEMANTIC ROLE
                    # -----------------------------------------

                    c_role = _extract_field(
                        c,
                        [
                            "semantic_role",
                            "analytical_role",
                            "role",
                        ],
                    )

                    # -----------------------------------------
                    # CONFIDENCE
                    # -----------------------------------------

                    c_confidence = c.get(
                        "confidence"
                    )

                    if (
                        c_confidence
                        is not None
                    ):

                        try:

                            c_confidence_value = (
                                float(
                                    c_confidence
                                )
                            )

                            if (
                                c_confidence_value
                                <= 1.0
                            ):

                                c_confidence_string = (
                                    f"{c_confidence_value * 100:.0f}%"
                                )

                            else:

                                c_confidence_string = (
                                    f"{c_confidence_value:.0f}%"
                                )

                        except Exception:

                            c_confidence_string = str(
                                c_confidence
                            )

                    else:

                        c_confidence_string = (
                            "Not available"
                        )

                    # -----------------------------------------
                    # POSSIBLE MEANING
                    # -----------------------------------------

                    c_meaning = c.get(
                        "possible_meaning",
                        "Not available",
                    )

                    # -----------------------------------------
                    # QUALITY CONCERNS
                    # -----------------------------------------

                    c_concerns = c.get(
                        "quality_concerns",
                        [],
                    )

                    if isinstance(
                        c_concerns,
                        list,
                    ):

                        c_concerns_string = (
                            ", ".join(
                                str(item)
                                for item in c_concerns
                            )
                            if c_concerns
                            else "None detected"
                        )

                    elif isinstance(
                        c_concerns,
                        str,
                    ):

                        c_concerns_string = (
                            c_concerns
                            if c_concerns.strip()
                            else "None detected"
                        )

                    else:

                        c_concerns_string = (
                            "Not available"
                        )

                    # -----------------------------------------
                    # ANALYTICAL USEFULNESS
                    # -----------------------------------------

                    c_analytical = c.get(
                        "analytical_usefulness",
                        "Not available",
                    )

                    # -----------------------------------------
                    # ML USEFULNESS
                    # -----------------------------------------

                    c_ml = c.get(
                        "ml_usefulness",
                        "Not available",
                    )

                    # =========================================
                    # EXPANDER
                    # =========================================

                    with st.expander(
                        f"📌 {c_name} "
                        f"({c_type} • {c_role})"
                    ):

                        st.markdown(
                            f"**Confidence:** "
                            f"`{c_confidence_string}`"
                        )

                        st.markdown(
                            f"**Possible Meaning:** "
                            f"{c_meaning}"
                        )

                        st.markdown(
                            f"**Quality Concerns:** "
                            f"{c_concerns_string}"
                        )

                        st.markdown(
                            f"**Analytical Usefulness:** "
                            f"{c_analytical}"
                        )

                        st.markdown(
                            f"**ML Usefulness:** "
                            f"{c_ml}"
                        )

            else:

                st.info(
                    "AI Data Dictionary is not available "
                    "for this dataset."
                )