"""
views/overview_view.py — Dataset Intelligence Command Center overview.
Displays real dataset metrics, "Wow Moment" agent decision, Dataset Intelligence Score,
Data Story Mode, Analyst Exploration Readiness Score, Next Best Analysis, and interactive Column Explorer.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards, state


def render(
    df: pd.DataFrame,
    uploaded_file,
    rows: int,
    columns: int,
    numeric_df: pd.DataFrame,
    categorical_columns: list,
    identifier_columns: list,
    date_columns: list,
    target_candidates: list,
    quality_score: int,
    quality_status: str,
    ml_score: int,
    ml_status: str,
) -> None:

    # --------------------------------------------------------
    # COMMAND CENTER HERO
    # --------------------------------------------------------
    st.markdown(
        """
        <div class="command-center-hero">
            <div class="hero-title-text">🌐 Dataset Intelligence Command Center</div>
            <div class="hero-subtitle-text">
                Upload, investigate, compare, and understand your datasets with an adaptive AI intelligence engine.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # "WOW MOMENT" AGENT DECISION CARD
    # --------------------------------------------------------
    change_result = state.get_change_analysis()
    if change_result:
        verdict = change_result.get("recommendation", "UNKNOWN")
        next_act = change_result.get("next_action", "Continue analysis")
        changes_n = len(change_result.get("changes", {}).get("added_columns", [])) + len(change_result.get("changes", {}).get("removed_columns", []))
        quality_n = len(change_result.get("impact", []))
        recon_n = 1 if change_result.get("reconsideration", {}).get("status") == "RECONSIDER" else 0

        cards.wow_moment_card(
            verdict=verdict,
            next_action=next_act,
            changes_count=changes_n,
            quality_issues_count=quality_n,
            reconsidered_count=recon_n,
        )
    else:
        cards.wow_moment_card(
            verdict=quality_status,
            next_action="Review quality findings & activate change intelligence comparison",
            changes_count=0,
            quality_issues_count=len(df.columns[df.isna().any()]),
            reconsidered_count=0,
        )

    # --------------------------------------------------------
    # SIGNATURE FEATURE 1 — AUTOMATED DATA STORY MODE
    # --------------------------------------------------------
    num_cols = list(numeric_df.columns)
    missing_pct = (df.isna().sum().sum() / (rows * columns) * 100) if rows * columns else 0.0
    dup_rows = df.duplicated().sum()

    story_happened = (
        f"The dataset contains **{rows:,} records** across **{columns:,} attributes** "
        f"({len(num_cols)} numerical, {len(categorical_columns)} categorical). "
        f"Overall missing cell rate is **{missing_pct:.2f}%** with **{dup_rows:,} duplicate rows**."
    )
    story_matters = (
        f"Data quality score is evaluated at **{quality_score}/100** ({quality_status}). "
        + ("High completeness allows reliable statistical modeling." if missing_pct < 5 else "Missing values require targeted imputation prior to downstream ML.")
    )
    top_num = num_cols[0] if num_cols else (df.columns[0] if len(df.columns) > 0 else "Primary attribute")
    top_num_2 = num_cols[1] if len(num_cols) > 1 else top_num
    story_investigate = (
        f"Investigate the distribution of **'{top_num}'** in the Calculation Lab, "
        f"and evaluate its 2D/3D spatial relationship with **'{top_num_2}'**."
    )

    cards.data_story_card(
        what_happened=story_happened,
        why_it_matters=story_matters,
        what_to_investigate=story_investigate,
    )

    # --------------------------------------------------------
    # SIGNATURE FEATURE 2 — ANALYST EXPLORATION READINESS SCORE (0-100)
    # --------------------------------------------------------
    readiness_reasons = [
        (missing_pct < 5.0, f"Completeness: {100 - missing_pct:.1f}% valid data cells"),
        (dup_rows == 0, f"Uniqueness: {dup_rows:,} duplicate rows detected"),
        (len(num_cols) >= 2, f"Numeric Coverage: {len(num_cols)} numerical features available for 2D/3D modeling"),
        (len(categorical_columns) >= 1, f"Categorical Diversity: {len(categorical_columns)} categorical dimensions available for segmentation"),
    ]
    readiness_score = int(quality_score * 0.5 + (100 - min(50, missing_pct * 5)) * 0.3 + (100 if dup_rows == 0 else 70) * 0.2)
    cards.analyst_readiness_card(score=readiness_score, reasons=readiness_reasons)

    # --------------------------------------------------------
    # SIGNATURE FEATURE 3 — RECOMMENDED NEXT BEST ANALYSIS
    # --------------------------------------------------------
    if len(num_cols) >= 2:
        next_title = f"Explore Relationship: '{num_cols[0]}' vs. '{num_cols[1]}'"
        next_reason = f"Both features are numerical continuous attributes. Pearson correlation and 3D spatial scatter will reveal multivariate trends."
    elif num_cols and categorical_columns:
        next_title = f"Segment '{num_cols[0]}' by Category '{categorical_columns[0]}'"
        next_reason = "Grouped mean/median calculation will highlight category-level variance heterogeneity."
    else:
        next_title = "Run Descriptive Summary in Calculation Lab"
        next_reason = "Basic statistics will reveal central tendency, quartiles, and skewness."

    cards.next_best_analysis_card(recommendation_title=next_title, reason=next_reason)

    st.divider()

    # --------------------------------------------------------
    # DATASET INTELLIGENCE SCORE BREAKDOWN
    # --------------------------------------------------------
    st.markdown("### 🏆 Dataset Intelligence Score Breakdown")

    struct_score = 100 - min(30, len(df.columns[df.isna().any()]) * 5)
    stat_score = 100 - min(40, int(df.isna().sum().sum() / (rows * columns) * 100 * 2)) if rows * columns else 100
    total_intel_score = int(quality_score * 0.35 + ml_score * 0.35 + struct_score * 0.15 + stat_score * 0.15)

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Overall Score", f"{total_intel_score}/100", "Composite")
    s2.metric("Data Quality", f"{quality_score}/100", quality_status)
    s3.metric("ML Readiness", f"{ml_score}/100", ml_status)
    s4.metric("Structural Health", f"{struct_score}/100")
    s5.metric("Statistical Stability", f"{stat_score}/100")

    st.divider()

    # --------------------------------------------------------
    # EXECUTIVE DATASET METRICS TABLE & GAUGES
    # --------------------------------------------------------
    cards.section_header("Executive Dataset Overview")
    c1, c2 = st.columns([1.15, 1])

    with c1:
        overview_df = pd.DataFrame({
            "Attribute Metric": [
                "Dataset File", "Total Records", "Total Attributes", "Numerical Features",
                "Categorical Features", "Identifier Candidates",
                "Date/Time Attributes", "Target Candidates",
            ],
            "Value": [
                uploaded_file.name if uploaded_file else "In-Memory Data",
                f"{rows:,}",
                f"{columns:,}",
                f"{len(numeric_df.columns):,}",
                f"{len(categorical_columns):,}",
                f"{len(identifier_columns):,}",
                f"{len(date_columns):,}",
                f"{len(target_candidates):,}",
            ],
        })
        st.dataframe(overview_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Health Gauges")
        st.markdown("**Data Quality Score**")
        st.progress(quality_score / 100)
        st.caption(f"Score: **{quality_score}/100** · Status: **{quality_status}**")

        st.markdown("**ML Readiness Score**")
        st.progress(ml_score / 100)
        st.caption(f"Score: **{ml_score}/100** · Status: **{ml_status}**")

    # --------------------------------------------------------
    # INTERACTIVE COLUMN EXPLORER
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 🔎 Interactive Column Explorer")
    cards.render_column_explorer(df)

    st.divider()
    st.markdown("### 📋 Dataset Sample Preview (Top 20 Records)")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)