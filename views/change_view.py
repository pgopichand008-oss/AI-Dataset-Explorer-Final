"""
views/change_view.py — Adaptive dataset change intelligence view.
Implements the 6-stage Mystery Mission workflow:
DETECT -> ANALYZE -> COMPARE -> RECONSIDER -> RE-PLAN -> REPORT
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards, state


def render(
    previous_file,
    updated_file,
    most_important_finding: str,
    load_dataset_fn,
    run_intelligence_analysis_fn,
) -> None:

    cards.section_header(
        "Dataset Change Intelligence",
        "Detect structural version changes, quality shifts, and adaptively reconsider previous findings.",
    )

    if previous_file is None or updated_file is None:
        cards.finding_card(
            "Upload both datasets to activate Change Intelligence",
            "Upload a Previous Dataset (V1) and an Updated Dataset (V2) from the sidebar to start comparing structural, statistical, and quality changes.",
            "info",
        )
        st.markdown(
            "**6-Stage Mission Control Timeline:**\n"
            "`01 DETECT` → `02 ANALYZE` → `03 COMPARE` → `04 RECONSIDER` → `05 RE-PLAN` → `06 REPORT`"
        )
        return

    old_df, old_error = load_dataset_fn(previous_file)
    new_df, new_error = load_dataset_fn(updated_file)

    if old_error is not None:
        cards.finding_card("Read error", f"Unable to read previous dataset: {old_error}", "danger")
        return
    if new_error is not None:
        cards.finding_card("Read error", f"Unable to read updated dataset: {new_error}", "danger")
        return
    if old_df is None or new_df is None or old_df.empty or new_df.empty:
        cards.finding_card("Empty dataset", "One of the datasets contains no usable records.", "danger")
        return

    cards.finding_card("Version Control Loaded", f"Comparing `{previous_file.name}` ({len(old_df):,} rows) with `{updated_file.name}` ({len(new_df):,} rows).", "success")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Previous Dataset (V1)**")
        st.caption(previous_file.name)
        st.metric("Rows", f"{len(old_df):,}")
        st.metric("Columns", f"{len(old_df.columns):,}")
    with c2:
        st.markdown("**Updated Dataset (V2)**")
        st.caption(updated_file.name)
        st.metric("Rows", f"{len(new_df):,}")
        st.metric("Columns", f"{len(new_df.columns):,}")

    st.divider()

    previous_finding = st.text_area(
        "Previous Important Finding (for adaptive reconsideration)",
        value=most_important_finding if most_important_finding else "Revenue is strongly related to customer age.",
        height=90,
        key="previous_finding_input",
    )

    if st.button("Run Dataset Intelligence Analysis", type="primary", key="run_change_analysis"):
        with st.spinner("Agent is running 6-stage change intelligence analysis..."):
            try:
                result = run_intelligence_analysis_fn(old_df, new_df, previous_finding)
                state.set_change_analysis(result)
            except Exception as e:
                cards.finding_card("Change intelligence failed", f"{type(e).__name__}: {e}", "danger")

    result = state.get_change_analysis()
    if result is None:
        return

    changes = result.get("changes", {})
    added = changes.get("added_columns", [])
    removed = changes.get("removed_columns", [])
    type_changes = changes.get("type_changes", [])
    renames = changes.get("possible_renames", [])
    invalid_vals = changes.get("invalid_values", {})

    # --------------------------------------------------------
    # "WHAT CHANGED?" VISUAL SUMMARY
    # --------------------------------------------------------
    st.divider()
    cards.what_changed_summary_card(
        changes_count=len(added) + len(removed) + len(type_changes) + len(renames),
        added=len(added),
        removed=len(removed),
        renames=len(renames),
        type_conflicts=len(type_changes),
    )

    # --------------------------------------------------------
    # 6-STAGE MISSION TIMELINE WORKFLOW STATUS ROW
    # --------------------------------------------------------
    st.markdown("### 🚀 Mission Timeline Workflow")
    step_statuses = result.get("step_statuses", [])
    if step_statuses:
        cols = st.columns(len(step_statuses))
        for col, step_info in zip(cols, step_statuses):
            with col:
                cards.step_card(
                    step_name=f"0{cols.index(col)+1} {step_info.get('step', '')}",
                    title=step_info.get("title", ""),
                    detail=step_info.get("detail", ""),
                    kind=step_info.get("kind", "info"),
                )

    # --------------------------------------------------------
    # DETECT & STRUCTURAL CHANGES
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 1. Detect: Structural & Rename Intelligence")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Added Columns", f"+{len(added)}")
    m2.metric("Removed Columns", f"-{len(removed)}")
    m3.metric("Type Conflicts", len(type_changes))
    m4.metric("Possible Renames", f"↔ {len(renames)}")

    if added:
        st.markdown("**Added Columns (+)**")
        for col in added:
            cards.finding_card("Added Column", f"+ {col}", "success")

    if removed:
        st.markdown("**Removed Columns (-)**")
        for col in removed:
            cards.finding_card("Removed Column", f"- {col}", "danger")

    if renames:
        st.markdown("**Column Rename Detection**")
        for r in renames:
            cards.rename_card(
                old_col=r.get("old_column", ""),
                new_col=r.get("new_column", ""),
                similarity=r.get("confidence", 0.0),
                reason=r.get("reason", "Similar names and compatible value patterns."),
            )

    if type_changes:
        st.markdown("**Data Type Changes & Conflicts**")
        st.dataframe(pd.DataFrame(type_changes), use_container_width=True, hide_index=True)

    if invalid_vals:
        st.markdown("**Invalid / Mixed-Type Values Detected**")
        invalid_rows = []
        for col_name, inv_info in invalid_vals.items():
            invalid_rows.append({
                "Column": col_name,
                "Invalid Count": inv_info.get("invalid_count", 0),
                "Valid Count": inv_info.get("valid_count", 0),
                "Examples": ", ".join([str(x) for x in inv_info.get("invalid_examples", [])]),
            })
        st.dataframe(pd.DataFrame(invalid_rows), use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # COMPARE & STATISTICAL DRIFT
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 2. Compare: Statistical Drift & Quality")
    statistics = result.get("statistics", [])
    if statistics:
        st.markdown("**Numerical Statistical Shift Analysis**")
        display_stats = []
        for s in statistics:
            col_name = s["column"]
            old_m = s["old_mean"]
            new_m = s["new_mean"]
            chg = s["mean_change"]
            pct_str = f"{(chg / old_m * 100):+.1f}%" if old_m != 0 else "N/A"

            display_stats.append({
                "Attribute": col_name,
                "Old Mean": old_m,
                "New Mean": new_m,
                "Mean Shift": chg,
                "Percentage Change": pct_str,
                "Old Std": s["old_std"],
                "New Std": s["new_std"],
            })

        st.dataframe(pd.DataFrame(display_stats), use_container_width=True, hide_index=True)

    quality = result.get("quality", {})
    missing_changes = quality.get("missing_changes", [])
    if missing_changes:
        st.markdown("**Missing Value Drift**")
        st.dataframe(pd.DataFrame(missing_changes), use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # RECONSIDER PREVIOUS FINDINGS
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 3. Reconsider: Adaptive Reasoning")
    reconsideration = result.get("reconsideration", {})
    status = reconsideration.get("status", "UNKNOWN")

    if status == "RECONSIDER":
        cards.finding_card(
            "⚠ RECONSIDERATION REQUIRED",
            reconsideration.get("message", "Previous finding is affected by dataset changes."),
            "warning",
        )
        for w in reconsideration.get("warnings", []):
            st.warning(f"• {w}")
    elif status == "STILL_VALID":
        cards.finding_card("✓ Previous Finding Still Valid", reconsideration.get("message", "Previous finding remains valid."), "success")
    else:
        cards.finding_card("ℹ No Prior Finding Provided", "No previous finding was supplied for reconsideration.", "info")

    # --------------------------------------------------------
    # RE-PLAN NEXT ACTION
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 4. Re-plan: Determined Next Best Action")
    next_action = result.get("next_action", "Continue analysis")
    cards.finding_card("🎯 Determined Next Action", f"**Recommended Action:** `{next_action}`", "info")

    # --------------------------------------------------------
    # FINAL RECOMMENDATION
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 5. Final ML & Dataset Recommendation")
    recommendation = result.get("recommendation", "UNKNOWN")
    kind = {
        "READY FOR FURTHER ANALYSIS": "success",
        "NEEDS ATTENTION BEFORE ANALYSIS": "warning",
        "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION": "danger",
    }.get(recommendation, "info")
    cards.finding_card("Dataset Verdict", recommendation, kind)