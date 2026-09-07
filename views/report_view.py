"""
views/report_view.py — Dedicated Executive Report view for judges and stakeholders.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from components import cards, state
from engine.ai_report import generate_ai_report


def render(
    df: pd.DataFrame | None,
    uploaded_file,
    previous_file,
    updated_file,
    profile: dict,
    quality_findings: list,
    agent_decisions: list,
    executive_summary: dict,
    numerical_summary,
    categorical_summary_df,
    correlation_pairs,
    anomaly_df,
    column_intelligence_df,
    quality_score: int,
    ml_score: int,
    client,
) -> None:

    cards.section_header(
        "Executive Intelligence Report",
        "A comprehensive executive summary of dataset health, evolution, risks, and recommended actions.",
    )

    change_result = state.get_change_analysis()

    # Top KPI Banner
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Quality", f"{quality_score}/100")
    c2.metric("ML Readiness", f"{ml_score}/100")

    if change_result:
        recommendation = change_result.get("recommendation", "UNKNOWN")
        next_action = change_result.get("next_action", "Continue analysis")
        c3.metric("Change Status", recommendation)
        c4.metric("Next Action", next_action)
    else:
        c3.metric("Change Analysis", "Not Run")
        c4.metric("Next Action", executive_summary.get("Next Best Action", "N/A"))

    st.divider()

    # Generate AI / Fallback Report
    if st.button("Generate Executive AI Report", type="primary", key="generate_exec_report"):
        with st.spinner("Generating executive report..."):
            report = generate_ai_report(
                profile=profile,
                quality_findings=quality_findings,
                agent_decisions=agent_decisions,
                executive_summary=executive_summary,
                numerical_summary=numerical_summary,
                categorical_summary=categorical_summary_df,
                visualization_plan=None,
                quality_score=quality_score,
                ml_readiness=ml_score,
            )
            state.set_ai_report(report)

    ai_report = state.get_ai_report()
    if ai_report:
        from ui import format_report_headings_for_web
        from utils import format_report_for_download

        st.markdown(format_report_headings_for_web(ai_report), unsafe_allow_html=True)
        st.download_button(
            "Download Executive Report",
            data=format_report_for_download(ai_report),
            file_name="AI_Dataset_Executive_Report.txt",
            mime="text/plain",
            key="download_exec_view_report",
        )
    else:
        # Default structured summary view when report hasn't been clicked yet
        st.markdown("### Dataset Summary & Health")
        st.write(
            f"**Dataset:** `{uploaded_file.name if uploaded_file else 'In-Memory Data'}` | "
            f"**Records:** `{profile.get('rows', 0):,}` | "
            f"**Attributes:** `{profile.get('columns', 0):,}` | "
            f"**Missing Cells:** `{profile.get('missing_cells', 0):,}` | "
            f"**Duplicate Rows:** `{profile.get('duplicate_rows', 0):,}`"
        )

        st.markdown("#### Key Executive Findings")
        f1, f2 = st.columns(2)
        with f1:
            cards.finding_card("Most Important Finding", executive_summary.get("Most Important Finding", "N/A"), "info")
        with f2:
            cards.finding_card("Key Opportunity", executive_summary.get("Key Opportunity", "N/A"), "success")

        cards.finding_card("Recommended Action", executive_summary.get("Next Best Action", "N/A"), "warning")

        if change_result:
            st.divider()
            st.markdown("### Change Intelligence Summary")
            changes = change_result.get("changes", {})
            reconsideration = change_result.get("reconsideration", {})

            st.write(
                f"- **Added Columns:** {len(changes.get('added_columns', []))}\n"
                f"- **Removed Columns:** {len(changes.get('removed_columns', []))}\n"
                f"- **Data Type Changes:** {len(changes.get('type_changes', []))}\n"
                f"- **Possible Renames:** {len(changes.get('possible_renames', []))}\n"
                f"- **Reconsideration Status:** {reconsideration.get('status', 'N/A')}\n"
                f"- **Final Verdict:** `{change_result.get('recommendation', 'UNKNOWN')}`"
            )
