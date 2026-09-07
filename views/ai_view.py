"""
views/ai_view.py — Executive AI Intelligence & "Ask the Dataset" workspace.
"""

from __future__ import annotations
import streamlit as st

from components import cards, state


def render(
    client,
    uploaded_file,
    rows: int,
    columns: int,
    numeric_df,
    categorical_columns: list,
    identifier_columns: list,
    date_columns: list,
    target_candidates: list,
    quality_score: int,
    ml_score: int,
    quality_findings: list,
    agent_decisions: list,
    numerical_summary,
    categorical_summary_df,
    correlation_pairs,
    anomaly_df,
    column_intelligence_df,
    df,
    generate_ai_report_fn,
    show_ai_report_fn,
) -> None:

    cards.section_header(
        "Executive AI Insights & Interactive Analyst Workspace",
        "Gemini interprets analytical evidence to produce executive briefings and answer targeted dataset questions.",
    )

    if client is None:
        cards.finding_card(
            "AI Service Status",
            "Gemini API key is not connected. All deterministic Python profiling, 3D visualizations, and local reports remain fully functional.",
            "warning",
        )

    # --------------------------------------------------------
    # "ASK THE DATASET" INTERACTIVE INTERVIEW / Q&A
    # --------------------------------------------------------
    st.markdown("### 💬 Ask the Dataset — Interactive Analyst")
    st.caption("Select a targeted analytical question to get an instant grounded answer based on calculated dataset facts.")

    suggested_questions = [
        "What changed between the two versions?",
        "Which columns need attention?",
        "Is this dataset ready for ML?",
        "What changed statistically?",
        "Which finding should I reconsider?",
        "What should I analyze next?",
    ]

    q_cols = st.columns(3)
    selected_question = None

    for i, q_text in enumerate(suggested_questions):
        with q_cols[i % 3]:
            if st.button(f"❓ {q_text}", key=f"ask_q_{i}", use_container_width=True):
                selected_question = q_text

    custom_q = st.text_input("Or enter a custom question about your dataset:", key="custom_question_input")
    if custom_q:
        selected_question = custom_q

    change_result = state.get_change_analysis()

    if selected_question:
        st.markdown(f"#### 🔍 Query: *\"{selected_question}\"*")
        with st.spinner("Analyzing dataset facts to generate grounded response..."):

            # Construct targeted answer based on real calculated facts
            answer_text = ""
            if "changed between" in selected_question.lower():
                if change_result:
                    changes = change_result.get("changes", {})
                    answer_text = (
                        f"**Dataset Evolution Summary:**\n"
                        f"• Added columns (+): `{len(changes.get('added_columns', []))}`\n"
                        f"• Removed columns (-): `{len(changes.get('removed_columns', []))}`\n"
                        f"• Type conflicts: `{len(changes.get('type_changes', []))}`\n"
                        f"• Renames detected (↔): `{len(changes.get('possible_renames', []))}`\n\n"
                        f"**Verdict:** `{change_result.get('recommendation', 'N/A')}`"
                    )
                else:
                    answer_text = "Upload a Previous Dataset (V1) and Updated Dataset (V2) in the sidebar to inspect version changes."
            elif "need attention" in selected_question.lower() or "columns" in selected_question.lower():
                null_cols = df.columns[df.isna().any()].tolist()
                answer_text = (
                    f"**Columns Requiring Attention:**\n"
                    f"• Columns with missing cells: `{', '.join(null_cols) if null_cols else 'None'}`\n"
                    f"• Data Quality Score: `{quality_score}/100`"
                )
            elif "ready for ml" in selected_question.lower():
                verdict = change_result.get("recommendation", "N/A") if change_result else ("READY" if ml_score >= 80 else "NEEDS ATTENTION")
                answer_text = (
                    f"**ML Readiness Evaluation:**\n"
                    f"• Score: `{ml_score}/100`\n"
                    f"• Overall Status: `{verdict}`\n"
                    f"• Target Candidates: `{', '.join(target_candidates) if target_candidates else 'None'}`"
                )
            elif "reconsider" in selected_question.lower():
                recon = change_result.get("reconsideration", {}) if change_result else {}
                answer_text = (
                    f"**Adaptive Reconsideration:**\n"
                    f"• Status: `{recon.get('status', 'NO PRIOR FINDING')}`\n"
                    f"• Message: {recon.get('message', 'No findings affected.')}"
                )
            else:
                next_act = change_result.get("next_action", "Continue standard exploratory data analysis.") if change_result else "Address identified missing values and validate column data types."
                answer_text = f"**Recommended Next Action:** `{next_act}`"

            cards.finding_card("Grounded Intelligence Answer", answer_text, "info")

    st.divider()

    # --------------------------------------------------------
    # EXECUTIVE BRIEFING GENERATION
    # --------------------------------------------------------
    st.markdown("### 📄 Full Executive Briefing")

    if st.button("Generate Complete Executive Briefing", type="primary", key="generate_ai_report_main"):
        with st.spinner("AI Agent is compiling full executive briefing..."):
            profile_data = {
                "dataset": uploaded_file.name if uploaded_file else "Data",
                "rows": rows,
                "columns": columns,
                "numerical_columns": list(numeric_df.columns),
                "categorical_columns": list(categorical_columns),
                "identifier_columns": identifier_columns,
                "date_columns": date_columns,
                "target_candidates": target_candidates,
            }

            numerical_evidence = (
                numerical_summary.reset_index().rename(columns={"index": "Attribute"}).head(30)
                if not numerical_summary.empty else None
            )
            categorical_evidence = categorical_summary_df.head(30) if not categorical_summary_df.empty else None
            correlation_evidence = (
                correlation_pairs[["Attribute 1", "Attribute 2", "Correlation"]].head(15)
                if not correlation_pairs.empty else None
            )
            anomaly_evidence = anomaly_df.head(30) if not anomaly_df.empty else None
            column_evidence = column_intelligence_df.head(50)

            report = generate_ai_report_fn(
                profile=profile_data,
                quality_score=quality_score,
                ml_score=ml_score,
                quality_findings=quality_findings,
                agent_decisions=agent_decisions,
                target_candidates=target_candidates,
                numerical_summary=numerical_evidence,
                categorical_summary=categorical_evidence,
                correlation_pairs=correlation_evidence,
                anomaly_data=anomaly_evidence,
                column_intelligence=column_evidence,
            )

            if report:
                state.set_ai_report(report)
                st.success("Executive intelligence generated successfully.")

    if state.get_ai_report():
        st.divider()
        show_ai_report_fn(state.get_ai_report())

        from utils import format_report_for_download
        download_data = format_report_for_download(state.get_ai_report())

        st.download_button(
            "Download Executive Report",
            data=download_data,
            file_name="AI_Dataset_Executive_Report.txt",
            mime="text/plain",
            key="download_ai_report",
        )