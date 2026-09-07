# FILE: ui.py

import streamlit as st


def show_metric_cards(rows, columns, missing_cells, duplicate_rows):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{rows:,}")

    with col2:
        st.metric("Columns", columns)

    with col3:
        st.metric("Missing Cells", f"{missing_cells:,}")

    with col4:
        st.metric("Duplicate Rows", f"{duplicate_rows:,}")


def show_quality_score(quality_score, quality_status):
    st.subheader("Data Quality Score")

    st.metric(
        "Quality Score",
        f"{quality_score}/100"
    )

    st.write(
        f"Status: **{quality_status}**"
    )


def show_ml_readiness(ml_score, ml_status):
    st.subheader("ML Readiness")

    st.metric(
        "ML Readiness Score",
        f"{ml_score}/100"
    )

    st.write(
        f"Status: **{ml_status}**"
    )


def show_quality_findings(quality_findings):
    st.subheader("Quality Findings")

    if not quality_findings:
        st.success("No major quality issues detected.")
        return

    for finding in quality_findings:
        severity = finding.get(
            "Severity",
            "Medium"
        )

        issue = finding.get(
            "Issue",
            "Unknown Issue"
        )

        details = finding.get(
            "Details",
            ""
        )

        if severity == "High":
            st.error(
                f"🔴 **{issue}** — {details}"
            )
        elif severity == "Medium":
            st.warning(
                f"🟡 **{issue}** — {details}"
            )
        else:
            st.info(
                f"🔵 **{issue}** — {details}"
            )


def show_agent_decisions(decisions):
    st.subheader("Agent Decisions")

    if not decisions:
        st.info("No agent decisions available.")
        return

    for decision in decisions:
        priority = decision.get(
            "Priority",
            "Low"
        )

        finding = decision.get(
            "Finding",
            ""
        )

        action = decision.get(
            "Action",
            ""
        )

        st.markdown(
            f"**{priority} Priority — {finding}**"
        )

        st.write(action)


def show_executive_summary(summary):
    st.subheader("Executive Summary")

    if not summary:
        st.info("No executive summary available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Quality Score",
            f"{summary.get('Quality Score', 0)}/100"
        )

    with col2:
        st.metric(
            "ML Readiness",
            f"{summary.get('ML Readiness Score', 0)}/100"
        )

    st.write(
        f"**Risk Level:** "
        f"{summary.get('Risk Level', 'Unknown')}"
    )

    st.write(
        f"**Most Important Finding:** "
        f"{summary.get('Most Important Finding', 'N/A')}"
    )

    st.write(
        f"**Key Opportunity:** "
        f"{summary.get('Key Opportunity', 'N/A')}"
    )

    st.write(
        f"**Next Best Action:** "
        f"{summary.get('Next Best Action', 'N/A')}"
    )


def show_column_table(column_intelligence_df):
    st.subheader("Attribute Intelligence")

    if column_intelligence_df.empty:
        st.info("No attribute information available.")
        return

    st.dataframe(
        column_intelligence_df,
        use_container_width=True,
        hide_index=True
    )


def show_data_preview(df, rows=10):
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(rows),
        use_container_width=True,
        hide_index=True
    )


def format_report_headings_for_web(report_text: str) -> str:
    """Formats Executive AI Report section headings with high-contrast accent blocks in the Streamlit Web UI."""
    if not report_text:
        return ""

    target_headings = [
        "Executive Assessment", "Dataset Overview", "Key Findings",
        "Data Quality Assessment", "Statistical Insights", "Visualization Insights",
        "Machine Learning Readiness", "Dataset Changes", "Impact Assessment",
        "Limitations", "Recommendations", "Next Best Actions", "AI Status",
    ]

    lines = report_text.splitlines()
    formatted_lines = []

    for line in lines:
        stripped = line.lstrip("#").strip()
        matched = None
        for th in target_headings:
            if th.lower() in stripped.lower():
                matched = th
                break

        if matched and (line.startswith("#") or line.isupper() or len(stripped) < 40):
            block = f"""
<div style="background: linear-gradient(90deg, rgba(99, 102, 241, 0.18), rgba(18, 23, 33, 0.95)); border-left: 4px solid #6366f1; border-radius: 6px; padding: 8px 14px; margin-top: 18px; margin-bottom: 10px;">
    <div style="font-size: 1.02rem; font-weight: 800; color: #edf2f7; letter-spacing: -0.01em; text-transform: uppercase;">
        📌 {matched.upper()}
    </div>
</div>
"""
            formatted_lines.append(block)
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def show_ai_report(report):
    st.subheader("🤖 Executive Intelligence Briefing")

    if not report:
        st.info("No AI report available.")
        return

    st.markdown(format_report_headings_for_web(report), unsafe_allow_html=True)

    