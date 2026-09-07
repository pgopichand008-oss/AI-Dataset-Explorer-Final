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
    from components import cards
    st.subheader("Quality Findings")

    if not quality_findings:
        st.success("No major quality issues detected.")
        return

    if isinstance(quality_findings, list):
        for finding in quality_findings:
            if isinstance(finding, dict):
                severity = finding.get("Severity", "Medium")
                issue = finding.get("Issue", finding.get("finding", "Quality Issue"))
                details = finding.get("Details", "")
                kind = "danger" if severity == "High" else ("warning" if severity == "Medium" else "info")
                cards.finding_card(issue, f"<strong>Severity:</strong> {severity}<br>{details}", kind)
            else:
                cards.finding_card("Finding", str(finding), "info")
    elif isinstance(quality_findings, dict):
        for issue, details in quality_findings.items():
            cards.finding_card(issue, str(details), "info")
    else:
        st.write(str(quality_findings))


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
    """Formats Executive AI Report section headings with high-contrast accent blocks in the Streamlit Web UI, preventing duplicates and stripping artifacts."""
    if not report_text:
        return ""

    target_headings = [
        "Executive Assessment", "Dataset Overview", "Key Findings",
        "Data Quality Assessment", "Data Quality", "Statistical Insights", "Visualization Insights",
        "Machine Learning Readiness", "ML Readiness", "Dataset Changes", "Impact Assessment",
        "Opportunities", "Limitations", "Recommendations", "Recommended Action", "Recommended Actions",
        "Final Verdict", "Final Recommendation", "Next Best Actions", "AI Status",
    ]

    lines = report_text.splitlines()
    formatted_lines = []
    seen_headings = set()

    for line in lines:
        # Strip [svg] and anchor fragments
        clean_line = line.split("[svg]")[0].split("http://localhost")[0].strip()
        if not clean_line:
            continue

        stripped = clean_line.lstrip("#").strip()
        matched = None
        for th in target_headings:
            if th.lower() in stripped.lower():
                matched = th
                break

        if matched and (clean_line.startswith("#") or clean_line.isupper() or len(stripped) < 40):
            heading_key = matched.upper()
            if heading_key in seen_headings:
                continue  # Skip duplicate headings
            seen_headings.add(heading_key)
            block = f"""
<div style="background: linear-gradient(90deg, rgba(99, 102, 241, 0.18), rgba(18, 23, 33, 0.95)); border-left: 4px solid #6366f1; border-radius: 6px; padding: 8px 14px; margin-top: 18px; margin-bottom: 10px;">
    <div style="font-size: 1.15rem; font-weight: 800; color: #edf2f7; letter-spacing: 0.02em; text-transform: uppercase;">
        📌 {heading_key}
    </div>
</div>
"""
            formatted_lines.append(block)
        else:
            formatted_lines.append(clean_line)

    return "\n".join(formatted_lines)


def show_ai_report(report):
    st.subheader("🤖 Executive Intelligence Briefing")

    if not report:
        st.info("No AI report available.")
        return

    st.markdown(format_report_headings_for_web(report), unsafe_allow_html=True)

    