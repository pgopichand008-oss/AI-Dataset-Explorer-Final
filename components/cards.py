"""
components/cards.py — Reusable, theme-aware UI building blocks used across every view.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def info_card(title: str, value: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="info-card-value">{value}</div>
            <div class="info-card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ml_card(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="ml-card">
            <div class="ml-card-title">{title}</div>
            <div class="ml-card-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, kind: str = "success") -> str:
    """Returns pill HTML; kind is one of success|warning|error|info."""
    return f'<div class="status-pill status-{kind}">{label}</div>'


def finding_card(title: str, body: str, kind: str = "info") -> None:
    """kind: success | warning | danger | info"""
    st.markdown(
        f"""
        <div class="finding-card finding-{kind}">
            <strong>{title}</strong><br>{body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def rename_card(old_col: str, new_col: str, similarity: float, reason: str) -> None:
    """Renders a visually explainable column rename card."""
    st.markdown(
        f"""
        <div class="finding-card finding-info" style="border-left: 4px solid #38bdf8;">
            <div style="font-size: 0.78rem; text-transform: uppercase; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">
                ↔ Possible Column Rename Detected
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 4px;">
                <code>{old_col}</code> → <code>{new_col}</code>
            </div>
            <div style="font-size: 0.85rem;">
                <strong>Similarity:</strong> {similarity}% &nbsp;|&nbsp; <strong>Reason:</strong> {reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_card(step_name: str, title: str, detail: str, kind: str = "success") -> None:
    """Renders a stage step card for the 6-stage Mystery Mission workflow."""
    colors = {
        "success": ("#3ddc97", "rgba(61, 220, 151, 0.12)"),
        "warning": ("#f6b93b", "rgba(246, 185, 59, 0.12)"),
        "danger": ("#fb7185", "rgba(251, 113, 133, 0.12)"),
        "info": ("#7c7cfb", "rgba(124, 124, 251, 0.12)"),
    }
    fg, bg = colors.get(kind, colors["info"])

    st.markdown(
        f"""
        <div style="background: {bg}; border: 1px solid {fg}44; border-radius: 12px; padding: 12px; margin-bottom: 8px; min-height: 110px;">
            <div style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: {fg}; letter-spacing: 0.05em;">
                {step_name}
            </div>
            <div style="font-size: 0.92rem; font-weight: 700; margin-top: 2px; margin-bottom: 4px;">
                {title}
            </div>
            <div style="font-size: 0.8rem; opacity: 0.9; line-height: 1.3;">
                {detail}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def wow_moment_card(
    verdict: str,
    next_action: str,
    changes_count: int,
    quality_issues_count: int,
    reconsidered_count: int,
) -> None:
    """High-impact Agent Decision card for the Overview dashboard hero."""
    badge_colors = {
        "READY FOR FURTHER ANALYSIS": ("#3ddc97", "rgba(61, 220, 151, 0.15)"),
        "NEEDS ATTENTION BEFORE ANALYSIS": ("#f6b93b", "rgba(246, 185, 59, 0.15)"),
        "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION": ("#fb7185", "rgba(251, 113, 133, 0.15)"),
    }
    fg, bg = badge_colors.get(verdict, ("#7c7cfb", "rgba(124, 124, 251, 0.15)"))

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {bg}, rgba(17, 19, 25, 0.95)); border: 1px solid {fg}66; border-radius: 16px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: {fg};">
                    ⚡ Agent Decision & Recommendation
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; background: {fg}22; color: {fg}; padding: 6px 14px; border-radius: 999px; border: 1px solid {fg}44;">
                    {verdict}
                </div>
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; margin-bottom: 14px;">
                🎯 Recommended Next Action: <span style="color: {fg};">{next_action}</span>
            </div>
            <div style="display: flex; gap: 20px; font-size: 0.86rem; opacity: 0.9;">
                <div>🔍 <strong>{changes_count}</strong> structural change(s) detected</div>
                <div>⚠️ <strong>{quality_issues_count}</strong> quality concern(s)</div>
                <div>🔄 <strong>{reconsidered_count}</strong> finding(s) reconsidered</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def what_changed_summary_card(
    changes_count: int,
    added: int,
    removed: int,
    renames: int,
    type_conflicts: int,
) -> None:
    """Renders concise visual summary at the top of Change Intelligence."""
    st.markdown(
        f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 18px 22px; margin-bottom: 18px;">
            <div style="font-size: 1.1rem; font-weight: 800; margin-bottom: 8px; color: #38bdf8;">
                🔄 What Changed? — Visual Executive Summary
            </div>
            <div style="font-size: 0.92rem; line-height: 1.6;">
                • <strong>{added}</strong> column(s) added (+)<br>
                • <strong>{removed}</strong> column(s) removed (-)<br>
                • <strong>{renames}</strong> possible column rename(s) detected (↔)<br>
                • <strong>{type_conflicts}</strong> data-type conflict(s) identified
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_column_explorer(df: pd.DataFrame) -> None:
    """Renders interactive Column Explorer for detailed column profiling."""
    col_selected = st.selectbox("Select Attribute to Inspect", list(df.columns), key="col_explorer_select")
    if col_selected:
        series = df[col_selected]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Type", str(series.dtype))
        c2.metric("Unique Values", f"{series.nunique():,}")
        c3.metric("Missing Count", f"{series.isna().sum():,}")
        c4.metric("Missing %", f"{(series.isna().mean() * 100):.1f}%")

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("**Sample Unique Values**")
            sample_vals = [str(v) for v in list(series.dropna().unique()[:10])]
            st.markdown(", ".join(f"`{v}`" for v in sample_vals))
        with c_b:
            if pd.api.types.is_numeric_dtype(series):
                st.markdown("**Numerical Summary Stats**")
                stats_df = pd.DataFrame([{
                    "Mean": round(float(series.mean()), 3),
                    "Median": round(float(series.median()), 3),
                    "Min": round(float(series.min()), 3),
                    "Max": round(float(series.max()), 3),
                    "Std": round(float(series.std()), 3),
                }])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.markdown("**Top Value Counts**")
                st.dataframe(series.value_counts().head(5), use_container_width=True)


def graph_guide_card(
    guide: dict | str,
    x_desc: str | None = None,
    y_desc: str | None = None,
    z_desc: str | None = None,
    color_desc: str | None = None,
    lookouts: str | None = None,
    valid_n: int | None = None,
    excl_n: int | None = None,
) -> None:
    """
    Renders clean, educational 📖 GRAPH GUIDE using native Streamlit Markdown.
    Contains ZERO raw HTML string tags (no <div>, <strong>, or style=).
    """
    if isinstance(guide, dict):
        title = guide.get("title", "Analytical Graph Guide")
        what_rep = guide.get("what_it_represents", "")
        x_txt = guide.get("x_axis", x_desc or "")
        y_txt = guide.get("y_axis", y_desc or "")
        z_txt = guide.get("z_axis", z_desc)
        col_txt = guide.get("color_meaning", color_desc)
        look_txt = guide.get("what_to_look_for", lookouts or "")
        interp_txt = guide.get("how_to_interpret", "")
        insight_txt = guide.get("data_science_insight", "")
    else:
        title = guide
        what_rep = ""
        x_txt = x_desc or ""
        y_txt = y_desc or ""
        z_txt = z_desc
        col_txt = color_desc
        look_txt = lookouts or ""
        interp_txt = ""
        insight_txt = ""

    with st.expander(f"📖 GRAPH GUIDE — {title}", expanded=True):
        if what_rep:
            st.markdown(f"**What it represents:** {what_rep}")
        if x_txt:
            st.markdown(f"• **X-axis:** {x_txt}")
        if y_txt:
            st.markdown(f"• **Y-axis:** {y_txt}")
        if z_txt:
            st.markdown(f"• **Z-axis:** {z_txt}")
        if col_txt:
            st.markdown(f"• **Color / Size:** {col_txt}")

        if valid_n is not None and excl_n is not None:
            st.markdown(f"• **Observations used:** `{valid_n:,}` valid values (`{excl_n:,}` excluded due to missing/invalid numbers)")

        if look_txt:
            st.markdown(f"💡 **What to look for:** {look_txt}")
        if interp_txt:
            st.markdown(f"🔍 **How to interpret:** {interp_txt}")
        if insight_txt:
            st.markdown(f"🧠 **Data Science Insight:** {insight_txt}")


def calculation_result_card(
    operation: str,
    result_fmt: str,
    primary_col: str,
    valid_n: int,
    total_n: int,
    method: str,
    explanation: str,
) -> None:
    """Renders large, compact Calculation Lab result panel using native Streamlit components."""
    excl_n = total_n - valid_n
    st.markdown(f"### 🧮 CALCULATION RESULT — `{operation.upper()}`")
    st.caption(f"Target Attribute: `{primary_col}`")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.metric(label=f"Calculated {operation}", value=result_fmt)
    with c2:
        st.markdown(f"• **Method:** {method}")
        st.markdown(f"• **Observations used:** `{valid_n:,}` valid of `{total_n:,}` total (`{excl_n:,}` excluded)")

    st.markdown(f"💡 **Plain-Language Insight:** {explanation}")


def compare_metrics_card(
    metric1_name: str,
    metric1_val: str,
    metric2_name: str,
    metric2_val: str,
    comparison_text: str,
) -> None:
    """Renders comparative analytical card using native Streamlit components."""
    st.markdown("#### ⚖️ METRIC COMPARISON ANALYSIS")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label=metric1_name, value=metric1_val)
    with c2:
        st.metric(label=metric2_name, value=metric2_val)

    st.markdown(f"🔍 **Comparison Insight:** {comparison_text}")


def pipeline_card(
    step_calc: str,
    step_result: str,
    step_visual: str,
    step_interp: str,
    step_action: str,
) -> None:
    """Renders the signature 🧮 CALCULATION → INSIGHT CHAIN pipeline card."""
    st.markdown("### 🔗 CALCULATION → INSIGHT CHAIN PIPELINE")
    cols = st.columns(5)
    steps = [
        ("1. CALCULATION", step_calc, "#6366f1"),
        ("2. RESULT", step_result, "#38bdf8"),
        ("3. VISUAL", step_visual, "#10b981"),
        ("4. INTERPRETATION", step_interp, "#f59e0b"),
        ("5. NEXT ACTION", step_action, "#a855f7"),
    ]
    for col, (label, val, color) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="background: rgba(18, 23, 33, 0.9); border: 1px solid {color}55; border-top: 3px solid {color}; border-radius: 10px; padding: 10px 12px; height: 100px;">
                    <div style="font-size: 0.68rem; font-weight: 800; color: {color}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                        {label}
                    </div>
                    <div style="font-size: 0.82rem; font-weight: 600; line-height: 1.3; color: #edf2f7;">
                        {val}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def data_story_card(
    what_happened: str,
    why_it_matters: str,
    what_to_investigate: str,
) -> None:
    """Renders automated Data Story Mode explanation card."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(18, 23, 33, 0.95)); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 14px; padding: 18px 22px; margin-bottom: 16px;">
            <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; color: #818cf8; letter-spacing: 0.06em; margin-bottom: 10px;">
                📖 AUTOMATED DATA STORY MODE
            </div>
            <div style="font-size: 0.92rem; margin-bottom: 8px; line-height: 1.5;">
                🔍 <strong>What happened?</strong><br>{what_happened}
            </div>
            <div style="font-size: 0.92rem; margin-bottom: 8px; line-height: 1.5;">
                💡 <strong>Why does it matter?</strong><br>{why_it_matters}
            </div>
            <div style="font-size: 0.92rem; line-height: 1.5;">
                🎯 <strong>What should an analyst investigate next?</strong><br>{what_to_investigate}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def analyst_readiness_card(
    score: int,
    reasons: list[tuple[bool, str]],
) -> None:
    """Renders 0-100 Analyst Exploration Readiness Score card."""
    color = "#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    status = "EXCELLENT READINESS" if score >= 80 else "MODERATE READINESS" if score >= 60 else "NEEDS DATA CLEANING"

    st.markdown(
        f"""
        <div style="background: rgba(18, 23, 33, 0.95); border: 1px solid {color}44; border-left: 5px solid {color}; border-radius: 14px; padding: 18px 22px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <div style="font-size: 0.76rem; font-weight: 800; text-transform: uppercase; color: #a0aec0; letter-spacing: 0.06em;">
                        🔬 ANALYST EXPLORATION READINESS
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 800; color: #edf2f7;">
                        {status}
                    </div>
                </div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {color};">
                    {score} <span style="font-size: 1rem; color: #a0aec0;">/ 100</span>
                </div>
            </div>
            <div style="font-size: 0.85rem; line-height: 1.6;">
                {"".join(f"{'✓' if pos else '⚠️'} <span style='color: {'#10b981' if pos else '#f59e0b'};'>{txt}</span><br>" for pos, txt in reasons)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def next_best_analysis_card(
    recommendation_title: str,
    reason: str,
    action_label: str = "Explore Relationship",
) -> None:
    """Renders adaptive Next Best Analysis recommendation card."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(18, 23, 33, 0.95)); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 14px; padding: 18px 22px; margin-bottom: 16px;">
            <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; color: #38bdf8; letter-spacing: 0.06em; margin-bottom: 6px;">
                🎯 RECOMMENDED NEXT BEST ANALYSIS
            </div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #edf2f7; margin-bottom: 4px;">
                {recommendation_title}
            </div>
            <div style="font-size: 0.86rem; color: #a0aec0; margin-bottom: 10px;">
                💡 <strong>Why investigate this:</strong> {reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )