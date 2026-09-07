"""
components/layout.py — Global SaaS layout header, sidebar controls, status rail, and metric row.
"""

from __future__ import annotations
import streamlit as st

from components import cards, state


# ============================================================
# TOP APPLICATION BANNER
# ============================================================

def render_top_banner(client, api_key: str | None) -> None:
    """Renders the compact top application header."""
    engine_status = '<span class="status-pill status-info">● Workspace Engine Active</span>'

    st.markdown(
        f"""
        <div class="top-app-banner">
            <div>
                <div class="top-app-title">
                    🧬 AI DATASET EXPLORER
                </div>
                <div class="top-app-subtitle">
                    Data Science & Intelligence Workspace
                </div>
            </div>
            <div class="top-status-pills">
                {engine_status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# THEME TOGGLE
# ============================================================

def render_theme_toggle() -> None:
    current = state.get_theme()
    label = "🌙 Dark" if current == "dark" else "☀️ Light"

    if st.sidebar.toggle(
        label,
        value=(current == "dark"),
        key="theme_toggle",
    ):
        state.set_theme("dark")
    else:
        state.set_theme("light")


# ============================================================
# REDESIGNED SIDEBAR & DATA INPUT PANEL
# ============================================================

def render_sidebar():
    """
    Renders the compact Data Scientist sidebar and returns:
    (current_file, previous_file, updated_file)
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom: 8px;">
                <div style="font-size: 1.05rem; font-weight: 800; letter-spacing: -0.02em; color: #f2f3f5;">
                    🧬 AI DATASET EXPLORER
                </div>
                <div style="font-size: 0.72rem; color: #9aa1b1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
                    DATA INTELLIGENCE WORKSPACE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_theme_toggle()

        st.divider()

        # Check if dataset is already uploaded
        current_file_session = st.session_state.get("current_dataset")
        has_file = current_file_session is not None or (state.get("loaded_file_name") or "") != ""

        # ----------------------------------------------------
        # DISTINCTIVE DATA INPUT UPLOAD PANEL
        # ----------------------------------------------------
        if not has_file:
            st.markdown(
                """
                <div class="data-input-panel-highlighted">
                    <div style="font-size: 0.72rem; font-weight: 800; text-transform: uppercase; color: #38bdf8; letter-spacing: 0.06em; margin-bottom: 4px;">
                        👇 START HERE — UPLOAD YOUR DATASET
                    </div>
                    <div class="data-input-header" style="font-size: 0.85rem; font-weight: 800; color: #ffffff; margin-bottom: 4px;">
                        📥 DATA INPUT PANEL
                    </div>
                    <div style="font-size: 0.76rem; color: #a0aec0;">
                        Upload a CSV or Excel file to begin analysis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="data-input-panel">
                    <div class="data-input-header">
                        📥 DATA INPUT PANEL
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        current_file = st.file_uploader(
            "Upload Active Dataset (CSV / Excel)",
            type=["csv", "xlsx"],
            key="current_dataset",
        )

        if current_file:
            st.markdown(
                f"""
                <div style="background: rgba(61, 220, 151, 0.12); border: 1px solid rgba(61, 220, 151, 0.3); border-radius: 8px; padding: 6px 10px; font-size: 0.76rem; margin-bottom: 8px;">
                    🟢 <strong>{current_file.name}</strong><br>
                    Size: <code>{round(current_file.size / 1024, 1)} KB</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ----------------------------------------------------
        # VERSION CONTROL / CHANGE INTELLIGENCE INPUTS
        # ----------------------------------------------------
        st.markdown(
            """
            <div style="font-size: 0.74rem; font-weight: 800; text-transform: uppercase; color: #38bdf8; letter-spacing: 0.05em; margin-bottom: 6px;">
                🔄 VERSION CONTROL COMPARISON
            </div>
            """,
            unsafe_allow_html=True,
        )

        previous_file = st.file_uploader(
            "Previous Version (V1) — Upload previous version",
            type=["csv", "xlsx"],
            key="previous_dataset",
        )

        updated_file = st.file_uploader(
            "Updated Version (V2) — Upload updated version",
            type=["csv", "xlsx"],
            key="updated_dataset",
        )

        if previous_file and updated_file:
            st.markdown(
                f"""
                <div style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 6px 10px; font-size: 0.74rem;">
                    ↔ <strong>V1:</strong> <code>{previous_file.name}</code><br>
                    ↔ <strong>V2:</strong> <code>{updated_file.name}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # ----------------------------------------------------
        # ANALYTICAL VERIFICATION RAIL
        # ----------------------------------------------------
        st.markdown(
            """
            <div style="font-size: 0.74rem; font-weight: 800; text-transform: uppercase; color: #a0aec0; letter-spacing: 0.05em; margin-bottom: 6px;">
                ⚙️ ANALYTICAL CHECKPOINTS
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "".join(
                cards.status_pill(label, "success")
                for label in [
                    "✓ Schema",
                    "✓ Stats",
                    "✓ Outliers",
                    "✓ Quality",
                    "✓ Visuals",
                    "✓ ML Readiness",
                    "✓ Drift",
                ]
            ),
            unsafe_allow_html=True,
        )

        st.divider()

    return current_file, previous_file, updated_file


# ============================================================
# STATUS ROW
# ============================================================

def render_status_row(
    client,
    ml_engine_available: bool,
    previous_file,
    updated_file,
) -> None:

    section_labels = [
        "Dataset",
        "Schema",
        "Quality",
        "Visuals",
        "AI Analyst",
        "ML Engine",
        "Drift Compare",
    ]

    completed_flags = [
        True,
        True,
        True,
        True,
        client is not None,
        ml_engine_available,
        (previous_file is not None and updated_file is not None),
    ]

    cols = st.columns(7)
    for col, label, completed in zip(cols, section_labels, completed_flags):
        with col:
            if completed:
                st.success(f"✓ {label}")
            else:
                st.warning(f"○ {label}")


# ============================================================
# METRIC ROW
# ============================================================

def render_metric_row(
    rows: int,
    columns: int,
    missing_cells: int,
    duplicate_rows: int,
) -> None:

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Records", f"{rows:,}")

    with c2:
        st.metric("Total Attributes", f"{columns:,}")

    with c3:
        missing_pct = ((missing_cells / (rows * columns)) * 100) if rows * columns else 0
        st.metric("Missing Cells", f"{missing_cells:,}", f"{missing_pct:.2f}%")

    with c4:
        duplicate_pct = ((duplicate_rows / rows) * 100) if rows else 0
        st.metric("Duplicate Rows", f"{duplicate_rows:,}", f"{duplicate_pct:.2f}%")