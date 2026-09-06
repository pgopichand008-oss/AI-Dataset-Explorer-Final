"""
components/layout.py

Global chrome: theme toggle, sidebar workspace controls,
pipeline status rail, and the top-line metric row.
"""

from __future__ import annotations

import streamlit as st

from components import cards, state


# ============================================================
# THEME TOGGLE
# ============================================================

def render_theme_toggle() -> None:
    current = state.get_theme()

    label = (
        "🌙 Dark mode"
        if current == "dark"
        else "☀️ Light mode"
    )

    if st.sidebar.toggle(
        label,
        value=(current == "dark"),
        key="theme_toggle",
    ):
        state.set_theme("dark")
    else:
        state.set_theme("light")


# ============================================================
# HERO HEADER
# ============================================================

def render_hero() -> None:
    """
    Global hero header disabled.

    The application can still call render_hero()
    without displaying the old hero section.
    """
    return


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """
    Renders the sidebar and returns:

    (current_file, previous_file, updated_file)
    """

    with st.sidebar:

        # ----------------------------------------------------
        # WORKSPACE
        # ----------------------------------------------------

        st.markdown("## Workspace")

        render_theme_toggle()

        st.divider()

        # ----------------------------------------------------
        # CURRENT DATASET
        # ----------------------------------------------------

        st.markdown("### Dataset")

        current_file = st.file_uploader(
            "Upload current dataset",
            type=["csv", "xlsx"],
            key="current_dataset",
        )

        st.caption(
            "CSV or Excel — activates the full analysis pipeline."
        )

        st.divider()

        # ----------------------------------------------------
        # CHANGE INTELLIGENCE
        # ----------------------------------------------------

        st.markdown("### Change Intelligence")

        previous_file = st.file_uploader(
            "Previous dataset",
            type=["csv", "xlsx"],
            key="previous_dataset",
        )

        updated_file = st.file_uploader(
            "Updated dataset",
            type=["csv", "xlsx"],
            key="updated_dataset",
        )

        st.caption(
            "Compare two versions for structural and quality drift."
        )

        st.divider()

        # ----------------------------------------------------
        # PIPELINE
        # ----------------------------------------------------

        st.markdown("### Pipeline")

        st.markdown(
            "".join(
                cards.status_pill(
                    label,
                    "success",
                )
                for label in [
                    "① Discovery",
                    "② Profiling",
                    "③ Quality",
                    "④ Visuals",
                    "⑤ AI",
                    "⑥ ML",
                    "⑦ Changes",
                ]
            ),
            unsafe_allow_html=True,
        )

        st.divider()

    return (
        current_file,
        previous_file,
        updated_file,
    )


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
        "Profile",
        "Quality",
        "Visuals",
        "AI",
        "ML",
        "Changes",
    ]

    completed_flags = [
        True,
        True,
        True,
        True,
        client is not None,
        ml_engine_available,
        (
            previous_file is not None
            and updated_file is not None
        ),
    ]

    cols = st.columns(7)

    for col, label, completed in zip(
        cols,
        section_labels,
        completed_flags,
    ):

        with col:

            if completed:

                st.success(
                    f"✓ {label}"
                )

            else:

                st.warning(
                    f"○ {label}"
                )


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

    # --------------------------------------------------------
    # RECORDS
    # --------------------------------------------------------

    with c1:

        st.metric(
            "Records",
            f"{rows:,}",
        )

    # --------------------------------------------------------
    # ATTRIBUTES
    # --------------------------------------------------------

    with c2:

        st.metric(
            "Attributes",
            f"{columns:,}",
        )

    # --------------------------------------------------------
    # MISSING CELLS
    # --------------------------------------------------------

    with c3:

        missing_pct = (
            (
                missing_cells
                / (rows * columns)
            ) * 100
            if rows * columns
            else 0
        )

        st.metric(
            "Missing Cells",
            f"{missing_cells:,}",
            f"{missing_pct:.2f}%",
        )

    # --------------------------------------------------------
    # DUPLICATE ROWS
    # --------------------------------------------------------

    with c4:

        duplicate_pct = (
            (
                duplicate_rows
                / rows
            ) * 100
            if rows
            else 0
        )

        st.metric(
            "Duplicate Rows",
            f"{duplicate_rows:,}",
            f"{duplicate_pct:.2f}%",
        )