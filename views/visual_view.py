"""
views/visual_view.py — Deep visual exploration, themed uniformly, with data-driven 3D Data Space and Graph Guides.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st

from components import cards
from components.charts_theme import apply_chart_theme
from utils import prepare_safe_dataframe, validate_chart_inputs
from engine.visualization import (
    create_numerical_histogram,
    create_numerical_boxplot,
    create_scatter_matrix,
    create_categorical_frequency,
    create_missing_value_chart,
    create_3d_scatter,
    create_3d_surface,
    create_3d_mesh,
    recommend_visualizations,
    get_graph_guide,
)


def render(
    df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    categorical_columns: list,
    date_columns: list,
    correlation_matrix: pd.DataFrame,
    correlation_pairs: pd.DataFrame,
    missing_cells: int,
    theme: str = "dark",
) -> None:

    cards.section_header(
        "📈 Deep Visual Exploration & 3D Feature Studio",
        "Distributions, relationships, intelligent chart recommendations, 3D feature space, and graph guides.",
    )

    if df is None or df.empty:
        st.warning("Please upload a dataset to activate visual exploration.")
        return

    # Prepare position-safe working dataframe & option maps
    safe_df, display_options, option_map, has_duplicates = prepare_safe_dataframe(df)

    if has_duplicates:
        st.warning(
            "⚠️ Duplicate column names detected in dataset. Position-based internal labels have been assigned for safe visualization and analysis. Your original dataset remains unchanged."
        )

    # Filter numeric & categorical options safely
    num_options = [opt for opt in display_options if pd.api.types.is_numeric_dtype(safe_df[option_map[opt]])]

    # --------------------------------------------------------
    # AUTOMATIC VISUALIZATION RECOMMENDATIONS
    # --------------------------------------------------------
    recs = recommend_visualizations(df)
    if recs:
        st.markdown("### 💡 Recommended Visualizations")
        r_cols = st.columns(min(len(recs), 3))
        for col, rec in zip(r_cols, recs[:3]):
            with col:
                cards.finding_card(
                    f"{rec['badge']} — {rec['title']}",
                    f"**Why:** {rec['reason']}",
                    "info",
                )

    # --------------------------------------------------------
    # 3D DATA SPACE EXPLORATION (MANDATORY REQUIREMENT)
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 🧊 3D Feature Explorer")

    if len(num_options) < 3:
        st.info(
            "ℹ️ 3D visualization unavailable: Three distinct numerical columns are required for 3D spatial feature exploration."
        )
    else:
        st.caption("Interactively explore 3D spatial patterns using real dataset variables.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            x_opt = st.selectbox("X Axis (3D)", num_options, index=0, key="3d_x")
        with c2:
            y_opt = st.selectbox("Y Axis (3D)", num_options, index=min(1, len(num_options) - 1), key="3d_y")
        with c3:
            z_opt = st.selectbox("Z Axis (3D)", num_options, index=min(2, len(num_options) - 1), key="3d_z")
        with c4:
            chart_type = st.selectbox("3D Model Type", ["3D Scatter", "3D Surface", "3D Mesh"], key="3d_type")

        color_options = ["None"] + display_options
        color_opt_sel = st.selectbox("Color / Grouping Attribute (Optional)", color_options, index=0, key="3d_color")
        color_col = None if color_opt_sel == "None" else option_map[color_opt_sel]

        x_col = option_map[x_opt]
        y_col = option_map[y_opt]
        z_col = option_map[z_opt]

        # Axis validation before plotting
        is_valid_3d, msg_3d = validate_chart_inputs(safe_df, x_col, y_col, z_col, chart_type)

        if not is_valid_3d:
            cards.finding_card("3D Visualization Guard", msg_3d, "warning")
        else:
            fig = None
            meta = {}

            try:
                if chart_type == "3D Scatter":
                    fig, meta = create_3d_scatter(safe_df, x_col, y_col, z_col, color_col=color_col)
                elif chart_type == "3D Surface":
                    fig, meta = create_3d_surface(safe_df, x_col, y_col, z_col)
                elif chart_type == "3D Mesh":
                    fig, meta = create_3d_mesh(safe_df, x_col, y_col, z_col)
            except Exception:
                fig = None

            if fig is not None:
                valid_n = meta.get("valid_rows", 0)
                excl_n = meta.get("excluded_rows", 0)

                st.plotly_chart(apply_chart_theme(fig, theme, height=650), use_container_width=True)

                cards.finding_card(
                    "3D Data Accuracy & Observation Filter",
                    f"This 3D plot visualizes **{valid_n:,} valid observations** using actual dataset values "
                    f"({excl_n:,} observations excluded due to invalid/missing numbers).",
                    "info",
                )

                # 3D GRAPH GUIDE
                guide = get_graph_guide(chart_type, x_opt, y_opt, z_opt, color_opt_sel if color_col else None)
                cards.graph_guide_card(guide, valid_n=valid_n, excl_n=excl_n)
            else:
                cards.finding_card(
                    "Visualization Guard",
                    "⚠️ We couldn't create this 3D visualization. The selected columns contain insufficient valid numeric observations.",
                    "warning",
                )

    # --------------------------------------------------------
    # 2D NUMERICAL DISTRIBUTIONS
    # --------------------------------------------------------
    if num_options:
        st.divider()
        st.markdown("### 📊 Numerical Distribution & Stats")
        selected_opt = st.selectbox("Select numerical attribute", num_options, key="visual_numeric")
        selected_col = option_map[selected_opt]
        series = safe_df[selected_col].dropna()

        c1, c2 = st.columns(2)
        with c1:
            try:
                fig = apply_chart_theme(create_numerical_histogram(safe_df, selected_col), theme)
                st.plotly_chart(fig, use_container_width=True)
                cards.graph_guide_card(get_graph_guide("Histogram", selected_opt, "Frequency"))
            except Exception:
                st.warning("Unable to generate histogram for selected attribute.")

        with c2:
            try:
                fig = apply_chart_theme(create_numerical_boxplot(safe_df, selected_col), theme)
                st.plotly_chart(fig, use_container_width=True)
                cards.graph_guide_card(get_graph_guide("Boxplot", "", selected_opt))
            except Exception:
                st.warning("Unable to generate boxplot for selected attribute.")

        if not series.empty:
            stat_cols = st.columns(5)
            stats = [
                ("Mean", series.mean()),
                ("Median", series.median()),
                ("Min", series.min()),
                ("Max", series.max()),
                ("Std Dev", series.std()),
            ]
            for col, (label, value) in zip(stat_cols, stats):
                with col:
                    if pd.notna(value):
                        st.metric(label, f"{value:.3f}")

    # --------------------------------------------------------
    # CORRELATION EXPLORER
    # --------------------------------------------------------
    if len(num_options) >= 2:
        st.divider()
        st.markdown("### 🔗 Correlation Explorer")
        threshold = st.slider("Minimum absolute correlation", 0.0, 1.0, 0.50, 0.05, key="correlation_threshold")

        if not correlation_matrix.empty:
            try:
                filtered = correlation_matrix.mask(correlation_matrix.abs() < threshold)
                fig = px.imshow(
                    filtered,
                    text_auto=".2f",
                    aspect="auto",
                    zmin=-1,
                    zmax=1,
                    title=f"Relationships with |correlation| ≥ {threshold:.2f}",
                )
                st.plotly_chart(apply_chart_theme(fig, theme, height=550), use_container_width=True)
                cards.graph_guide_card(get_graph_guide("Heatmap", "Feature 1", "Feature 2"))
            except Exception:
                st.warning("Unable to render correlation matrix visualization.")

        if not correlation_pairs.empty:
            st.markdown("**Strongest Numerical Relationships**")
            st.dataframe(
                correlation_pairs[["Attribute 1", "Attribute 2", "Correlation"]].head(10),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("#### Multidimensional Explorer")
        default_opts = num_options[: min(4, len(num_options))]
        matrix_opts = st.multiselect(
            "Select 2–5 numerical attributes",
            num_options,
            default=default_opts,
            max_selections=5,
            key="matrix_columns",
        )
        if len(matrix_opts) >= 2:
            matrix_cols = [option_map[o] for o in matrix_opts]
            matrix_df = safe_df[matrix_cols].dropna()
            if len(matrix_df) > 1000:
                matrix_df = matrix_df.sample(1000, random_state=42)
            try:
                fig = apply_chart_theme(create_scatter_matrix(matrix_df, matrix_cols), theme, height=650)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Unable to render multidimensional scatter matrix.")

    # --------------------------------------------------------
    # CATEGORICAL EXPLORATION
    # --------------------------------------------------------
    if display_options:
        cat_opts = [opt for opt in display_options if option_map[opt] in categorical_columns]
        if cat_opts:
            st.divider()
            st.markdown("### 🏷️ Categorical Exploration")
            selected_cat_opt = st.selectbox("Select categorical attribute", cat_opts, key="visual_category")
            selected_cat_col = option_map[selected_cat_opt]
            try:
                fig = apply_chart_theme(create_categorical_frequency(safe_df, selected_cat_col), theme, height=450)
                st.plotly_chart(fig, use_container_width=True)
                cards.graph_guide_card(get_graph_guide("Bar Chart", selected_cat_opt, "Frequency"))
            except Exception:
                st.warning("Unable to render categorical frequency chart.")

    # --------------------------------------------------------
    # TEMPORAL EXPLORATION
    # --------------------------------------------------------
    if date_columns:
        st.divider()
        st.markdown("### 📅 Temporal Exploration")
        selected_date = st.selectbox("Select date/time attribute", date_columns, key="visual_date")
        try:
            date_series = pd.to_datetime(safe_df[selected_date], errors="coerce")
            temporal_df = pd.DataFrame({selected_date: date_series}).dropna()
            if not temporal_df.empty:
                temporal_df["Period"] = temporal_df[selected_date].dt.to_period("M").astype(str)
                trend_df = temporal_df.groupby("Period").size().reset_index(name="Records")
                if len(trend_df) > 1:
                    fig = px.line(trend_df, x="Period", y="Records", markers=True, title=f"Record Trend — {selected_date}")
                    st.plotly_chart(apply_chart_theme(fig, theme), use_container_width=True)
                    cards.graph_guide_card(get_graph_guide("Line Chart", selected_date, "Record Count"))
        except Exception:
            st.warning("Unable to render temporal trend chart.")

    # --------------------------------------------------------
    # MISSING DATA PATTERN
    # --------------------------------------------------------
    if missing_cells > 0:
        st.divider()
        st.markdown("### ⚠️ Missing Data Pattern")
        try:
            fig = create_missing_value_chart(safe_df)
            if fig is not None:
                st.plotly_chart(apply_chart_theme(fig, theme), use_container_width=True)
                cards.graph_guide_card(get_graph_guide("Bar Chart", "Attribute", "Missing Count"))
        except Exception:
            st.warning("Unable to render missing value chart.")