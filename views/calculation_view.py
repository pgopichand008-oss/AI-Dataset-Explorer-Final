"""
views/calculation_view.py — Interactive Calculation Lab View.

Allows users to select, execute, visualize, and compare statistical and mathematical calculations on their dataset.
"""

from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components import cards
from components.charts_theme import apply_chart_theme
from utils import prepare_safe_dataframe
from engine.calculation_engine import CATEGORIES, perform_calculation
from engine.visualization import get_graph_guide, create_3d_scatter


def render(
    df: pd.DataFrame,
    df_prev: pd.DataFrame | None = None,
    theme: str = "dark",
) -> None:
    cards.section_header(
        "🧮 Calculation Lab & Mathematical Explorer",
        "Choose custom calculations, inspect exact mathematical results, and explore dynamic visual representations.",
    )

    if df is None or df.empty:
        st.warning("Please upload a dataset to enable the Calculation Lab.")
        return

    # Prepare position-safe working dataframe & option maps
    safe_df, display_options, option_map, has_duplicates = prepare_safe_dataframe(df)

    if has_duplicates:
        st.warning(
            "⚠️ Duplicate column names detected in dataset. Position-based internal labels have been assigned for safe calculation and analysis. Your original dataset remains unchanged."
        )

    # Initialize calculation history in session state
    if "calc_history" not in st.session_state:
        st.session_state["calc_history"] = []

    numeric_opts = [opt for opt in display_options if pd.api.types.is_numeric_dtype(safe_df[option_map[opt]])]
    cat_opts = [opt for opt in display_options if not pd.api.types.is_numeric_dtype(safe_df[option_map[opt]])]

    # ---------------------------------------------------------
    # STEP 1: CATEGORY & OPERATION SELECTION
    # ---------------------------------------------------------
    col_cat, col_op = st.columns(2)
    with col_cat:
        cat_selected = st.selectbox("1️⃣ Select Calculation Category", list(CATEGORIES.keys()), key="calc_cat_select")
    with col_op:
        op_options = CATEGORIES.get(cat_selected, [])
        op_selected = st.selectbox("2️⃣ Select Specific Operation", op_options, key="calc_op_select")

    # ---------------------------------------------------------
    # STEP 2: DYNAMIC COLUMN INPUTS
    # ---------------------------------------------------------
    st.markdown("##### 3️⃣ Column Inputs & Parameters")

    primary_opt = None
    secondary_opt = None
    primary_col = None
    secondary_col = None

    if cat_selected in ["Basic Statistics", "Position / Spread", "Distribution", "Data Quality"]:
        if cat_selected == "Data Quality":
            primary_opt = st.selectbox("Select Attribute", display_options, key="calc_primary_all")
        else:
            if not numeric_opts:
                st.info("This category requires at least one numerical column.")
                return
            primary_opt = st.selectbox("Select Numerical Attribute", numeric_opts, key="calc_primary_num")

        if primary_opt:
            primary_col = option_map[primary_opt]

    elif cat_selected == "Categorical Analysis":
        if not cat_opts:
            st.info("This category requires at least one categorical column.")
            return
        primary_opt = st.selectbox("Select Categorical Attribute", cat_opts, key="calc_primary_cat")
        if primary_opt:
            primary_col = option_map[primary_opt]

    elif cat_selected == "Relationships":
        if op_selected in ["Pearson Correlation", "Covariance"]:
            if len(numeric_opts) < 2:
                st.info("Correlation requires at least two numerical columns.")
                return
            c1, c2 = st.columns(2)
            with c1:
                primary_opt = st.selectbox("Select Feature X (Numeric)", numeric_opts, index=0, key="calc_rel_x")
            with c2:
                secondary_opt = st.selectbox("Select Feature Y (Numeric)", numeric_opts, index=min(1, len(numeric_opts) - 1), key="calc_rel_y")
            if primary_opt and secondary_opt:
                primary_col = option_map[primary_opt]
                secondary_col = option_map[secondary_opt]

        elif op_selected == "Grouped Statistics":
            if not cat_opts or not numeric_opts:
                st.info("Grouped statistics require one Categorical column and one Numerical column.")
                return
            c1, c2 = st.columns(2)
            with c1:
                primary_opt = st.selectbox("Group By (Categorical)", cat_opts, key="calc_grp_cat")
            with c2:
                secondary_opt = st.selectbox("Measure Column (Numerical)", numeric_opts, key="calc_grp_num")
            if primary_opt and secondary_opt:
                primary_col = option_map[primary_opt]
                secondary_col = option_map[secondary_opt]

    elif cat_selected == "Dataset Comparison":
        if df_prev is None or df_prev.empty:
            st.info("Dataset Comparison requires both a Previous and an Updated dataset uploaded.")
            return

        if op_selected in ["Mean Change", "Median Change", "Missing-Value Change"]:
            common_cols = [c for c in df.columns if c in df_prev.columns]
            if not common_cols:
                st.info("No matching column names found between Previous and Updated datasets.")
                return
            primary_col = st.selectbox("Select Common Attribute to Compare", common_cols, key="calc_cmp_col")
            primary_opt = primary_col

    # ---------------------------------------------------------
    # STEP 3: PERFORM CALCULATION
    # ---------------------------------------------------------
    calc_btn = st.button("🧮 Run Calculation", type="primary", key="btn_run_calc")

    res = perform_calculation(
        safe_df,
        df_prev=df_prev,
        category=cat_selected,
        operation=op_selected,
        primary_col=primary_col,
        secondary_col=secondary_col,
    )

    if "error" in res:
        cards.finding_card("Unable to Perform Calculation", res["error"], "warning")
        return

    # Record in session history
    hist_entry = f"{op_selected} → {primary_opt or ''} {('vs ' + str(secondary_opt)) if secondary_opt else ''}"
    if hist_entry not in st.session_state["calc_history"]:
        st.session_state["calc_history"].insert(0, hist_entry)
        st.session_state["calc_history"] = st.session_state["calc_history"][:5]

    st.divider()

    # ---------------------------------------------------------
    # STEP 4: SIGNATURE CALCULATION → INSIGHT CHAIN PIPELINE
    # ---------------------------------------------------------
    cards.pipeline_card(
        step_calc=f"{cat_selected}: {op_selected}",
        step_result=res["formatted_result"],
        step_visual=res["chart_type"].replace("_", " ").title(),
        step_interp=res["explanation"][:55] + ("..." if len(res["explanation"]) > 55 else ""),
        step_action=f"Analyze {primary_opt or 'Attribute'} distribution",
    )

    st.divider()

    # ---------------------------------------------------------
    # STEP 5: RESULT PANEL DISPLAY
    # ---------------------------------------------------------
    cards.calculation_result_card(
        operation=res["operation"],
        result_fmt=res["formatted_result"],
        primary_col=primary_opt or "Dataset",
        valid_n=res["observations_used"],
        total_n=res["total_rows"],
        method=res["method"],
        explanation=res["explanation"],
    )

    # ---------------------------------------------------------
    # STEP 6: 2D CALCULATION VISUALIZATION
    # ---------------------------------------------------------
    st.markdown("### 📈 2D Visual Representation")
    fig = None
    guide_info = None

    try:
        if res["chart_type"] in ["mean_vs_median", "distribution"] and primary_col:
            series = pd.to_numeric(safe_df[primary_col], errors="coerce").dropna()
            if not series.empty:
                mean_val = float(series.mean())
                med_val = float(series.median())
                fig = px.histogram(safe_df, x=primary_col, title=f"Distribution of {primary_opt} with Mean & Median")
                fig.add_vline(x=mean_val, line_dash="dash", line_color="#ef4444", annotation_text=f"Mean: {mean_val:.2f}")
                fig.add_vline(x=med_val, line_dash="dot", line_color="#10b981", annotation_text=f"Median: {med_val:.2f}")
                guide_info = get_graph_guide("Histogram", primary_opt or primary_col, "Frequency")

        elif res["chart_type"] == "box" and primary_col:
            fig = px.box(safe_df, y=primary_col, title=f"Boxplot & Quantiles of {primary_opt}")
            guide_info = get_graph_guide("Boxplot", "", primary_opt or primary_col)

        elif res["chart_type"] == "scatter" and primary_col and secondary_col:
            clean_pair = safe_df[[primary_col, secondary_col]].dropna()
            fig = px.scatter(
                clean_pair,
                x=primary_col,
                y=secondary_col,
                trendline="ols",
                title=f"{op_selected}: {primary_opt} vs {secondary_opt}",
            )
            guide_info = get_graph_guide("Scatter", primary_opt or primary_col, secondary_opt or secondary_col)

        elif res["chart_type"] == "grouped_bar" and "grouped_df" in res:
            grp_df = res["grouped_df"]
            fig = px.bar(
                grp_df,
                x="Group",
                y="mean",
                title=f"Mean {secondary_opt} by {primary_opt}",
                labels={"mean": f"Mean {secondary_opt}", "Group": primary_opt},
            )
            guide_info = get_graph_guide("Bar Chart", primary_opt or primary_col, f"Mean {secondary_opt}")

        elif res["chart_type"] == "bar" and primary_col:
            counts = safe_df[primary_col].dropna().astype(str).value_counts().head(10).reset_index()
            counts.columns = ["Category", "Count"]
            fig = px.bar(counts, x="Category", y="Count", title=f"Category Distribution: {primary_opt}")
            guide_info = get_graph_guide("Bar Chart", primary_opt or primary_col, "Count")

        elif res["chart_type"] == "comparison" and "val_prev" in res:
            comp_df = pd.DataFrame({
                "Version": ["Previous Dataset", "Updated Dataset"],
                "Value": [res["val_prev"], res["val_curr"]],
            })
            fig = px.bar(comp_df, x="Version", y="Value", color="Version", title=f"Dataset Comparison — {op_selected}")
            guide_info = get_graph_guide("Bar Chart", "Dataset Version", "Metric Value")
    except Exception:
        fig = None

    if fig is not None:
        st.plotly_chart(apply_chart_theme(fig, theme), use_container_width=True)
        if guide_info:
            cards.graph_guide_card(guide_info)
    else:
        st.info("ℹ️ 2D visual representation not applicable for this summary metric.")

    # ---------------------------------------------------------
    # STEP 7: 3D VISUAL APPLICABILITY & MODELING
    # ---------------------------------------------------------
    st.markdown("### 🧊 3D Visual Representation")
    if len(numeric_opts) >= 3 and primary_col and primary_col in [option_map[o] for o in numeric_opts]:
        other_num = [o for o in numeric_opts if option_map[o] != primary_col]
        z_3d_opt = other_num[0] if other_num else None
        y_3d_opt = secondary_opt if secondary_opt and secondary_opt in numeric_opts else (other_num[1] if len(other_num) > 1 else None)

        if z_3d_opt and y_3d_opt:
            try:
                fig_3d, stats_3d = create_3d_scatter(
                    safe_df,
                    x_col=primary_col,
                    y_col=option_map[y_3d_opt],
                    z_col=option_map[z_3d_opt],
                )
                if fig_3d is not None:
                    st.plotly_chart(apply_chart_theme(fig_3d, theme, height=550), use_container_width=True)
                    guide_3d = get_graph_guide("3D Scatter", primary_opt or primary_col, y_3d_opt, z_3d_opt)
                    cards.graph_guide_card(guide_3d, valid_n=stats_3d.get("valid_rows"), excl_n=stats_3d.get("excluded_rows"))
                else:
                    st.info("ℹ️ 3D visualization unavailable: Insufficient valid numeric observations across 3D attributes.")
            except Exception:
                st.info("ℹ️ 3D visualization not applicable for this calculation.")
        else:
            st.info("ℹ️ 3D visualization unavailable: Requires at least three compatible numeric columns.")
    else:
        st.info(
            "ℹ️ 3D visualization not applicable: This calculation produces a single summary value, so a 3D spatial representation would not add extra analytical information."
        )

    # ---------------------------------------------------------
    # STEP 8: METRIC COMPARISON TOOL (e.g. Mean vs Median)
    # ---------------------------------------------------------
    if primary_col and primary_opt in numeric_opts:
        st.divider()
        st.markdown("### ⚖️ Compare Mean vs. Median (Skewness Check)")
        series = pd.to_numeric(safe_df[primary_col], errors="coerce").dropna()
        if not series.empty:
            mean_v = float(series.mean())
            med_v = float(series.median())
            diff = mean_v - med_v
            pct_diff = (diff / med_v * 100) if med_v != 0 else 0.0

            if abs(pct_diff) < 2.0:
                comp_text = f"Mean ({mean_v:,.2f}) and Median ({med_v:,.2f}) are nearly identical. Distribution is symmetric."
            elif pct_diff > 0:
                comp_text = f"Mean ({mean_v:,.2f}) is higher than Median ({med_v:,.2f}) by +{pct_diff:.1f}%. Indicates right-skewness (positive tail)."
            else:
                comp_text = f"Mean ({mean_v:,.2f}) is lower than Median ({med_v:,.2f}) by {pct_diff:.1f}%. Indicates left-skewness (negative tail)."

            cards.compare_metrics_card(
                metric1_name="Arithmetic Mean",
                metric1_val=f"{mean_v:,.2f}",
                metric2_name="Median (50th Percentile)",
                metric2_val=f"{med_v:,.2f}",
                comparison_text=comp_text,
            )

    # ---------------------------------------------------------
    # STEP 9: RECENT CALCULATION HISTORY
    # ---------------------------------------------------------
    if st.session_state.get("calc_history"):
        st.markdown("##### 📜 Recent Calculation History")
        st.caption("Recent calculations performed during this session:")
        for h in st.session_state["calc_history"]:
            st.markdown(f"• `{h}`")
