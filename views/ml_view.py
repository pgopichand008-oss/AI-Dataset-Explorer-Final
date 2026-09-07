"""views/ml_view.py — Machine learning workspace."""

from __future__ import annotations

import inspect
import re
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from components import cards, state
from components.charts_theme import apply_chart_theme


def _clean_text_artifacts(text: Any) -> str:
    """Sanitizes text by removing HTML/SVG tags, canvas tags, and localhost anchors."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # Remove HTML/SVG elements
    text = re.sub(r'<[^>]*>', '', text)
    # Remove markdown link anchors to localhost/svg/canvas
    text = re.sub(r'\[(?:svg|canvas|link)\]\(http://localhost:\d+/[^\)]*\)', '', text)
    text = re.sub(r'\[.*?\]\(http://localhost:\d+/[^\)]*\)', '', text)
    # Remove canvas duplication artifacts
    text = re.sub(r'canvas+', '', text)
    return text.strip()


def _find_kwargs(fn, df, target) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    kwargs = {}

    for name in params:
        lowered = name.lower()

        if lowered in ["df", "data", "dataset", "dataframe"]:
            kwargs[name] = df

        elif lowered in [
            "target",
            "target_column",
            "target_col",
            "target_name",
            "y_column",
            "label",
        ]:
            kwargs[name] = target

    return kwargs


def call_ml_engine(run_ml_analysis, df: pd.DataFrame, target: str) -> Any:
    kwargs = _find_kwargs(run_ml_analysis, df, target)

    try:
        return run_ml_analysis(**kwargs) if kwargs else run_ml_analysis(df, target)
    except TypeError:
        return run_ml_analysis(df, target)


def _extract(result: dict, keys: list[str], default=None):
    if not isinstance(result, dict):
        return default
    for key in keys:
        if key in result:
            return result[key]

    return default


def display_ml_result(
    result: Any,
    build_ml_summary_fn,
    theme: str,
) -> None:

    if result is None:
        return

    if not isinstance(result, dict):
        cards.finding_card("ML Model Output", _clean_text_artifacts(result), "info")
        return

    try:
        summary = (
            build_ml_summary_fn(result)
            if build_ml_summary_fn
            else None
        )
    except Exception:
        summary = None

    # --------------------------------------------------------
    # 📋 ML SUMMARY TABLE
    # --------------------------------------------------------
    st.subheader("📋 ML Summary")

    summary_rows = []

    problem_type = _extract(result, ["problem_type", "task_type", "task", "problem"])
    if problem_type:
        summary_rows.append({"Attribute": "Task / Problem Type", "Value": _clean_text_artifacts(problem_type)})

    best_model = _extract(result, ["best_model", "selected_model", "best_model_name"])
    if best_model:
        summary_rows.append({"Attribute": "Selected Model", "Value": _clean_text_artifacts(best_model)})

    target_col = _extract(result, ["target", "target_column", "target_col", "selected_target"])
    if target_col:
        summary_rows.append({"Attribute": "Target Variable", "Value": _clean_text_artifacts(target_col)})

    n_rows = _extract(result, ["n_rows", "row_count", "samples"])
    if n_rows is not None:
        summary_rows.append({"Attribute": "Dataset Rows", "Value": f"{n_rows:,}"})

    features_used = _extract(result, ["features_used", "feature_count", "n_features"])
    if features_used is not None:
        summary_rows.append({"Attribute": "Features Used", "Value": str(features_used)})

    if summary and isinstance(summary, dict):
        for k, v in summary.items():
            if not any(r["Attribute"].lower() == str(k).lower() for r in summary_rows):
                summary_rows.append({"Attribute": _clean_text_artifacts(k), "Value": _clean_text_artifacts(v)})

    if summary_rows:
        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            hide_index=True,
        )
    elif summary and isinstance(summary, str):
        cards.finding_card("Summary", _clean_text_artifacts(summary), "info")

    # --------------------------------------------------------
    # 📈 ML READINESS
    # --------------------------------------------------------
    readiness = _extract(
        result,
        ["ml_readiness", "readiness", "readiness_score", "score"],
    )

    if readiness is not None:
        try:
            val = max(0, min(100, float(readiness)))
            st.subheader("📈 ML Readiness Assessment")
            st.progress(val / 100)
            st.metric("ML Readiness Score", f"{val:.0f}/100")
        except Exception:
            pass

    # --------------------------------------------------------
    # ⚠️ WARNINGS
    # --------------------------------------------------------
    warnings = _extract(
        result,
        ["warnings", "limitations", "issues"],
        [],
    )

    if warnings:
        st.subheader("⚠️ ML Warnings & Constraints")
        items = warnings if isinstance(warnings, list) else [warnings]
        for warning in items:
            cards.finding_card("Warning", _clean_text_artifacts(warning), "warning")

    # --------------------------------------------------------
    # 🎯 MODEL PERFORMANCE METRICS
    # --------------------------------------------------------
    metrics = _extract(
        result,
        ["metrics", "best_metrics", "evaluation", "performance"],
    )

    if metrics is not None:
        st.subheader("🎯 Model Performance")

        if isinstance(metrics, dict):
            cols = st.columns(min(4, max(1, len(metrics))))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i % len(cols)]:
                    if isinstance(value, float):
                        display_value = f"{value:.4f}"
                    else:
                        display_value = _clean_text_artifacts(value)
                    st.metric(_clean_text_artifacts(key), display_value)

        elif isinstance(metrics, pd.DataFrame):
            st.dataframe(metrics, use_container_width=True, hide_index=True)

        else:
            cards.finding_card("Performance Metrics", _clean_text_artifacts(metrics), "info")

    # --------------------------------------------------------
    # 📊 MODEL COMPARISON
    # --------------------------------------------------------
    model_comparison = _extract(
        result,
        ["model_comparison", "model_results", "models", "comparison"],
    )

    if model_comparison is not None:
        st.subheader("📊 Candidate Model Comparison")

        if isinstance(model_comparison, pd.DataFrame):
            comp_df = model_comparison
        elif isinstance(model_comparison, list):
            comp_df = pd.DataFrame(model_comparison)
        elif isinstance(model_comparison, dict):
            try:
                comp_df = pd.DataFrame(model_comparison)
            except Exception:
                comp_df = pd.DataFrame(
                    [{"Model": k, "Result": v} for k, v in model_comparison.items()]
                )
        else:
            comp_df = pd.DataFrame()

        if not comp_df.empty:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # 📊 FEATURE IMPORTANCE
    # --------------------------------------------------------
    feature_importance = _extract(
        result,
        ["feature_importance", "feature_importances", "importance"],
    )

    if feature_importance is not None:
        st.subheader("📊 Feature Importance")

        if isinstance(feature_importance, pd.DataFrame):
            imp_df = feature_importance.copy()
        elif isinstance(feature_importance, list):
            imp_df = pd.DataFrame(feature_importance)
        elif isinstance(feature_importance, dict):
            imp_df = pd.DataFrame(
                [{"Feature": k, "Importance": v} for k, v in feature_importance.items()]
            )
        else:
            imp_df = pd.DataFrame()

        if not imp_df.empty:
            numeric_cols = imp_df.select_dtypes(include="number").columns.tolist()
            label_col = next(
                (col for col in ["Feature", "feature", "Attribute", "attribute", "feature_name"] if col in imp_df.columns),
                None,
            )

            if numeric_cols and label_col:
                chart_df = (
                    imp_df[[label_col, numeric_cols[-1]]]
                    .dropna()
                    .sort_values(numeric_cols[-1])
                    .tail(15)
                )

                if not chart_df.empty:
                    fig = px.bar(
                        chart_df,
                        x=numeric_cols[-1],
                        y=label_col,
                        orientation="h",
                        title="Top Feature Importance",
                    )
                    st.plotly_chart(
                        apply_chart_theme(fig, theme, height=450),
                        use_container_width=True,
                    )

            st.dataframe(imp_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # 🔮 PREDICTION SAMPLES
    # --------------------------------------------------------
    predictions = _extract(
        result,
        ["predictions", "prediction_samples", "sample_predictions"],
    )

    if predictions is not None:
        st.subheader("🔮 Prediction Samples")

        if isinstance(predictions, pd.DataFrame):
            pred_df = predictions
        else:
            pred_df = pd.DataFrame(predictions or [])

        if not pred_df.empty:
            st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # 🧠 MODEL INSIGHTS & RELIABILITY
    # --------------------------------------------------------
    reliability = _extract(
        result,
        ["reliability", "ml_reliability", "reliability_assessment"],
    )
    recommendation = _extract(
        result,
        ["final_recommendation", "recommendation", "ml_recommendation"],
    )

    if reliability or recommendation:
        st.subheader("🧠 Model Insights & Reliability")

        if reliability:
            if isinstance(reliability, dict):
                st.dataframe(
                    pd.DataFrame(
                        [{"Assessment": k, "Result": _clean_text_artifacts(v)} for k, v in reliability.items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                cards.finding_card("Reliability Assessment", _clean_text_artifacts(reliability), "info")

        if recommendation:
            rec_text = _clean_text_artifacts(recommendation)
            if "NOT SUITABLE" in rec_text.upper():
                kind = "danger"
            elif "NEEDS ATTENTION" in rec_text.upper():
                kind = "warning"
            else:
                kind = "success"

            cards.finding_card("Recommendation", rec_text, kind)


def render(
    df: pd.DataFrame,
    uploaded_file,
    rows: int,
    columns: int,
    ml_score: int,
    target_candidates: list[str],
    ml_engine_available: bool,
    ml_engine_error: str,
    run_ml_analysis_fn,
    build_ml_summary_fn,
    theme: str,
) -> None:

    cards.section_header(
        "Machine Learning Intelligence",
        "Evaluate readiness, identify a target, train candidate models, and explain the result.",
    )

    if not ml_engine_available:
        cards.finding_card(
            "ML engine unavailable",
            f"engine/ml_engine.py could not be loaded: {_clean_text_artifacts(ml_engine_error)}",
            "danger",
        )
        return

    # Top metrics
    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("ML Readiness", f"{ml_score}/100")
    with top_cols[1]:
        st.metric("Rows", f"{rows:,}")
    with top_cols[2]:
        st.metric("Features", f"{max(0, columns - 1):,}")
    with top_cols[3]:
        st.metric("Target Candidates", f"{len(target_candidates):,}")

    st.divider()

    # Target Selection
    st.subheader("🎯 Target Selection")

    if target_candidates:
        cards.finding_card(
            "Auto-detected",
            "Possible target variables were detected. First candidate selected by default.",
            "info",
        )
    else:
        cards.finding_card(
            "No target detected",
            "Select a target column manually.",
            "warning",
        )

    target_mode = st.radio(
        "Target strategy",
        ["Automatic", "Manual"],
        horizontal=True,
        key="ml_target_mode",
    )

    selected_target: Optional[str] = None

    if target_mode == "Automatic":
        selected_target = target_candidates[0] if target_candidates else None
        if selected_target is None:
            cards.finding_card(
                "Cannot continue",
                "Automatic mode needs a detected target candidate.",
                "warning",
            )
    else:
        selected_target = st.selectbox(
            "Choose target column",
            list(df.columns),
            key="manual_ml_target",
        )

    if selected_target:
        cards.ml_card("Selected Target", selected_target)

    if ml_score < 40:
        cards.finding_card(
            "Low ML readiness",
            "The agent can still run, but results may be unreliable.",
            "warning",
        )

    if rows < 10:
        cards.finding_card(
            "Very few rows",
            "Model evaluation may be unstable.",
            "warning",
        )

    run_ml = st.button(
        "Run Intelligent ML Analysis",
        type="primary",
        disabled=selected_target is None,
        key="run_ml_button",
    )

    if run_ml and selected_target:
        with st.spinner("ML agent is preparing data, training, and evaluating models..."):
            try:
                result = call_ml_engine(
                    run_ml_analysis_fn,
                    df,
                    selected_target,
                )
                state.set_ml_result(
                    result,
                    uploaded_file.name,
                    selected_target,
                )
                st.success("Machine learning analysis completed.")
            except Exception as e:
                state.set_ml_result(
                    None,
                    uploaded_file.name,
                    selected_target,
                )
                cards.finding_card(
                    "Analysis failed",
                    _clean_text_artifacts(str(e)),
                    "danger",
                )

    result = state.get_ml_result()

    if (
        result is not None
        and state.get("ml_file_name") == uploaded_file.name
    ):
        st.divider()

        st.subheader("🤖 Machine Learning Results")

        display_ml_result(
            result,
            build_ml_summary_fn,
            theme,
        )

        st.divider()

        cards.finding_card(
            "🔒 Dataset Protection",
            "Your original dataset is protected. ML analysis is performed on an in-memory copy, so the uploaded file remains unchanged.",
            "info",
        )
