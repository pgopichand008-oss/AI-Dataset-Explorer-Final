import os
import inspect
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import get_gemini_client
from data_loader import load_dataset

from engine.profiling import (
    build_basic_profile,
    build_column_intelligence,
    build_numerical_summary,
    build_categorical_summary
)

from engine.quality import (
    detect_outliers,
    build_quality_findings,
    calculate_quality_score,
    calculate_ml_readiness
)

from engine.intelligence import (
    build_agent_decisions,
    build_visualization_plan,
    build_executive_summary
)

from engine.visualization import (
    create_numerical_histogram,
    create_numerical_boxplot,
    create_correlation_heatmap,
    create_scatter_matrix,
    create_categorical_frequency,
    create_temporal_trend,
    create_missing_value_chart,
    create_binary_class_balance
)

from engine.ai_report import generate_ai_report

from ui import (
    show_metric_cards,
    show_quality_score,
    show_ml_readiness,
    show_quality_findings,
    show_agent_decisions,
    show_executive_summary,
    show_column_table,
    show_data_preview,
    show_ai_report
)

from engine.ai_agent import run_intelligence_analysis


# ============================================================
# OPTIONAL ML ENGINE
# ============================================================

try:
    from engine.ml_engine import run_ml_analysis, build_ml_summary
    ML_ENGINE_AVAILABLE = True
    ML_ENGINE_ERROR = ""
except Exception as e:
    ML_ENGINE_AVAILABLE = False
    ML_ENGINE_ERROR = str(e)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Dataset Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99, 102, 241, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14, 165, 233, 0.07),
                transparent 28%
            ),
            #f8fafc;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .hero-card {
        padding: 28px 32px;
        border-radius: 22px;
        margin-bottom: 25px;
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.98),
                rgba(30, 41, 59, 0.96)
            );
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.16);
    }

    .hero-title {
        color: white;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #e2e8f0;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 12px;
        border: 1px solid rgba(255,255,255,0.10);
    }

    /* --------------------------------------------------------
       CARDS
    -------------------------------------------------------- */

    .info-card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(255,255,255,0.86);
        border: 1px solid #e2e8f0;
        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.06);
        margin-bottom: 15px;
    }

    .info-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 5px;
    }

    .info-card-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
    }

    .info-card-description {
        font-size: 0.84rem;
        color: #64748b;
        margin-top: 4px;
    }

    .ml-card {
        padding: 24px;
        border-radius: 20px;
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.98),
                rgba(51, 65, 85, 0.96)
            );
        color: white;
        box-shadow:
            0 15px 35px rgba(15, 23, 42, 0.15);
        margin-bottom: 18px;
    }

    .ml-card-title {
        font-size: 1.15rem;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .ml-card-text {
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    /* --------------------------------------------------------
       SECTION HEADERS
    -------------------------------------------------------- */

    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 18px;
    }

    /* --------------------------------------------------------
       STATUS
    -------------------------------------------------------- */

    .status-pill {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 5px;
        margin-bottom: 5px;
    }

    .status-success {
        background: #dcfce7;
        color: #166534;
    }

    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }

    /* --------------------------------------------------------
       STREAMLIT COMPONENTS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.88);
        border: 1px solid #e2e8f0;
        padding: 14px 16px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(15,23,42,0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
    }

    .stButton > button {
        border-radius: 11px;
        font-weight: 700;
        border: 1px solid #cbd5e1;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(15,23,42,0.10);
    }

    button[kind="primary"] {
        border-radius: 12px !important;
        font-weight: 750 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        padding: 5px;
        background: rgba(226,232,240,0.65);
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 9px 15px;
        font-weight: 650;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 3px 12px rgba(15,23,42,0.08);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 15px;
    }

    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 100%
            );
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: #94a3b8;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.10);
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        padding: 20px;
        color: #64748b;
        font-size: 0.82rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "ai_report": "",
    "loaded_file_name": "",
    "analysis_status": False,
    "previous_file_name": "",
    "updated_file_name": "",
    "change_analysis": None,
    "ml_analysis": None,
    "ml_file_name": "",
    "ml_target": "",
    "ml_run_target": ""
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# GEMINI
# ============================================================

client = get_gemini_client()
api_key = os.getenv("GEMINI_API_KEY")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">AI • DATA • ML • INTELLIGENCE</div>
        <div class="hero-title">🤖 AI Dataset Intelligence</div>
        <div class="hero-subtitle">
            From raw data to actionable intelligence, machine learning,
            interactive exploration, and adaptive dataset reasoning.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📂 Dataset Workspace")

    uploaded_file = st.file_uploader(
        "Upload Current Dataset",
        type=["csv", "xlsx"],
        key="current_dataset"
    )

    st.caption(
        "Upload a CSV or Excel dataset for automated analysis."
    )

    st.divider()

    st.markdown("### 🔄 Mystery Mission")

    previous_file = st.file_uploader(
        "Upload Previous Dataset",
        type=["csv", "xlsx"],
        key="previous_dataset"
    )

    updated_file = st.file_uploader(
        "Upload Updated Dataset",
        type=["csv", "xlsx"],
        key="updated_dataset"
    )

    st.caption(
        "Compare two versions and detect structural, quality, "
        "and analytical changes."
    )

    st.divider()

    st.markdown("### 🧠 Intelligence Pipeline")

    st.markdown(
        """
        <div class="status-pill status-success">① Discovery</div>
        <div class="status-pill status-success">② Profiling</div>
        <div class="status-pill status-success">③ Quality</div>
        <div class="status-pill status-success">④ Visuals</div>
        <div class="status-pill status-success">⑤ AI</div>
        <div class="status-pill status-success">⑥ ML</div>
        <div class="status-pill status-success">⑦ Changes</div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if api_key:
        st.success("🟢 Gemini AI connected")
    else:
        st.warning("🟡 Gemini API not connected")

    if ML_ENGINE_AVAILABLE:
        st.success("🟢 ML engine available")
    else:
        st.warning("🟡 ML engine unavailable")


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-card-title">Welcome</div>
            <div class="info-card-value">
                Start with your dataset
            </div>
            <div class="info-card-description">
                Upload a CSV or Excel file from the sidebar to activate
                automated profiling, quality analysis, visualization,
                AI intelligence, and machine learning.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if previous_file is not None and updated_file is not None:
        st.success(
            "Previous and updated datasets detected. "
            "Upload a current dataset to open the full workspace."
        )

    st.stop()


# ============================================================
# LOAD CURRENT DATASET
# ============================================================

df, error = load_dataset(uploaded_file)

if error is not None:
    st.error(
        f"Unable to read the dataset: {error}"
    )
    st.stop()

if df is None or df.empty:
    st.error(
        "The uploaded dataset contains no usable records."
    )
    st.stop()


# ============================================================
# RESET STATE WHEN CURRENT DATASET CHANGES
# ============================================================

if st.session_state.loaded_file_name != uploaded_file.name:

    st.session_state.loaded_file_name = uploaded_file.name

    st.session_state.ai_report = ""
    st.session_state.analysis_status = False

    st.session_state.ml_analysis = None
    st.session_state.ml_file_name = ""
    st.session_state.ml_target = ""
    st.session_state.ml_run_target = ""


# ============================================================
# RESET CHANGE ANALYSIS WHEN FILES CHANGE
# ============================================================

current_previous_name = (
    previous_file.name
    if previous_file is not None
    else ""
)

current_updated_name = (
    updated_file.name
    if updated_file is not None
    else ""
)

if (
    st.session_state.previous_file_name != current_previous_name
    or
    st.session_state.updated_file_name != current_updated_name
):

    st.session_state.previous_file_name = current_previous_name
    st.session_state.updated_file_name = current_updated_name
    st.session_state.change_analysis = None


# ============================================================
# BASIC PROFILE
# ============================================================

profile = build_basic_profile(df)

rows = profile["rows"]
columns = profile["columns"]
missing_cells = profile["missing_cells"]
duplicate_rows = profile["duplicate_rows"]
numeric_df = profile["numeric_df"]
categorical_columns = profile["categorical_columns"]


# ============================================================
# COLUMN INTELLIGENCE
# ============================================================

(
    column_intelligence_df,
    identifier_columns,
    date_columns,
    target_candidates
) = build_column_intelligence(
    df,
    rows
)


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

numerical_summary = build_numerical_summary(
    numeric_df
)


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

categorical_summary_df = build_categorical_summary(
    df,
    categorical_columns,
    rows
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

(
    outlier_columns,
    total_outlier_values,
    anomaly_df
) = detect_outliers(
    df,
    numeric_df
)


# ============================================================
# QUALITY FINDINGS
# ============================================================

(
    quality_findings,
    empty_columns,
    constant_columns,
    high_cardinality_columns
) = build_quality_findings(
    df,
    rows,
    columns,
    missing_cells,
    duplicate_rows,
    outlier_columns
)

quality_df = pd.DataFrame(
    quality_findings
)


# ============================================================
# QUALITY SCORE
# ============================================================

quality_score, quality_status = calculate_quality_score(
    rows,
    columns,
    missing_cells,
    duplicate_rows,
    outlier_columns,
    constant_columns
)


# ============================================================
# ML READINESS
# ============================================================

ml_score, ml_status = calculate_ml_readiness(
    rows,
    missing_cells,
    duplicate_rows,
    outlier_columns,
    constant_columns,
    identifier_columns,
    columns
)


# ============================================================
# CORRELATION
# ============================================================

correlation_matrix = pd.DataFrame()
correlation_pairs = pd.DataFrame()

if len(numeric_df.columns) >= 2:

    correlation_matrix = (
        numeric_df
        .corr()
        .round(3)
    )

    pair_rows = []

    numeric_columns = list(
        correlation_matrix.columns
    )

    for i in range(len(numeric_columns)):

        for j in range(i + 1, len(numeric_columns)):

            value = correlation_matrix.iloc[i, j]

            if pd.notna(value):

                pair_rows.append(
                    {
                        "Attribute 1": numeric_columns[i],
                        "Attribute 2": numeric_columns[j],
                        "Correlation": value,
                        "Absolute Correlation": abs(value)
                    }
                )

    if pair_rows:

        correlation_pairs = pd.DataFrame(
            pair_rows
        )

        correlation_pairs = (
            correlation_pairs
            .sort_values(
                "Absolute Correlation",
                ascending=False
            )
        )


# ============================================================
# AGENT DECISIONS
# ============================================================

agent_decisions = build_agent_decisions(
    df,
    missing_cells,
    rows,
    outlier_columns,
    identifier_columns,
    date_columns,
    categorical_columns
)

agent_priority_df = pd.DataFrame(
    agent_decisions
)


# ============================================================
# VISUALIZATION PLAN
# ============================================================

visual_plan = build_visualization_plan(
    df,
    list(numeric_df.columns),
    list(categorical_columns),
    date_columns
)

visual_plan_df = pd.DataFrame(
    visual_plan
)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

executive_summary = build_executive_summary(
    quality_score,
    ml_score,
    quality_findings,
    outlier_columns,
    identifier_columns,
    target_candidates
)

risk_level = executive_summary["Risk Level"]

most_important_finding = (
    executive_summary["Most Important Finding"]
)

key_opportunity = (
    executive_summary["Key Opportunity"]
)

next_best_action = (
    executive_summary["Next Best Action"]
)


# ============================================================
# ANALYSIS STATUS
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Analysis Status</div>',
    unsafe_allow_html=True
)

status_cols = st.columns(7)

status_items = [
    ("Dataset", True),
    ("Profile", True),
    ("Quality", True),
    ("Visuals", True),
    ("AI", client is not None),
    ("ML", ML_ENGINE_AVAILABLE),
    (
        "Changes",
        previous_file is not None
        and updated_file is not None
    )
]

for container, (label, completed) in zip(
    status_cols,
    status_items
):

    with container:

        if completed:
            st.success(
                f"✓ {label}"
            )
        else:
            st.warning(
                f"○ {label}"
            )


# ============================================================
# TOP METRICS
# ============================================================

st.divider()

show_metric_cards(
    rows,
    columns,
    missing_cells,
    duplicate_rows
)


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Executive Dataset Summary</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'A high-level view of dataset health, analytical risk, and opportunity.'
    '</div>',
    unsafe_allow_html=True
)

summary_cols = st.columns(3)

with summary_cols[0]:

    st.metric(
        "Data Quality",
        f"{quality_score}/100",
        quality_status
    )

with summary_cols[1]:

    st.metric(
        "ML Readiness",
        f"{ml_score}/100",
        ml_status
    )

with summary_cols[2]:

    st.metric(
        "Risk Level",
        risk_level
    )


finding_col, opportunity_col = st.columns(2)

with finding_col:

    st.markdown(
        "### 🔎 Most Important Finding"
    )

    st.info(
        most_important_finding
    )

with opportunity_col:

    st.markdown(
        "### 💡 Key Opportunity"
    )

    st.success(
        key_opportunity
    )


st.markdown(
    "### 🎯 Next Best Action"
)

st.warning(
    next_best_action
)


# ============================================================
# WORKSPACES
# ============================================================

(
    overview_tab,
    quality_tab,
    visual_tab,
    intelligence_tab,
    ml_tab,
    ai_tab,
    change_tab
) = st.tabs(
    [
        "🏠 Overview",
        "🛡️ Quality",
        "📊 Visuals",
        "🧠 Intelligence",
        "🤖 Machine Learning",
        "💼 Executive AI",
        "🔄 Change Intelligence"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    st.markdown(
        '<div class="section-title">📌 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(
        [1.15, 1]
    )

    with c1:

        overview_df = pd.DataFrame(
            {
                "Metric": [
                    "Dataset",
                    "Records",
                    "Attributes",
                    "Numerical Attributes",
                    "Categorical Attributes",
                    "Identifier Candidates",
                    "Date/Time Fields",
                    "Target Candidates"
                ],

                "Value": [
                    uploaded_file.name,
                    f"{rows:,}",
                    f"{columns:,}",
                    f"{len(numeric_df.columns):,}",
                    f"{len(categorical_columns):,}",
                    f"{len(identifier_columns):,}",
                    f"{len(date_columns):,}",
                    f"{len(target_candidates):,}"
                ]
            }
        )

        st.dataframe(
            overview_df,
            use_container_width=True,
            hide_index=True
        )

    with c2:

        st.markdown(
            "### 🎯 Dataset Readiness"
        )

        show_ml_readiness(
            ml_score,
            ml_status
        )

        st.progress(
            ml_score / 100
        )

        show_quality_score(
            quality_score,
            quality_status
        )

        st.progress(
            quality_score / 100
        )

    st.markdown(
        "### 👀 Dataset Preview"
    )

    show_data_preview(
        df,
        rows=20
    )


# ============================================================
# QUALITY
# ============================================================

with quality_tab:

    st.markdown(
        '<div class="section-title">'
        '🛡️ Quality & Anomaly Command Center'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Detect missing data, duplicates, anomalies, and structural quality risks.'
        '</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.metric(
            "Quality Score",
            f"{quality_score}/100"
        )

    with q2:

        missing_ratio = (
            missing_cells /
            (rows * columns) *
            100
            if rows * columns > 0
            else 0
        )

        st.metric(
            "Missing Data",
            f"{missing_ratio:.2f}%"
        )

    with q3:

        duplicate_ratio = (
            duplicate_rows /
            rows *
            100
            if rows > 0
            else 0
        )

        st.metric(
            "Duplicate Data",
            f"{duplicate_ratio:.2f}%"
        )

    with q4:

        st.metric(
            "Potential Anomalies",
            f"{total_outlier_values:,}"
        )

    st.divider()

    if not quality_df.empty:

        st.markdown(
            "### 📋 Quality Findings"
        )

        st.dataframe(
            quality_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "✅ No major quality issues detected."
        )

    st.markdown(
        "### 🚨 Numerical Anomalies"
    )

    if not anomaly_df.empty:

        st.dataframe(
            anomaly_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "✅ No IQR-based numerical anomalies detected."
        )


# ============================================================
# VISUAL EXPLORATION
# ============================================================

with visual_tab:

    st.markdown(
        '<div class="section-title">📊 Deep Visual Exploration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore distributions, relationships, categories, time trends, '
        'missingness, and multidimensional patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # NUMERICAL
    # --------------------------------------------------------

    if not numeric_df.empty:

        st.markdown(
            "### 📈 Numerical Distribution"
        )

        selected_numeric = st.selectbox(
            "Select numerical attribute",
            list(numeric_df.columns),
            key="visual_numeric"
        )

        series = (
            numeric_df[
                selected_numeric
            ]
            .dropna()
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            histogram = create_numerical_histogram(
                df,
                selected_numeric
            )

            histogram.update_layout(
                height=430,
                template="plotly_white"
            )

            st.plotly_chart(
                histogram,
                use_container_width=True
            )

        with chart_col2:

            box = create_numerical_boxplot(
                df,
                selected_numeric
            )

            box.update_layout(
                height=430,
                template="plotly_white"
            )

            st.plotly_chart(
                box,
                use_container_width=True
            )

        if not series.empty:

            stat_cols = st.columns(5)

            stats = [
                ("Mean", series.mean()),
                ("Median", series.median()),
                ("Minimum", series.min()),
                ("Maximum", series.max()),
                ("Std Dev", series.std())
            ]

            for container, (label, value) in zip(
                stat_cols,
                stats
            ):

                with container:

                    if pd.notna(value):

                        st.metric(
                            label,
                            f"{value:.3f}"
                        )


    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    if len(numeric_df.columns) >= 2:

        st.divider()

        st.markdown(
            "### 🔥 Correlation Explorer"
        )

        threshold = st.slider(
            "Minimum absolute correlation",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            key="correlation_threshold"
        )

        filtered_matrix = (
            correlation_matrix
            .copy()
        )

        filtered_matrix = (
            filtered_matrix.mask(
                filtered_matrix.abs() < threshold
            )
        )

        correlation_chart = px.imshow(
            filtered_matrix,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            title=(
                f"Relationships with "
                f"|correlation| ≥ {threshold:.2f}"
            )
        )

        correlation_chart.update_layout(
            height=600,
            template="plotly_white"
        )

        st.plotly_chart(
            correlation_chart,
            use_container_width=True
        )

        if not correlation_pairs.empty:

            st.markdown(
                "### Strongest Numerical Relationships"
            )

            st.dataframe(
                correlation_pairs[
                    [
                        "Attribute 1",
                        "Attribute 2",
                        "Correlation"
                    ]
                ].head(10),
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # MULTIDIMENSIONAL
        # ----------------------------------------------------

        st.markdown(
            "### 🔬 Multidimensional Explorer"
        )

        default_columns = list(
            numeric_df.columns
        )[
            :min(
                4,
                len(numeric_df.columns)
            )
        ]

        matrix_columns = st.multiselect(
            "Select 2–5 numerical attributes",
            list(numeric_df.columns),
            default=default_columns,
            max_selections=5,
            key="matrix_columns"
        )

        if len(matrix_columns) >= 2:

            matrix_df = (
                df[matrix_columns]
                .dropna()
            )

            if len(matrix_df) > 1000:

                matrix_df = (
                    matrix_df
                    .sample(
                        1000,
                        random_state=42
                    )
                )

            matrix_chart = create_scatter_matrix(
                matrix_df,
                matrix_columns
            )

            matrix_chart.update_layout(
                height=720,
                template="plotly_white"
            )

            st.plotly_chart(
                matrix_chart,
                use_container_width=True
            )


        # ----------------------------------------------------
        # 3D VISUALIZATION
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "### 🌐 3D Data Space"
        )

        st.caption(
            "Interactive 3D visualization of three numerical dimensions."
        )

        if len(numeric_df.columns) >= 3:

            three_d_columns = st.multiselect(
                "Choose 3 numerical attributes",
                list(numeric_df.columns),
                default=list(
                    numeric_df.columns
                )[:3],
                max_selections=3,
                key="three_d_columns"
            )

            if len(three_d_columns) == 3:

                plot_3d_df = (
                    df[three_d_columns]
                    .dropna()
                    .copy()
                )

                if len(plot_3d_df) > 2000:

                    plot_3d_df = (
                        plot_3d_df
                        .sample(
                            2000,
                            random_state=42
                        )
                    )

                x_col = three_d_columns[0]
                y_col = three_d_columns[1]
                z_col = three_d_columns[2]

                fig_3d = px.scatter_3d(
                    plot_3d_df,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    title="Interactive 3D Numerical Data Space",
                    opacity=0.75
                )

                fig_3d.update_layout(
                    height=650,
                    template="plotly_white",
                    margin=dict(
                        l=0,
                        r=0,
                        t=60,
                        b=0
                    )
                )

                st.plotly_chart(
                    fig_3d,
                    use_container_width=True
                )

                st.info(
                    "💡 Drag the chart to rotate the dataset in 3D. "
                    "Use the mouse wheel to zoom."
                )

        else:

            st.info(
                "At least three numerical attributes are required "
                "for the 3D explorer."
            )


    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    if len(categorical_columns) > 0:

        st.divider()

        st.markdown(
            "### 📊 Categorical Exploration"
        )

        selected_category = st.selectbox(
            "Select categorical attribute",
            list(categorical_columns),
            key="visual_category"
        )

        category_chart = (
            create_categorical_frequency(
                df,
                selected_category
            )
        )

        category_chart.update_layout(
            height=450,
            template="plotly_white"
        )

        st.plotly_chart(
            category_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # DATE / TIME
    # --------------------------------------------------------

    if date_columns:

        st.divider()

        st.markdown(
            "### 📅 Temporal Exploration"
        )

        selected_date = st.selectbox(
            "Select date/time attribute",
            date_columns,
            key="visual_date"
        )

        date_series = pd.to_datetime(
            df[selected_date],
            errors="coerce"
        )

        temporal_df = pd.DataFrame(
            {
                selected_date:
                    date_series
            }
        ).dropna()

        if not temporal_df.empty:

            temporal_df["Period"] = (
                temporal_df[selected_date]
                .dt.to_period("M")
                .astype(str)
            )

            trend_df = (
                temporal_df
                .groupby("Period")
                .size()
                .reset_index(
                    name="Records"
                )
            )

            if len(trend_df) > 1:

                trend_chart = px.line(
                    trend_df,
                    x="Period",
                    y="Records",
                    markers=True,
                    title=(
                        f"Record Trend — "
                        f"{selected_date}"
                    )
                )

                trend_chart.update_layout(
                    height=430,
                    template="plotly_white"
                )

                st.plotly_chart(
                    trend_chart,
                    use_container_width=True
                )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    if missing_cells > 0:

        st.divider()

        st.markdown(
            "### 🕳️ Missing Data Pattern"
        )

        missing_chart = (
            create_missing_value_chart(df)
        )

        if missing_chart is not None:

            missing_chart.update_layout(
                height=430,
                template="plotly_white"
            )

            st.plotly_chart(
                missing_chart,
                use_container_width=True
            )


# ============================================================
# INTELLIGENCE
# ============================================================

with intelligence_tab:

    st.markdown(
        '<div class="section-title">'
        '🧠 Explainable Dataset Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Understand what the agent detected and why it recommends each action.'
        '</div>',
        unsafe_allow_html=True
    )

    show_column_table(
        column_intelligence_df
    )

    st.markdown(
        "### 🤖 Agent Decisions"
    )

    show_agent_decisions(
        agent_decisions
    )

    st.markdown(
        "### 📊 Smart Visualization Plan"
    )

    st.dataframe(
        visual_plan_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### 🎯 Potential Targets"
    )

    if target_candidates:

        st.success(
            ", ".join(
                target_candidates
            )
        )

    else:

        st.info(
            "No obvious target field was detected."
        )

    st.markdown(
        "### ⚠️ Important Findings"
    )

    if not agent_priority_df.empty:

        important_rows = (
            agent_priority_df[
                agent_priority_df["Priority"].isin(
                    ["Critical", "High"]
                )
            ]
        )

    else:

        important_rows = pd.DataFrame()

    if not important_rows.empty:

        for _, row in (
            important_rows
            .head(8)
            .iterrows()
        ):

            priority = str(
                row.get(
                    "Priority",
                    "Medium"
                )
            )

            finding = str(
                row.get(
                    "Finding",
                    "No finding available"
                )
            )

            action = str(
                row.get(
                    "Action",
                    "No recommended action available"
                )
            )

            st.warning(
                f"**{priority} Priority**  \n"
                f"🔎 **Finding:** {finding}  \n"
                f"🎯 **Recommended Action:** {action}"
            )

    else:

        st.success(
            "✅ No critical or high-priority findings detected."
        )


# ============================================================
# MACHINE LEARNING HELPERS
# ============================================================

def _find_target_candidates(result):

    possible = []

    if not isinstance(result, dict):
        return possible

    keys = [
        "target_candidates",
        "target_candidate",
        "candidates",
        "targets"
    ]

    for key in keys:

        value = result.get(key)

        if isinstance(value, list):

            for item in value:

                if isinstance(item, dict):

                    name = (
                        item.get("column")
                        or item.get("target")
                        or item.get("name")
                    )

                    if name:
                        possible.append(str(name))

                elif item is not None:

                    possible.append(str(item))

        elif isinstance(value, str):

            possible.append(value)

    return list(
        dict.fromkeys(possible)
    )


def _extract_result_value(result, keys, default=None):

    if not isinstance(result, dict):
        return default

    for key in keys:

        if key in result:
            return result[key]

    return default


def _call_ml_engine(df, target):

    """
    Calls the existing ml_engine.py while being tolerant of
    small differences in parameter names.
    """

    if not ML_ENGINE_AVAILABLE:
        return None

    function = run_ml_analysis

    try:

        signature = inspect.signature(function)

        parameters = signature.parameters

        kwargs = {}

        for name in parameters:

            lowered = name.lower()

            if lowered in [
                "df",
                "data",
                "dataset",
                "dataframe"
            ]:

                kwargs[name] = df

            elif lowered in [
                "target",
                "target_column",
                "target_col",
                "target_name",
                "y_column",
                "label"
            ]:

                kwargs[name] = target

        if kwargs:

            return function(**kwargs)

        return function(df, target)

    except TypeError:

        try:
            return function(df, target)
        except Exception as e:
            raise e

    except Exception as e:

        raise e


def _display_ml_result(result):

    if result is None:
        return

    if not isinstance(result, dict):

        st.write(result)
        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = None

    try:

        if "build_ml_summary" in globals():

            summary = build_ml_summary(result)

    except Exception:
        summary = None

    if summary:

        st.markdown(
            "### 🧾 ML Summary"
        )

        if isinstance(summary, str):

            st.info(summary)

        elif isinstance(summary, dict):

            summary_rows = []

            for key, value in summary.items():

                summary_rows.append(
                    {
                        "Metric": str(key),
                        "Value": str(value)
                    }
                )

            if summary_rows:

                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True
                )

    # --------------------------------------------------------
    # READINESS
    # --------------------------------------------------------

    readiness = _extract_result_value(
        result,
        [
            "ml_readiness",
            "readiness",
            "readiness_score",
            "score"
        ]
    )

    if readiness is not None:

        try:

            readiness_number = float(
                readiness
            )

            readiness_number = max(
                0,
                min(
                    100,
                    readiness_number
                )
            )

            st.markdown(
                "### 🎯 ML Readiness"
            )

            st.progress(
                readiness_number / 100
            )

            st.metric(
                "ML Readiness Score",
                f"{readiness_number:.0f}/100"
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # PROBLEM TYPE
    # --------------------------------------------------------

    problem_type = _extract_result_value(
        result,
        [
            "problem_type",
            "task_type",
            "task",
            "problem"
        ]
    )

    if problem_type:

        st.markdown(
            "### 🧩 Problem Type"
        )

        st.success(
            str(problem_type)
        )

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warnings = _extract_result_value(
        result,
        [
            "warnings",
            "limitations",
            "issues"
        ],
        []
    )

    if warnings:

        st.markdown(
            "### ⚠️ ML Warnings"
        )

        if isinstance(warnings, list):

            for warning in warnings:

                st.warning(
                    str(warning)
                )

        else:

            st.warning(
                str(warnings)
            )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    model_comparison = _extract_result_value(
        result,
        [
            "model_comparison",
            "model_results",
            "models",
            "comparison"
        ]
    )

    if model_comparison is not None:

        st.markdown(
            "### 🏆 Model Comparison"
        )

        if isinstance(
            model_comparison,
            pd.DataFrame
        ):

            comparison_df = model_comparison.copy()

        elif isinstance(
            model_comparison,
            list
        ):

            comparison_df = pd.DataFrame(
                model_comparison
            )

        elif isinstance(
            model_comparison,
            dict
        ):

            try:

                comparison_df = pd.DataFrame(
                    model_comparison
                )

            except Exception:

                comparison_df = pd.DataFrame(
                    [
                        {
                            "Model": key,
                            "Result": value
                        }
                        for key, value
                        in model_comparison.items()
                    ]
                )

        else:

            comparison_df = pd.DataFrame()

        if not comparison_df.empty:

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = _extract_result_value(
        result,
        [
            "best_model",
            "selected_model",
            "best_model_name"
        ]
    )

    if best_model:

        st.markdown(
            "### 🥇 Selected Best Model"
        )

        st.success(
            str(best_model)
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = _extract_result_value(
        result,
        [
            "metrics",
            "best_metrics",
            "evaluation"
        ]
    )

    if metrics is not None:

        st.markdown(
            "### 📏 Evaluation Metrics"
        )

        if isinstance(metrics, dict):

            metric_cols = st.columns(
                min(
                    4,
                    max(
                        1,
                        len(metrics)
                    )
                )
            )

            for index, (
                key,
                value
            ) in enumerate(
                metrics.items()
            ):

                with metric_cols[
                    index % len(metric_cols)
                ]:

                    if isinstance(
                        value,
                        float
                    ):

                        st.metric(
                            str(key),
                            f"{value:.4f}"
                        )

                    else:

                        st.metric(
                            str(key),
                            str(value)
                        )

        elif isinstance(
            metrics,
            pd.DataFrame
        ):

            st.dataframe(
                metrics,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.write(metrics)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance = _extract_result_value(
        result,
        [
            "feature_importance",
            "feature_importances",
            "importance"
        ]
    )

    if feature_importance is not None:

        st.markdown(
            "### 🔍 Feature Importance"
        )

        if isinstance(
            feature_importance,
            pd.DataFrame
        ):

            importance_df = (
                feature_importance
                .copy()
            )

        elif isinstance(
            feature_importance,
            list
        ):

            importance_df = pd.DataFrame(
                feature_importance
            )

        elif isinstance(
            feature_importance,
            dict
        ):

            importance_df = pd.DataFrame(
                [
                    {
                        "Feature": key,
                        "Importance": value
                    }
                    for key, value
                    in feature_importance.items()
                ]
            )

        else:

            importance_df = pd.DataFrame()

        if not importance_df.empty:

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

            numeric_importance_columns = (
                importance_df
                .select_dtypes(
                    include="number"
                )
                .columns
                .tolist()
            )

            if numeric_importance_columns:

                value_column = (
                    numeric_importance_columns[-1]
                )

                label_column = None

                for candidate in [
                    "Feature",
                    "feature",
                    "Attribute",
                    "attribute",
                    "feature_name"
                ]:

                    if candidate in importance_df.columns:

                        label_column = candidate
                        break

                if label_column is not None:

                    chart_df = (
                        importance_df[
                            [
                                label_column,
                                value_column
                            ]
                        ]
                        .dropna()
                        .sort_values(
                            value_column,
                            ascending=True
                        )
                        .tail(15)
                    )

                    if not chart_df.empty:

                        importance_chart = px.bar(
                            chart_df,
                            x=value_column,
                            y=label_column,
                            orientation="h",
                            title="Top Feature Importance"
                        )

                        importance_chart.update_layout(
                            height=500,
                            template="plotly_white"
                        )

                        st.plotly_chart(
                            importance_chart,
                            use_container_width=True
                        )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predictions = _extract_result_value(
        result,
        [
            "predictions",
            "prediction_samples",
            "sample_predictions"
        ]
    )

    if predictions is not None:

        st.markdown(
            "### 🔮 Prediction Samples"
        )

        if isinstance(
            predictions,
            pd.DataFrame
        ):

            prediction_df = predictions

        elif isinstance(
            predictions,
            list
        ):

            prediction_df = pd.DataFrame(
                predictions
            )

        else:

            prediction_df = pd.DataFrame()

        if not prediction_df.empty:

            st.dataframe(
                prediction_df.head(20),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------------

    reliability = _extract_result_value(
        result,
        [
            "reliability",
            "ml_reliability",
            "reliability_assessment"
        ]
    )

    if reliability:

        st.markdown(
            "### 🧪 Model Reliability"
        )

        if isinstance(
            reliability,
            dict
        ):

            reliability_rows = []

            for key, value in reliability.items():

                reliability_rows.append(
                    {
                        "Assessment": str(key),
                        "Result": str(value)
                    }
                )

            st.dataframe(
                pd.DataFrame(
                    reliability_rows
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                str(reliability)
            )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    recommendation = _extract_result_value(
        result,
        [
            "final_recommendation",
            "recommendation",
            "ml_recommendation"
        ]
    )

    if recommendation:

        st.markdown(
            "### 🎯 ML Recommendation"
        )

        recommendation_text = str(
            recommendation
        )

        if (
            "NOT SUITABLE" in
            recommendation_text.upper()
        ):

            st.error(
                f"🔴 {recommendation_text}"
            )

        elif (
            "NEEDS ATTENTION" in
            recommendation_text.upper()
        ):

            st.warning(
                f"🟡 {recommendation_text}"
            )

        else:

            st.success(
                f"🟢 {recommendation_text}"
            )


# ============================================================
# MACHINE LEARNING WORKSPACE
# ============================================================

with ml_tab:

    st.markdown(
        '<div class="section-title">'
        '🤖 Machine Learning Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'The agent first evaluates ML readiness, identifies a target, '
        'selects an appropriate learning task, trains candidate models, '
        'compares them, and explains the result.'
        '</div>',
        unsafe_allow_html=True
    )

    if not ML_ENGINE_AVAILABLE:

        st.error(
            "The ML engine could not be loaded."
        )

        st.code(
            ML_ENGINE_ERROR,
            language="text"
        )

        st.info(
            "Make sure ml_engine.py is present in the same project folder."
        )

    else:

        # ----------------------------------------------------
        # ML OVERVIEW CARDS
        # ----------------------------------------------------

        ml_top_cols = st.columns(4)

        with ml_top_cols[0]:

            st.metric(
                "ML Readiness",
                f"{ml_score}/100"
            )

        with ml_top_cols[1]:

            st.metric(
                "Rows",
                f"{rows:,}"
            )

        with ml_top_cols[2]:

            st.metric(
                "Features",
                f"{max(0, columns - 1):,}"
            )

        with ml_top_cols[3]:

            st.metric(
                "Target Candidates",
                f"{len(target_candidates):,}"
            )

        st.divider()

        # ----------------------------------------------------
        # TARGET SELECTION
        # ----------------------------------------------------

        st.markdown(
            "### 🎯 Target Selection"
        )

        if target_candidates:

            st.info(
                "The dataset intelligence engine detected possible "
                "target variables. The first candidate is selected by default."
            )

            target_options = [
                "Automatic target selection"
            ] + list(
                dict.fromkeys(
                    target_candidates
                )
            )

        else:

            st.warning(
                "No obvious target was detected automatically. "
                "Select a target column manually."
            )

            target_options = [
                "Manual target selection"
            ]

        target_mode = st.radio(
            "Target strategy",
            [
                "Automatic",
                "Manual"
            ],
            horizontal=True,
            key="ml_target_mode"
        )

        if target_mode == "Automatic":

            if target_candidates:

                selected_target = target_candidates[0]

            else:

                selected_target = None

                st.warning(
                    "Automatic mode cannot continue because no target "
                    "candidate was detected."
                )

        else:

            selected_target = st.selectbox(
                "Choose target column",
                list(df.columns),
                key="manual_ml_target"
            )

        if selected_target:

            st.markdown(
                f"""
                <div class="ml-card">
                    <div class="ml-card-title">
                        🎯 Selected Target
                    </div>
                    <div class="ml-card-text">
                        {selected_target}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # ML GATE
        # ----------------------------------------------------

        readiness_allowed = (
            ml_score >= 40
            and
            rows >= 10
            and
            selected_target is not None
        )

        if ml_score < 40:

            st.warning(
                "⚠️ The dataset has low ML readiness. "
                "The agent can still attempt analysis, but results "
                "may be unreliable."
            )

        if rows < 10:

            st.warning(
                "⚠️ The dataset contains very few rows. "
                "Model evaluation may be unstable."
            )

        # ----------------------------------------------------
        # RUN ML
        # ----------------------------------------------------

        run_ml = st.button(
            "🚀 Run Intelligent ML Analysis",
            type="primary",
            disabled=selected_target is None,
            key="run_ml_button"
        )

        if run_ml and selected_target:

            with st.spinner(
                "🤖 ML agent is preparing data, selecting models, "
                "training candidates, and evaluating performance..."
            ):

                try:

                    result = _call_ml_engine(
                        df,
                        selected_target
                    )

                    st.session_state.ml_analysis = result
                    st.session_state.ml_file_name = (
                        uploaded_file.name
                    )
                    st.session_state.ml_target = (
                        selected_target
                    )
                    st.session_state.ml_run_target = (
                        selected_target
                    )

                    st.success(
                        "✅ Machine learning analysis completed."
                    )

                except Exception as e:

                    st.session_state.ml_analysis = None

                    st.error(
                        "Machine learning analysis failed."
                    )

                    st.exception(e)

        # ----------------------------------------------------
        # DISPLAY ML RESULT
        # ----------------------------------------------------

        if (
            st.session_state.ml_analysis is not None
            and
            st.session_state.ml_file_name
            == uploaded_file.name
        ):

            st.divider()

            st.markdown(
                "## 🧠 ML Agent Results"
            )

            _display_ml_result(
                st.session_state.ml_analysis
            )

            st.divider()

            st.markdown(
                "### 🔐 Original Dataset Protection"
            )

            st.info(
                "The ML workflow analyzes the dataset in memory. "
                "The uploaded original dataset is never overwritten."
            )


# ============================================================
# EXECUTIVE AI
# ============================================================

with ai_tab:

    st.markdown(
        '<div class="section-title">'
        '💼 Executive AI Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Gemini interprets the analytical evidence and produces a '
        'professional executive-level intelligence briefing.'
        '</div>',
        unsafe_allow_html=True
    )

    if client is None:

        st.warning(
            "AI analysis is unavailable because the "
            "Gemini API key was not detected."
        )

        st.info(
            "The complete statistical, quality, intelligence, "
            "visualization, ML, and change analysis remains available."
        )

    else:

        generate_ai = st.button(
            "✨ Generate Executive Intelligence",
            type="primary",
            key="generate_ai_report"
        )

        if generate_ai:

            with st.spinner(
                "🤖 AI Agent is investigating your dataset..."
            ):

                profile_data = {
                    "dataset": uploaded_file.name,
                    "rows": rows,
                    "columns": columns,
                    "numerical_columns":
                        list(numeric_df.columns),
                    "categorical_columns":
                        list(categorical_columns),
                    "identifier_columns":
                        identifier_columns,
                    "date_columns":
                        date_columns,
                    "target_candidates":
                        target_candidates
                }

                numerical_evidence = None

                if not numerical_summary.empty:

                    numerical_evidence = (
                        numerical_summary
                        .reset_index()
                        .rename(
                            columns={
                                "index": "Attribute"
                            }
                        )
                        .head(30)
                    )

                categorical_evidence = None

                if not categorical_summary_df.empty:

                    categorical_evidence = (
                        categorical_summary_df
                        .head(30)
                    )

                correlation_evidence = None

                if not correlation_pairs.empty:

                    correlation_evidence = (
                        correlation_pairs[
                            [
                                "Attribute 1",
                                "Attribute 2",
                                "Correlation"
                            ]
                        ]
                        .head(15)
                    )

                anomaly_evidence = None

                if not anomaly_df.empty:

                    anomaly_evidence = (
                        anomaly_df
                        .head(30)
                    )

                column_evidence = (
                    column_intelligence_df
                    .head(50)
                )

                missing_evidence = (
                    df.isna()
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                missing_evidence = (
                    missing_evidence[
                        missing_evidence > 0
                    ]
                    .head(30)
                    .to_dict()
                )

                report = generate_ai_report(

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

                    missing_data=missing_evidence
                )

                if report:

                    st.session_state.ai_report = report
                    st.session_state.analysis_status = True

                    st.success(
                        "✅ Executive intelligence generated successfully."
                    )

                else:

                    st.warning(
                        "The AI service returned no report."
                    )

    if st.session_state.ai_report:

        st.divider()

        show_ai_report(
            st.session_state.ai_report
        )

        st.download_button(
            "⬇️ Download Executive Report",
            data=st.session_state.ai_report,
            file_name="AI_Dataset_Executive_Report.txt",
            mime="text/plain",
            key="download_ai_report"
        )


# ============================================================
# CHANGE INTELLIGENCE
# ============================================================

with change_tab:

    st.markdown(
        '<div class="section-title">'
        '🔄 Dataset Change Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Detect structural changes, quality changes, and whether previous '
        'findings should still be trusted.'
        '</div>',
        unsafe_allow_html=True
    )

    if previous_file is None or updated_file is None:

        st.info(
            "Upload both a Previous Dataset and an Updated Dataset "
            "from the sidebar to activate the adaptive intelligence workflow."
        )

        st.markdown(
            """
            ### 🔬 Adaptive Agent Workflow

            **DETECT → ANALYZE → COMPARE → RECONSIDER → RE-PLAN → REPORT**

            The agent evaluates:

            - Added columns
            - Removed columns
            - Possible renamed columns
            - Data-type conflicts
            - Missing-value changes
            - Duplicate changes
            - Impact on previous findings
            - Final analytical recommendation
            """
        )

    else:

        old_df, old_error = load_dataset(
            previous_file
        )

        new_df, new_error = load_dataset(
            updated_file
        )

        if old_error is not None:

            st.error(
                f"Unable to read previous dataset: {old_error}"
            )

        elif new_error is not None:

            st.error(
                f"Unable to read updated dataset: {new_error}"
            )

        elif (
            old_df is None
            or new_df is None
            or old_df.empty
            or new_df.empty
        ):

            st.error(
                "One of the datasets contains no usable records."
            )

        else:

            st.success(
                "✅ Previous and updated datasets loaded successfully."
            )

            comparison_cols = st.columns(2)

            with comparison_cols[0]:

                st.markdown(
                    "### 📄 Previous Dataset"
                )

                st.write(
                    f"**File:** {previous_file.name}"
                )

                st.metric(
                    "Rows",
                    f"{len(old_df):,}"
                )

                st.metric(
                    "Columns",
                    f"{len(old_df.columns):,}"
                )

            with comparison_cols[1]:

                st.markdown(
                    "### 📄 Updated Dataset"
                )

                st.write(
                    f"**File:** {updated_file.name}"
                )

                st.metric(
                    "Rows",
                    f"{len(new_df):,}"
                )

                st.metric(
                    "Columns",
                    f"{len(new_df.columns):,}"
                )

            st.divider()

            st.markdown(
                "### 🧠 Previous Finding"
            )

            previous_finding = st.text_area(
                "Enter the important finding from the previous analysis",
                value=most_important_finding,
                height=100,
                key="previous_finding"
            )

            run_changes = st.button(
                "🚀 Run Dataset Intelligence",
                type="primary",
                key="run_change_analysis"
            )

            if run_changes:

                with st.spinner(
                    "🧠 Agent is comparing datasets and reconsidering previous findings..."
                ):

                    try:

                        result = run_intelligence_analysis(
                            old_df,
                            new_df,
                            previous_finding
                        )

                        st.session_state.change_analysis = result

                    except Exception as e:

                        st.session_state.change_analysis = None

                        st.error(
                            f"Change intelligence failed: "
                            f"{type(e).__name__}: {str(e)}"
                        )

            if st.session_state.change_analysis is not None:

                result = (
                    st.session_state.change_analysis
                )

                st.divider()

                st.markdown(
                    "## 🤖 Intelligence Workflow"
                )

                workflow_cols = st.columns(6)

                workflow_steps = [
                    "🔍 Detect",
                    "📊 Analyze",
                    "⚖️ Compare",
                    "🧠 Reconsider",
                    "🔄 Re-plan",
                    "📋 Report"
                ]

                for container, step in zip(
                    workflow_cols,
                    workflow_steps
                ):

                    with container:

                        st.success(
                            step
                        )

                # ------------------------------------------------
                # STRUCTURAL CHANGES
                # ------------------------------------------------

                changes = result.get(
                    "changes",
                    {}
                )

                st.divider()

                st.markdown(
                    "## 🔍 Detected Changes"
                )

                added_columns = changes.get(
                    "added_columns",
                    []
                )

                removed_columns = changes.get(
                    "removed_columns",
                    []
                )

                type_changes = changes.get(
                    "type_changes",
                    []
                )

                possible_renames = changes.get(
                    "possible_renames",
                    []
                )

                change_metrics = st.columns(4)

                with change_metrics[0]:

                    st.metric(
                        "Added Columns",
                        len(added_columns)
                    )

                with change_metrics[1]:

                    st.metric(
                        "Removed Columns",
                        len(removed_columns)
                    )

                with change_metrics[2]:

                    st.metric(
                        "Type Changes",
                        len(type_changes)
                    )

                with change_metrics[3]:

                    st.metric(
                        "Possible Renames",
                        len(possible_renames)
                    )

                if added_columns:

                    st.markdown(
                        "### ➕ Added Columns"
                    )

                    for column in added_columns:

                        st.success(
                            f"Added: **{column}**"
                        )

                if removed_columns:

                    st.markdown(
                        "### ➖ Removed Columns"
                    )

                    for column in removed_columns:

                        st.error(
                            f"Removed: **{column}**"
                        )

                if type_changes:

                    st.markdown(
                        "### 🔄 Data-Type Changes"
                    )

                    st.dataframe(
                        pd.DataFrame(
                            type_changes
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                if possible_renames:

                    st.markdown(
                        "### ✏️ Possible Renamed Columns"
                    )

                    rename_df = pd.DataFrame(
                        possible_renames
                    )

                    st.dataframe(
                        rename_df,
                        use_container_width=True,
                        hide_index=True
                    )

                if (
                    not added_columns
                    and
                    not removed_columns
                    and
                    not type_changes
                    and
                    not possible_renames
                ):

                    st.success(
                        "✅ No structural changes detected."
                    )

                # ------------------------------------------------
                # QUALITY COMPARISON
                # ------------------------------------------------

                quality = result.get(
                    "quality",
                    {}
                )

                st.divider()

                st.markdown(
                    "## 🛡️ Quality Comparison"
                )

                old_duplicates = quality.get(
                    "old_duplicates",
                    0
                )

                new_duplicates = quality.get(
                    "new_duplicates",
                    0
                )

                duplicate_change = quality.get(
                    "duplicate_change",
                    0
                )

                quality_cols = st.columns(3)

                with quality_cols[0]:

                    st.metric(
                        "Previous Duplicates",
                        old_duplicates
                    )

                with quality_cols[1]:

                    st.metric(
                        "Updated Duplicates",
                        new_duplicates
                    )

                with quality_cols[2]:

                    st.metric(
                        "Duplicate Change",
                        duplicate_change
                    )

                missing_changes = quality.get(
                    "missing_changes",
                    []
                )

                if missing_changes:

                    st.markdown(
                        "### 🕳️ Missing-Value Changes"
                    )

                    missing_df = pd.DataFrame(
                        missing_changes
                    )

                    st.dataframe(
                        missing_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.success(
                        "✅ No missing-value rate changes detected."
                    )

                # ------------------------------------------------
                # IMPACT ASSESSMENT
                # ------------------------------------------------

                impact = result.get(
                    "impact",
                    []
                )

                st.divider()

                st.markdown(
                    "## ⚠️ Impact Assessment"
                )

                if impact:

                    for item in impact:

                        area = item.get(
                            "area",
                            "UNKNOWN"
                        )

                        severity = item.get(
                            "severity",
                            "MEDIUM"
                        )

                        message = item.get(
                            "message",
                            ""
                        )

                        if severity == "HIGH":

                            st.error(
                                f"🔴 **{area} — {severity}**  \n"
                                f"{message}"
                            )

                        elif severity == "MEDIUM":

                            st.warning(
                                f"🟡 **{area} — {severity}**  \n"
                                f"{message}"
                            )

                        else:

                            st.info(
                                f"🔵 **{area} — {severity}**  \n"
                                f"{message}"
                            )

                else:

                    st.success(
                        "✅ No significant impact detected."
                    )

                # ------------------------------------------------
                # RECONSIDERATION
                # ------------------------------------------------

                reconsideration = result.get(
                    "reconsideration",
                    {}
                )

                st.divider()

                st.markdown(
                    "## 🧠 Reconsider Previous Finding"
                )

                reconsideration_status = (
                    reconsideration.get(
                        "status",
                        "UNKNOWN"
                    )
                )

                warnings = reconsideration.get(
                    "warnings",
                    []
                )

                if reconsideration_status == "RECONSIDER":

                    st.warning(
                        "⚠️ Previous finding should be reconsidered."
                    )

                    for warning in warnings:

                        st.warning(
                            warning
                        )

                elif reconsideration_status == "STILL_VALID":

                    st.success(
                        "✅ Previous finding is still valid."
                    )

                else:

                    st.info(
                        "No previous finding was supplied."
                    )

                # ------------------------------------------------
                # FINAL RECOMMENDATION
                # ------------------------------------------------

                recommendation = result.get(
                    "recommendation",
                    "UNKNOWN"
                )

                st.divider()

                st.markdown(
                    "## 🎯 Final Recommendation"
                )

                if recommendation == (
                    "READY FOR FURTHER ANALYSIS"
                ):

                    st.success(
                        "🟢 READY FOR FURTHER ANALYSIS"
                    )

                elif recommendation == (
                    "NEEDS ATTENTION BEFORE ANALYSIS"
                ):

                    st.warning(
                        "🟡 NEEDS ATTENTION BEFORE ANALYSIS"
                    )

                elif recommendation == (
                    "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"
                ):

                    st.error(
                        "🔴 NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"
                    )

                else:

                    st.info(
                        recommendation
                    )

                st.caption(
                    "Recommendation generated by the dataset intelligence backend."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>AI Dataset Intelligence</strong><br>
        Automated Profiling • Quality Intelligence •
        Interactive Visualization • Machine Learning •
        Executive AI • Adaptive Dataset Change Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
