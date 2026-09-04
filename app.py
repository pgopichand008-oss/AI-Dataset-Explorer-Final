import os
import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai


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
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

client = None

if api_key:
    client = genai.Client(api_key=api_key)


# ============================================================
# SESSION STATE
# ============================================================

if "ai_report" not in st.session_state:
    st.session_state.ai_report = ""

if "loaded_file_name" not in st.session_state:
    st.session_state.loaded_file_name = ""

if "analysis_status" not in st.session_state:
    st.session_state.analysis_status = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_percentage(value, total):
    if total == 0:
        return 0.0

    return (
        float(value)
        / float(total)
        * 100.0
    )


def infer_column_role(df, column):

    series = df[column]

    name = str(
        column
    ).strip().lower()

    rows = len(df)

    unique_count = int(
        series.nunique(
            dropna=True
        )
    )

    unique_ratio = (
        unique_count / rows
        if rows > 0
        else 0
    )

    identifier_keywords = [
        "id",
        "identifier",
        "customer_id",
        "user_id",
        "student_id",
        "employee_id",
        "account_id",
        "record_id",
        "transaction_id",
        "code"
    ]

    date_keywords = [
        "date",
        "time",
        "timestamp",
        "datetime",
        "year",
        "month",
        "day"
    ]

    target_keywords = [
        "target",
        "label",
        "class",
        "outcome",
        "result",
        "response",
        "churn",
        "default",
        "status"
    ]

    possible_target = any(
        word in name
        for word in target_keywords
    )

    if (
        pd.api.types.is_datetime64_any_dtype(series)
        or any(
            word in name
            for word in date_keywords
        )
    ):

        parsed = pd.to_datetime(
            series,
            errors="coerce"
        )

        if (
            rows > 0
            and parsed.notna().mean() >= 0.70
        ):

            return (
                "Date / Time",
                "High",
                possible_target,
                "Temporal information detected."
            )

    if (
        any(
            word == name
            or word in name
            for word in identifier_keywords
        )
        and unique_ratio >= 0.80
    ):

        return (
            "Identifier",
            "High",
            possible_target,
            "High uniqueness and identifier-like naming detected."
        )

    if pd.api.types.is_bool_dtype(series):

        return (
            "Binary Feature",
            "High",
            possible_target,
            "Boolean feature detected."
        )

    if pd.api.types.is_numeric_dtype(series):

        if unique_count <= 2:

            return (
                "Binary Feature",
                "High",
                possible_target,
                "Numeric field contains at most two distinct values."
            )

        return (
            "Numerical Feature",
            "High",
            possible_target,
            "Numeric feature with multiple distinct values detected."
        )

    if (
        unique_count <= 20
        or unique_ratio <= 0.20
    ):

        return (
            "Categorical Feature",
            "High",
            possible_target,
            "Low-to-moderate category diversity detected."
        )

    return (
        "Categorical Feature",
        "Medium",
        possible_target,
        "Categorical field detected."
    )


def clean_ai_text(text):

    if not text:
        return ""

    text = re.sub(
        r"```[\s\S]*?```",
        "",
        text
    )

    text = text.replace(
        "```",
        ""
    )

    text = re.sub(
        r"<[^>]+>",
        "",
        text
    )

    return text.strip()


def split_ai_sections(report):

    headings = [
        "EXECUTIVE SUMMARY",
        "DATASET DESCRIPTION",
        "COLUMN INTELLIGENCE",
        "NUMERICAL INSIGHTS",
        "CATEGORICAL INSIGHTS",
        "DATA QUALITY",
        "ANOMALY FINDINGS",
        "RELATIONSHIP INSIGHTS",
        "ML READINESS",
        "KEY INSIGHTS",
        "RECOMMENDED ACTIONS"
    ]

    matches = []

    for heading in headings:

        match = re.search(
            rf"(?im)^\s*(?:#+\s*)?{re.escape(heading)}\s*:?\s*$",
            report
        )

        if match:

            matches.append(
                (
                    match.start(),
                    match.end(),
                    heading
                )
            )

    matches.sort(
        key=lambda item: item[0]
    )

    sections = []

    for index, (
        start,
        end,
        heading
    ) in enumerate(matches):

        if index + 1 < len(matches):

            next_start = matches[
                index + 1
            ][0]

        else:

            next_start = len(
                report
            )

        body = report[
            end:next_start
        ].strip()

        body = body.replace(
            "```",
            ""
        )

        if body:

            sections.append(
                (
                    heading,
                    body
                )
            )

    return sections


# ============================================================
# HEADER
# ============================================================

st.title(
    "🤖 AI Dataset Intelligence"
)

st.caption(
    "From raw data to actionable intelligence."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📂 Dataset Workspace"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=[
            "csv",
            "xlsx"
        ]
    )

    st.divider()

    st.subheader(
        "AI Analysis Pipeline"
    )

    st.write(
        "① Dataset Discovery"
    )

    st.write(
        "② Column Intelligence"
    )

    st.write(
        "③ Quality & Anomalies"
    )

    st.write(
        "④ Smart Visual Analysis"
    )

    st.write(
        "⑤ Executive Intelligence"
    )

    st.divider()

    if api_key:

        st.success(
            "🟢 AI engine connected"
        )

    else:

        st.warning(
            "🟡 AI engine not connected"
        )


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV or Excel dataset from the sidebar to begin."
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

try:

    if uploaded_file.name.lower().endswith(
        ".csv"
    ):

        df = pd.read_csv(
            uploaded_file
        )

    else:

        df = pd.read_excel(
            BytesIO(
                uploaded_file.getvalue()
            )
        )

except Exception as e:

    st.error(
        f"Unable to read the dataset: {e}"
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded dataset contains no records."
    )

    st.stop()


if (
    st.session_state.loaded_file_name
    != uploaded_file.name
):

    st.session_state.loaded_file_name = (
        uploaded_file.name
    )

    st.session_state.ai_report = ""

    st.session_state.analysis_status = False


# ============================================================
# BASIC PROFILE
# ============================================================

rows = len(df)

columns = len(
    df.columns
)

missing_cells = int(
    df.isna().sum().sum()
)

duplicate_rows = int(
    df.duplicated().sum()
)

numeric_df = df.select_dtypes(
    include="number"
)

categorical_columns = df.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns


# ============================================================
# COLUMN INTELLIGENCE
# ============================================================

column_rows = []

identifier_columns = []

date_columns = []

target_candidates = []

for column in df.columns:

    (
        role,
        confidence,
        possible_target,
        reason
    ) = infer_column_role(
        df,
        column
    )

    if role == "Identifier":

        identifier_columns.append(
            column
        )

    if role == "Date / Time":

        date_columns.append(
            column
        )

    if possible_target:

        target_candidates.append(
            column
        )

    column_rows.append(
        {
            "Attribute": column,
            "Role": role,
            "Confidence": confidence,
            "Missing %": round(
                safe_percentage(
                    df[column].isna().sum(),
                    rows
                ),
                2
            ),
            "Unique %": round(
                safe_percentage(
                    df[column].nunique(
                        dropna=True
                    ),
                    rows
                ),
                2
            ),
            "Potential Target": (
                "Yes"
                if possible_target
                else "No"
            ),
            "Reason": reason
        }
    )


column_intelligence_df = pd.DataFrame(
    column_rows
)


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

if not numeric_df.empty:

    numerical_summary = (
        numeric_df
        .describe()
        .T
    )

    numerical_summary[
        "median"
    ] = numeric_df.median()

    numerical_summary[
        "range"
    ] = (
        numerical_summary[
            "max"
        ]
        -
        numerical_summary[
            "min"
        ]
    )

    numerical_summary[
        "skewness"
    ] = numeric_df.skew()

else:

    numerical_summary = pd.DataFrame()


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

categorical_rows = []

for column in categorical_columns:

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
    )

    if counts.empty:

        top_category = "N/A"

        top_frequency = 0

    else:

        top_category = str(
            counts.index[0]
        )

        top_frequency = int(
            counts.iloc[0]
        )

    categorical_rows.append(
        {
            "Attribute": column,
            "Unique Values": int(
                df[column].nunique(
                    dropna=True
                )
            ),
            "Top Category": top_category,
            "Frequency": top_frequency,
            "Dominance %": round(
                safe_percentage(
                    top_frequency,
                    rows
                ),
                2
            )
        }
    )


categorical_summary_df = pd.DataFrame(
    categorical_rows
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

anomaly_rows = []

outlier_columns = []

total_outlier_values = 0

for column in numeric_df.columns:

    series = (
        numeric_df[column]
        .dropna()
    )

    if series.empty:

        continue

    q1 = series.quantile(
        0.25
    )

    q3 = series.quantile(
        0.75
    )

    iqr = q3 - q1

    if iqr <= 0:

        continue

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    mask = (
        (numeric_df[column] < lower_bound)
        |
        (numeric_df[column] > upper_bound)
    )

    count = int(
        mask.sum()
    )

    percentage = safe_percentage(
        count,
        rows
    )

    if count > 0:

        outlier_columns.append(
            column
        )

        total_outlier_values += count

        anomaly_rows.append(
            {
                "Attribute": column,
                "Outlier Count": count,
                "Outlier %": round(
                    percentage,
                    2
                ),
                "Lower Bound": round(
                    lower_bound,
                    3
                ),
                "Upper Bound": round(
                    upper_bound,
                    3
                )
            }
        )


anomaly_df = pd.DataFrame(
    anomaly_rows
)


# ============================================================
# QUALITY FINDINGS
# ============================================================

missing_ratio = safe_percentage(
    missing_cells,
    rows * columns
)

duplicate_ratio = safe_percentage(
    duplicate_rows,
    rows
)

quality_rows = []

for column in df.columns:

    missing = int(
        df[column].isna().sum()
    )

    if missing > 0:

        percentage = safe_percentage(
            missing,
            rows
        )

        quality_rows.append(
            {
                "Issue": "Missing Values",
                "Attribute": column,
                "Severity": (
                    "High"
                    if percentage >= 10
                    else "Medium"
                ),
                "Details": (
                    f"{missing} missing values "
                    f"({percentage:.2f}%)"
                )
            }
        )


if duplicate_rows > 0:

    quality_rows.append(
        {
            "Issue": "Duplicate Rows",
            "Attribute": "Dataset",
            "Severity": (
                "High"
                if duplicate_ratio >= 10
                else "Medium"
            ),
            "Details": (
                f"{duplicate_rows} duplicate rows "
                f"({duplicate_ratio:.2f}%)"
            )
        }
    )


for column in df.columns:

    if df[column].isna().all():

        quality_rows.append(
            {
                "Issue": "Empty Column",
                "Attribute": column,
                "Severity": "High",
                "Details": (
                    "No usable values detected."
                )
            }
        )


for column in df.columns:

    if df[column].nunique(
        dropna=False
    ) <= 1:

        quality_rows.append(
            {
                "Issue": "Constant Column",
                "Attribute": column,
                "Severity": "Medium",
                "Details": (
                    "Only one distinct value detected."
                )
            }
        )


for column in categorical_columns:

    unique_count = int(
        df[column].nunique()
    )

    unique_ratio = (
        unique_count
        / rows
        if rows > 0
        else 0
    )

    if unique_ratio >= 0.50:

        quality_rows.append(
            {
                "Issue": "High Cardinality",
                "Attribute": column,
                "Severity": "Medium",
                "Details": (
                    f"{unique_count} unique categories."
                )
            }
        )


quality_df = pd.DataFrame(
    quality_rows
)


# ============================================================
# QUALITY SCORE
# ============================================================

constant_count = sum(
    item["Issue"] == "Constant Column"
    for item in quality_rows
)

quality_score = 100

quality_score -= min(
    missing_ratio * 0.60,
    30
)

quality_score -= min(
    duplicate_ratio * 0.40,
    15
)

quality_score -= min(
    len(outlier_columns) * 3,
    15
)

quality_score -= min(
    constant_count * 5,
    15
)

quality_score = int(
    max(
        0,
        min(
            100,
            round(
                quality_score
            )
        )
    )
)


if quality_score >= 90:

    quality_status = "Excellent"

elif quality_score >= 75:

    quality_status = "Good"

elif quality_score >= 50:

    quality_status = "Moderate"

else:

    quality_status = "Poor"


# ============================================================
# ML READINESS
# ============================================================

ml_score = 100

if rows < 100:

    ml_score -= 15

elif rows < 500:

    ml_score -= 8

ml_score -= min(
    missing_ratio * 0.75,
    25
)

ml_score -= min(
    duplicate_ratio * 0.50,
    15
)

ml_score -= min(
    len(outlier_columns) * 2,
    10
)

ml_score -= min(
    constant_count * 5,
    15
)

ml_score -= min(
    len(identifier_columns) * 2,
    6
)

ml_score = int(
    max(
        0,
        min(
            100,
            round(
                ml_score
            )
        )
    )
)


if ml_score >= 90:

    ml_status = "Highly Ready"

elif ml_score >= 75:

    ml_status = "Mostly Ready"

elif ml_score >= 50:

    ml_status = "Needs Cleaning"

else:

    ml_status = "Not Ready"


# ============================================================
# CORRELATION ENGINE
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

    for i in range(
        len(numeric_columns)
    ):

        for j in range(
            i + 1,
            len(numeric_columns)
        ):

            value = (
                correlation_matrix.iloc[i, j]
            )

            if pd.notna(value):

                pair_rows.append(
                    {
                        "Attribute 1":
                            numeric_columns[i],
                        "Attribute 2":
                            numeric_columns[j],
                        "Correlation":
                            value,
                        "Absolute Correlation":
                            abs(value)
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
# EXPLAINABLE AGENT DECISIONS
# ============================================================

agent_rows = []

for _, row in column_intelligence_df.iterrows():

    attribute = row["Attribute"]

    role = row["Role"]

    missing_pct = float(
        row["Missing %"]
    )

    unique_pct = float(
        row["Unique %"]
    )

    priority = "Low"

    focus = "Routine Review"

    reason = "No major issue detected."

    if missing_pct >= 30:

        priority = "Critical"

        focus = "Missingness Investigation"

        reason = (
            "A large percentage of values are missing."
        )

    elif missing_pct >= 10:

        priority = "High"

        focus = "Missingness Investigation"

        reason = (
            "Significant missingness may affect analysis."
        )

    elif missing_pct > 0:

        priority = "Medium"

        focus = "Missingness Review"

        reason = (
            "Some values require review."
        )

    if attribute in outlier_columns:

        priority = "High"

        focus = "Anomaly Investigation"

        reason = (
            "Potential numerical anomalies were detected."
        )

    if role == "Identifier":

        if priority == "Low":

            priority = "Medium"

        focus = "Identifier Review"

        reason = (
            "Identifier-like fields are generally not useful "
            "as predictive features."
        )

    elif role == "Date / Time":

        if priority == "Low":

            priority = "Medium"

        focus = "Temporal Analysis"

        reason = (
            "Temporal information may reveal useful trends."
        )

    elif role == "Categorical Feature":

        if unique_pct >= 50:

            priority = "High"

            focus = "Category Structure"

            reason = (
                "High category diversity may affect encoding."
            )

    agent_rows.append(
        {
            "Attribute": attribute,
            "Role": role,
            "Priority": priority,
            "Focus": focus,
            "Reason": reason
        }
    )


agent_priority_df = pd.DataFrame(
    agent_rows
)


# ============================================================
# SMART VISUALIZATION PLAN
# ============================================================

visual_plan_rows = []

for _, row in column_intelligence_df.iterrows():

    attribute = row["Attribute"]

    role = row["Role"]

    missing_pct = float(
        row["Missing %"]
    )

    if role == "Numerical Feature":

        visualization = (
            "Distribution + Box Plot"
        )

        reason = (
            "Shows spread, shape, and potential anomalies."
        )

        priority = "High"

        if attribute in outlier_columns:

            visualization = (
                "Distribution + Anomaly View"
            )

            priority = "Critical"

            reason = (
                "Potential anomalies require investigation."
            )

    elif role == "Categorical Feature":

        visualization = (
            "Category Frequency"
        )

        reason = (
            "Shows dominant categories and diversity."
        )

        priority = "Medium"

        if float(
            row["Unique %"]
        ) >= 50:

            priority = "High"

            reason = (
                "High category diversity detected."
            )

    elif role == "Binary Feature":

        visualization = (
            "Class Balance"
        )

        priority = "Medium"

        reason = (
            "Shows distribution between binary values."
        )

    elif role == "Date / Time":

        visualization = (
            "Temporal Trend"
        )

        priority = "High"

        reason = (
            "Shows changes over time."
        )

    else:

        visualization = (
            "No Dedicated Chart"
        )

        priority = "Low"

        reason = (
            "Limited analytical value for visualization."
        )

    if missing_pct >= 10:

        visual_plan_rows.append(
            {
                "Attribute": attribute,
                "Visualization": "Missing Value Analysis",
                "Priority": "High",
                "Reason": (
                    "Significant missingness detected."
                )
            }
        )

    visual_plan_rows.append(
        {
            "Attribute": attribute,
            "Visualization": visualization,
            "Priority": priority,
            "Reason": reason
        }
    )


visual_plan_df = pd.DataFrame(
    visual_plan_rows
)


# ============================================================
# EXECUTIVE SUMMARY ENGINE
# ============================================================

if (
    quality_score < 50
    or ml_score < 50
):

    risk_level = "High"

elif (
    quality_score < 75
    or ml_score < 75
):

    risk_level = "Medium"

else:

    risk_level = "Low"


high_quality_findings = (
    quality_df[
        quality_df["Severity"] == "High"
    ]
    if not quality_df.empty
    else pd.DataFrame()
)


if not high_quality_findings.empty:

    issue = high_quality_findings.iloc[0]

    most_important_finding = (
        f"{issue['Attribute']}: "
        f"{issue['Details']}"
    )

elif not anomaly_df.empty:

    anomaly = anomaly_df.iloc[0]

    most_important_finding = (
        f"{anomaly['Attribute']} contains "
        f"{anomaly['Outlier Count']} potential anomalies "
        f"({anomaly['Outlier %']:.2f}%)."
    )

elif duplicate_rows > 0:

    most_important_finding = (
        f"{duplicate_rows} duplicate records were detected."
    )

else:

    most_important_finding = (
        "No major high-priority data issue was detected."
    )


if not correlation_pairs.empty:

    strongest = correlation_pairs.iloc[0]

    key_opportunity = (
        f"{strongest['Attribute 1']} and "
        f"{strongest['Attribute 2']} have the strongest "
        f"observed numerical relationship "
        f"(correlation {strongest['Correlation']:.2f})."
    )

elif date_columns:

    key_opportunity = (
        f"Temporal analysis is available through "
        f"{date_columns[0]}."
    )

elif not numeric_df.empty:

    key_opportunity = (
        "The numerical attributes provide opportunities "
        "for deeper statistical and relationship analysis."
    )

else:

    key_opportunity = (
        "The categorical structure provides the main "
        "opportunity for deeper exploration."
    )


if quality_score < 75:

    next_best_action = (
        "Resolve the highest-priority data-quality issues "
        "before relying on downstream analysis."
    )

elif ml_score < 75:

    next_best_action = (
        "Perform preprocessing and review ML readiness "
        "before model development."
    )

elif outlier_columns:

    next_best_action = (
        "Investigate the detected numerical anomalies "
        "before downstream analysis."
    )

elif identifier_columns:

    next_best_action = (
        "Review identifier-like fields before using the "
        "dataset for machine-learning features."
    )

else:

    next_best_action = (
        "Proceed to deeper analysis; no major preparation "
        "barrier was detected."
    )


# ============================================================
# ANALYSIS STATUS
# ============================================================

st.subheader(
    "⚙️ Analysis Status"
)

status_cols = st.columns(5)

status_items = [
    (
        "Dataset",
        True
    ),
    (
        "Profile",
        True
    ),
    (
        "Quality",
        True
    ),
    (
        "Visuals",
        True
    ),
    (
        "AI",
        client is not None
    )
]

for container, (
    label,
    completed
) in zip(
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

metric_cols = st.columns(6)

metrics = [
    (
        "Rows",
        f"{rows:,}"
    ),
    (
        "Columns",
        f"{columns:,}"
    ),
    (
        "Missing",
        f"{missing_cells:,}"
    ),
    (
        "Duplicates",
        f"{duplicate_rows:,}"
    ),
    (
        "Quality",
        f"{quality_score}/100"
    ),
    (
        "ML Ready",
        f"{ml_score}/100"
    )
]

for container, (
    label,
    value
) in zip(
    metric_cols,
    metrics
):

    with container:

        st.metric(
            label,
            value
        )


st.divider()


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.header(
    "🧠 Executive Dataset Summary"
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

    st.subheader(
        "🔎 Most Important Finding"
    )

    st.info(
        most_important_finding
    )


with opportunity_col:

    st.subheader(
        "💡 Key Opportunity"
    )

    st.success(
        key_opportunity
    )


st.subheader(
    "🎯 Next Best Action"
)

st.warning(
    next_best_action
)


st.divider()


# ============================================================
# WORKSPACES
# ============================================================

overview_tab, quality_tab, visual_tab, intelligence_tab, ai_tab = st.tabs(
    [
        "🏠 Overview",
        "🛡️ Quality & Anomalies",
        "📊 Visual Exploration",
        "🧠 Intelligence",
        "💼 Executive AI"
    ]
)


# ============================================================
# OVERVIEW WORKSPACE
# ============================================================

with overview_tab:

    st.subheader(
        "Dataset Overview"
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
            width="stretch",
            hide_index=True
        )

    with c2:

        st.subheader(
            "🎯 Dataset Readiness"
        )

        st.metric(
            "ML Readiness",
            f"{ml_score}/100",
            ml_status
        )

        st.progress(
            ml_score / 100
        )

        st.metric(
            "Data Quality",
            f"{quality_score}/100",
            quality_status
        )

        st.progress(
            quality_score / 100
        )

    st.subheader(
        "👀 Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        width="stretch",
        hide_index=True
    )


# ============================================================
# QUALITY & ANOMALIES
# ============================================================

with quality_tab:

    st.subheader(
        "🛡️ Quality & Anomaly Command Center"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.metric(
            "Quality Score",
            f"{quality_score}/100"
        )

    with q2:

        st.metric(
            "Missing Data",
            f"{missing_ratio:.2f}%"
        )

    with q3:

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

    st.subheader(
        "Detected Quality Issues"
    )

    if not quality_df.empty:

        st.dataframe(
            quality_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.success(
            "✅ No major quality issues detected."
        )

    st.subheader(
        "🚨 Numerical Anomalies"
    )

    if not anomaly_df.empty:

        st.dataframe(
            anomaly_df,
            width="stretch",
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

    st.subheader(
        "📊 Deep Visual Exploration"
    )

    if not numeric_df.empty:

        st.markdown(
            "### Numerical Distribution Deep Dive"
        )

        selected_numeric = st.selectbox(
            "Select numerical attribute",
            list(
                numeric_df.columns
            )
        )

        series = (
            numeric_df[
                selected_numeric
            ]
            .dropna()
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            histogram = px.histogram(
                df,
                x=selected_numeric,
                nbins=30,
                marginal="box",
                title=(
                    f"{selected_numeric} Distribution"
                )
            )

            histogram.update_layout(
                height=430
            )

            st.plotly_chart(
                histogram,
                width="stretch"
            )

        with chart_col2:

            box = px.box(
                df,
                y=selected_numeric,
                points="outliers",
                title=(
                    f"{selected_numeric} Outlier View"
                )
            )

            box.update_layout(
                height=430
            )

            st.plotly_chart(
                box,
                width="stretch"
            )

        if not series.empty:

            stat_cols = st.columns(5)

            stats = [
                (
                    "Mean",
                    series.mean()
                ),
                (
                    "Median",
                    series.median()
                ),
                (
                    "Minimum",
                    series.min()
                ),
                (
                    "Maximum",
                    series.max()
                ),
                (
                    "Std Dev",
                    series.std()
                )
            ]

            for container, (
                label,
                value
            ) in zip(
                stat_cols,
                stats
            ):

                with container:

                    st.metric(
                        label,
                        f"{value:.3f}"
                    )


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
            step=0.05
        )

        filtered_matrix = correlation_matrix.copy()

        filtered_matrix = filtered_matrix.mask(
            filtered_matrix.abs()
            < threshold
        )

        correlation_chart = px.imshow(
            filtered_matrix,
            text_auto=".2f",
            aspect="auto",
            zmin=-1,
            zmax=1,
            title=(
                f"Relationships with |correlation| ≥ {threshold:.2f}"
            )
        )

        correlation_chart.update_layout(
            height=600
        )

        st.plotly_chart(
            correlation_chart,
            width="stretch"
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
                width="stretch",
                hide_index=True
            )

        st.markdown(
            "### 🔬 Multidimensional Explorer"
        )

        matrix_columns = st.multiselect(
            "Select 2–5 numerical attributes",
            list(
                numeric_df.columns
            ),
            default=list(
                numeric_df.columns[
                    :min(
                        4,
                        len(
                            numeric_df.columns
                        )
                    )
                ]
            ),
            max_selections=5
        )

        if len(matrix_columns) >= 2:

            matrix_df = (
                df[
                    matrix_columns
                ]
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

            matrix_chart = px.scatter_matrix(
                matrix_df,
                dimensions=matrix_columns,
                title=(
                    "Multidimensional Relationship Explorer"
                )
            )

            matrix_chart.update_layout(
                height=720
            )

            st.plotly_chart(
                matrix_chart,
                width="stretch"
            )

    if len(categorical_columns) > 0:

        st.divider()

        st.markdown(
            "### 📊 Categorical Exploration"
        )

        selected_category = st.selectbox(
            "Select categorical attribute",
            list(
                categorical_columns
            )
        )

        category_counts = (
            df[
                selected_category
            ]
            .value_counts(
                dropna=False
            )
            .head(15)
            .reset_index()
        )

        category_counts.columns = [
            "Category",
            "Count"
        ]

        category_chart = px.bar(
            category_counts,
            x="Category",
            y="Count",
            text="Count",
            title=(
                f"Top Categories — {selected_category}"
            )
        )

        category_chart.update_layout(
            height=450
        )

        category_chart.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            category_chart,
            width="stretch"
        )

    if date_columns:

        st.divider()

        st.markdown(
            "### 📅 Temporal Exploration"
        )

        selected_date = st.selectbox(
            "Select date/time attribute",
            date_columns
        )

        date_series = pd.to_datetime(
            df[
                selected_date
            ],
            errors="coerce"
        )

        temporal_df = pd.DataFrame(
            {
                "Date": date_series
            }
        ).dropna()

        if not temporal_df.empty:

            temporal_df[
                "Period"
            ] = (
                temporal_df[
                    "Date"
                ]
                .dt.to_period(
                    "M"
                )
                .astype(str)
            )

            trend_df = (
                temporal_df
                .groupby(
                    "Period"
                )
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
                        f"Record Trend — {selected_date}"
                    )
                )

                trend_chart.update_layout(
                    height=430
                )

                st.plotly_chart(
                    trend_chart,
                    width="stretch"
                )

    if missing_cells > 0:

        st.divider()

        st.markdown(
            "### 🕳️ Missing Data Pattern"
        )

        missing_plot = (
            df.isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        missing_plot = missing_plot[
            missing_plot > 0
        ]

        missing_plot_df = (
            missing_plot
            .reset_index()
        )

        missing_plot_df.columns = [
            "Attribute",
            "Missing Values"
        ]

        missing_plot_df[
            "Missing %"
        ] = (
            missing_plot_df[
                "Missing Values"
            ]
            / rows
            * 100
        )

        missing_chart = px.bar(
            missing_plot_df,
            x="Attribute",
            y="Missing Values",
            text=missing_plot_df[
                "Missing %"
            ].apply(
                lambda x: f"{x:.1f}%"
            ),
            title="Missing Values by Attribute"
        )

        missing_chart.update_layout(
            height=430
        )

        missing_chart.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            missing_chart,
            width="stretch"
        )


# ============================================================
# INTELLIGENCE WORKSPACE
# ============================================================

with intelligence_tab:

    st.subheader(
        "🧠 Explainable Dataset Intelligence"
    )

    st.markdown(
        "### Column Intelligence"
    )

    st.dataframe(
        column_intelligence_df,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### 🤖 Agent Decisions"
    )

    st.dataframe(
        agent_priority_df,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### 📊 Smart Visualization Plan"
    )

    st.dataframe(
        visual_plan_df,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### ⚠️ Why These Columns Matter"
    )

    important_agent_rows = agent_priority_df[
        agent_priority_df["Priority"].isin(
            [
                "Critical",
                "High"
            ]
        )
    ]

    if not important_agent_rows.empty:

        for _, row in (
            important_agent_rows
            .head(8)
            .iterrows()
        ):

            st.warning(
                f"**{row['Attribute']}** — "
                f"{row['Focus']}  \n"
                f"**Why:** {row['Reason']}"
            )

    else:

        st.success(
            "✅ No critical or high-priority attributes detected."
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
            "No obvious target field was detected from column naming."
        )


# ============================================================
# EXECUTIVE AI WORKSPACE
# ============================================================

with ai_tab:

    st.subheader(
        "💼 Executive AI Intelligence"
    )

    st.write(
        "The AI agent interprets the analytical evidence and "
        "produces an executive-level intelligence briefing."
    )

    if client is None:

        st.warning(
            "AI analysis is unavailable because the Gemini API "
            "key was not detected."
        )

        st.info(
            "The complete statistical, quality, intelligence, "
            "and visualization analysis remains available."
        )

    else:

        generate_ai = st.button(
            "✨ Generate Executive Intelligence",
            type="primary"
        )

        if generate_ai:

            with st.spinner(
                "🤖 AI Agent is investigating your dataset..."
            ):

                try:

                    attribute_text = (
                        column_intelligence_df
                        .to_string(
                            index=False
                        )
                    )

                    numerical_text = (
                        numerical_summary
                        .round(3)
                        .to_string()
                        if not numerical_summary.empty
                        else "No numerical statistics."
                    )

                    categorical_text = (
                        categorical_summary_df
                        .to_string(
                            index=False
                        )
                        if not categorical_summary_df.empty
                        else "No categorical statistics."
                    )

                    quality_text = (
                        quality_df
                        .to_string(
                            index=False
                        )
                        if not quality_df.empty
                        else "No quality issues detected."
                    )

                    anomaly_text = (
                        anomaly_df
                        .to_string(
                            index=False
                        )
                        if not anomaly_df.empty
                        else "No numerical anomalies detected."
                    )

                    agent_text = (
                        agent_priority_df
                        .to_string(
                            index=False
                        )
                    )

                    visual_text = (
                        visual_plan_df
                        .to_string(
                            index=False
                        )
                    )

                    correlation_text = (
                        correlation_pairs[
                            [
                                "Attribute 1",
                                "Attribute 2",
                                "Correlation"
                            ]
                        ]
                        .head(10)
                        .to_string(
                            index=False
                        )
                        if not correlation_pairs.empty
                        else "No strong numerical relationships detected."
                    )

                    prompt = f"""
You are an expert Autonomous Dataset Intelligence Agent.

Investigate the dataset using ONLY the supplied evidence.

Never invent facts.
Never invent numerical values.
Never fabricate business context.
Never claim correlation proves causation.
Never output code.
Never output HTML.
Never output JSON.
Never output Markdown tables.
Never use code fences.

DATASET
Name: {uploaded_file.name}
Rows: {rows}
Columns: {columns}

QUALITY
Score: {quality_score}/100
Status: {quality_status}

ML READINESS
Score: {ml_score}/100
Status: {ml_status}

RISK LEVEL
{risk_level}

COLUMN INTELLIGENCE
{attribute_text}

NUMERICAL STATISTICS
{numerical_text}

CATEGORICAL STATISTICS
{categorical_text}

QUALITY FINDINGS
{quality_text}

ANOMALY FINDINGS
{anomaly_text}

CORRELATION INFORMATION
{correlation_text}

AGENT DECISIONS
{agent_text}

VISUALIZATION PLAN
{visual_text}

EXECUTIVE SIGNALS

Most Important Finding:
{most_important_finding}

Key Opportunity:
{key_opportunity}

Next Best Action:
{next_best_action}

Create a professional executive intelligence report.

Use exactly these headings:

EXECUTIVE SUMMARY

Summarize:
- overall dataset condition
- most important risk
- strongest opportunity
- next best action

DATASET DESCRIPTION

Describe the dataset structure, size, major attribute
types, and likely purpose only when reasonably supported.

COLUMN INTELLIGENCE

Explain the most important column roles and why they matter.

NUMERICAL INSIGHTS

Explain the most meaningful numerical patterns.
Focus on averages, medians, ranges, spread, skewness,
and unusual values where supported.

CATEGORICAL INSIGHTS

Explain important categorical distributions,
dominant categories, diversity, and high-cardinality fields.

DATA QUALITY

Explain the important detected quality issues and their
potential effect on analysis.

ANOMALY FINDINGS

Explain the important numerical anomalies and why they
deserve investigation.

RELATIONSHIP INSIGHTS

Discuss meaningful numerical relationships supported by
the supplied evidence.

ML READINESS

Explain the ML readiness score and the factors affecting it.

KEY INSIGHTS

Provide exactly 5 specific, ranked, evidence-based insights.

RECOMMENDED ACTIONS

Provide exactly 5 prioritized actions.

Every action must directly correspond to an actual
finding or meaningful signal.

Use concise professional language suitable for an analyst,
manager, or executive.
"""

                    interaction = (
                        client.interactions.create(
                            model="gemini-3.6-flash",
                            input=prompt,
                            generation_config={
                                "thinking_level": "low"
                            }
                        )
                    )

                    report = getattr(
                        interaction,
                        "output_text",
                        ""
                    )

                    report = clean_ai_text(
                        report
                    )

                    if report:

                        st.session_state.ai_report = (
                            report
                        )

                        st.success(
                            "✅ Executive intelligence generated successfully."
                        )

                    else:

                        st.warning(
                            "The AI service returned no report."
                        )

                        st.info(
                            "Your dataset profiling, quality analysis, "
                            "visualizations, and intelligence results "
                            "are still available."
                        )

                except Exception as e:

                    error_text = str(
                        e
                    )

                    if (
                        "503" in error_text
                        or "UNAVAILABLE" in error_text
                    ):

                        st.warning(
                            "⚠️ The AI service is temporarily unavailable."
                        )

                        st.info(
                            "The dataset analysis is complete. "
                            "Please use the other workspaces while "
                            "the AI service is unavailable."
                        )

                    elif (
                        "404" in error_text
                        or "NOT_FOUND" in error_text
                    ):

                        st.error(
                            "⚠️ The configured Gemini model is unavailable."
                        )

                        st.info(
                            "Update the Gemini SDK/model configuration "
                            "before using AI analysis."
                        )

                    elif (
                        "401" in error_text
                        or "403" in error_text
                    ):

                        st.error(
                            "⚠️ Gemini authentication failed."
                        )

                        st.info(
                            "Check the GEMINI_API_KEY environment variable."
                        )

                    else:

                        st.error(
                            "⚠️ AI analysis could not be completed."
                        )

                        st.info(
                            "The rest of the dataset analysis is still available."
                        )

    if st.session_state.ai_report:

        st.divider()

        st.subheader(
            "📋 Executive Intelligence Report"
        )

        report_sections = split_ai_sections(
            st.session_state.ai_report
        )

        if report_sections:

            for heading, body in report_sections:

                st.markdown(
                    f"### {heading}"
                )

                st.write(
                    body
                )

                st.divider()

        else:

            st.write(
                st.session_state.ai_report
            )

        st.download_button(
            "⬇️ Download Executive Report",
            data=st.session_state.ai_report,
            file_name=(
                "AI_Dataset_Executive_Report.txt"
            ),
            mime="text/plain"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Dataset Intelligence • Automated Profiling • "
    "Explainable Agent Decisions • Anomaly Detection • "
    "Multidimensional Exploration • Executive Intelligence"
)