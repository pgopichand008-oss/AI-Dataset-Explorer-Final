import pandas as pd


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0
    return (numerator / denominator) * 100


def infer_column_role(df, column):
    series = df[column]

    # Date / Time
    if pd.api.types.is_datetime64_any_dtype(series):
        return (
            "Date / Time",
            0.95,
            False,
            "Detected datetime column"
        )

    # Try detecting date-like object columns
    if series.dtype == "object":
        try:
            converted = pd.to_datetime(
                series.dropna(),
                errors="coerce"
            )

            if len(converted) > 0:
                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.8:
                    return (
                        "Date / Time",
                        0.85,
                        False,
                        "Values appear to represent dates or timestamps"
                    )
        except Exception:
            pass

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):

        unique_count = series.nunique(dropna=True)
        total_count = len(series)

        if total_count > 0:
            unique_ratio = unique_count / total_count
        else:
            unique_ratio = 0

        # Possible target
        if unique_count == 2:
            return (
                "Numerical",
                0.90,
                True,
                "Binary numeric column; possible classification target"
            )

        if unique_count <= 10:
            return (
                "Numerical",
                0.85,
                True,
                "Low-cardinality numerical column; possible target"
            )

        return (
            "Numerical",
            0.90,
            False,
            "Continuous numerical feature"
        )

    # Categorical columns
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):

        unique_count = series.nunique(dropna=True)
        total_count = len(series)

        if total_count > 0:
            unique_ratio = unique_count / total_count
        else:
            unique_ratio = 0

        # Identifier-like column
        if unique_ratio >= 0.95 and unique_count > 10:
            return (
                "Identifier",
                0.90,
                False,
                "Very high uniqueness suggests an identifier column"
            )

        # Possible target
        if 2 <= unique_count <= 10:
            return (
                "Categorical",
                0.85,
                True,
                "Low-cardinality categorical column; possible target"
            )

        return (
            "Categorical",
            0.90,
            False,
            "Categorical feature"
        )

    # Fallback
    return (
        "Other",
        0.50,
        False,
        "Column type could not be confidently classified"
    )


def clean_ai_text(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("```markdown", "")
    text = text.replace("```text", "")
    text = text.replace("```", "")

    return text.strip()


def split_ai_sections(text):
    sections = {}

    if not text:
        return sections

    current_section = "General"
    sections[current_section] = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            heading = line.lstrip("#").strip()

            if heading:
                current_section = heading
                sections[current_section] = []

        else:
            sections[current_section].append(line)

    for key in sections:
        sections[key] = "\n".join(sections[key])

    return sections


def prepare_safe_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str], bool]:
    """
    Creates a position-safe internal dataframe for calculation and visualization
    without altering the original dataframe.

    Returns:
        (safe_df, display_options, option_to_safe_col_map, has_duplicates)
    """
    if df is None or df.empty:
        return df, [], {}, False

    has_duplicates = bool(df.columns.duplicated().any())
    col_counts = {}
    new_cols = []
    display_options = []
    option_to_safe_col_map = {}

    for idx, col in enumerate(df.columns):
        col_str = str(col)
        col_counts[col_str] = col_counts.get(col_str, 0) + 1
        count = col_counts[col_str]

        if has_duplicates:
            safe_name = f"{col_str} [Col #{idx + 1}]" if count > 1 or list(df.columns).count(col) > 1 else col_str
            display_label = f"{col_str} — Column #{idx + 1}" if list(df.columns).count(col) > 1 else col_str
        else:
            safe_name = col_str
            display_label = col_str

        new_cols.append(safe_name)
        display_options.append(display_label)
        option_to_safe_col_map[display_label] = safe_name

    safe_df = df.copy()
    safe_df.columns = new_cols

    return safe_df, display_options, option_to_safe_col_map, has_duplicates


def validate_chart_inputs(
    df: pd.DataFrame,
    x_col: str | None,
    y_col: str | None,
    z_col: str | None = None,
    chart_type: str = "Scatter",
) -> tuple[bool, str]:
    """
    Validates chart selection before passing data to Plotly.
    Prevents runtime crashes, tracebacks, and ambiguous duplicate column errors.
    """
    if df is None or df.empty:
        return False, "⚠️ Visualization unavailable: The dataset is empty or invalid."

    # Validate X
    if not x_col or x_col not in df.columns:
        return False, f"⚠️ Visualization unavailable: Selected X-axis '{x_col}' does not exist."

    # Validate Y
    if y_col and y_col not in df.columns:
        return False, f"⚠️ Visualization unavailable: Selected Y-axis '{y_col}' does not exist."

    # Validate Z for 3D
    if "3D" in chart_type:
        if not z_col or z_col not in df.columns:
            return False, "⚠️ 3D Visualization unavailable: Requires three distinct numeric columns (X, Y, and Z)."

        if x_col == y_col or x_col == z_col or y_col == z_col:
            return False, "⚠️ 3D Visualization unavailable: X, Y, and Z axes must represent three distinct attributes."

        # Check numeric data types for 3D
        for c in [x_col, y_col, z_col]:
            if not pd.api.types.is_numeric_dtype(df[c]):
                try:
                    pd.to_numeric(df[c], errors="raise")
                except Exception:
                    return False, f"⚠️ 3D Visualization unavailable: Column '{c}' contains non-numeric values."

        valid_count = len(df[[x_col, y_col, z_col]].dropna())
        if valid_count < 3:
            return False, f"⚠️ 3D Visualization unavailable: Insufficient valid 3D observations (found {valid_count}, minimum 3 required)."

    # 2D relationship checks
    if chart_type in ["Scatter", "Line Chart", "Boxplot"] and x_col and y_col:
        valid_count = len(df[[x_col, y_col]].dropna())
        if valid_count < 2:
            return False, f"⚠️ Visualization unavailable: Insufficient valid observations for '{x_col}' and '{y_col}' (found {valid_count})."

    return True, ""


def format_report_for_download(report_text: str) -> str:
    """
    Formats Executive AI Report for plain-text / Markdown downloads
    with clear section heading hierarchy and zero raw HTML string tags.
    """
    if not report_text:
        return ""

    lines = report_text.splitlines()
    formatted = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading_title = stripped.lstrip("#").strip().upper()
            formatted.append("")
            formatted.append("=" * 60)
            formatted.append(f"# {heading_title}")
            formatted.append("=" * 60)
        else:
            clean_line = (
                stripped
                .replace("<div>", "").replace("</div>", "")
                .replace("<strong>", "").replace("</strong>", "")
                .replace("<span", "").replace("</span>", "")
            )
            formatted.append(clean_line)

    return "\n".join(formatted)
