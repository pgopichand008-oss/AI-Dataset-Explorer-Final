"""
engine/visualization.py — Data-driven 2D and 3D visualization engine.
Ensures zero fake data, explicit observational filtering, intelligent chart recommendations,
and data-driven 3D exploration (Scatter, Surface, Mesh).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_numerical_histogram(df: pd.DataFrame, column: str):
    return px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        labels={column: column},
    )


def create_numerical_boxplot(df: pd.DataFrame, column: str):
    return px.box(
        df,
        y=column,
        title=f"Box Plot of {column}",
        labels={column: column},
    )


def create_correlation_heatmap(df: pd.DataFrame, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = list(df.select_dtypes(include="number").columns)

    if len(numeric_columns) < 2:
        return None

    correlation = df[numeric_columns].corr()

    return px.imshow(
        correlation,
        text_auto=".2f",
        title="Correlation Heatmap",
        aspect="auto",
        zmin=-1,
        zmax=1,
    )


def create_scatter_matrix(df: pd.DataFrame, numeric_columns=None):
    if numeric_columns is None:
        numeric_columns = list(df.select_dtypes(include="number").columns)

    if len(numeric_columns) < 2:
        return None

    return px.scatter_matrix(
        df,
        dimensions=numeric_columns,
        title="Multidimensional Explorer",
    )


def create_categorical_frequency(df: pd.DataFrame, column: str):
    counts = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Category", "Frequency"]

    return px.bar(
        counts,
        x="Category",
        y="Frequency",
        title=f"Frequency of {column}",
    )


def create_temporal_trend(df: pd.DataFrame, date_column: str, value_column: str):
    temp = df[[date_column, value_column]].copy()

    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp[value_column] = pd.to_numeric(temp[value_column], errors="coerce")

    temp = temp.dropna(subset=[date_column, value_column]).sort_values(date_column)

    if temp.empty:
        return None

    return px.line(
        temp,
        x=date_column,
        y=value_column,
        title=f"Temporal Trend: {value_column} over {date_column}",
    )


def create_missing_value_chart(df: pd.DataFrame):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        return None

    chart_df = missing.reset_index()
    chart_df.columns = ["Attribute", "Missing Values"]

    return px.bar(
        chart_df,
        x="Attribute",
        y="Missing Values",
        title="Missing Value Distribution",
    )


def create_binary_class_balance(df: pd.DataFrame, column: str):
    counts = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .reset_index()
    )

    counts.columns = ["Class", "Count"]

    return px.bar(
        counts,
        x="Class",
        y="Count",
        title=f"Class Balance: {column}",
    )


# ============================================================
# DATA-DRIVEN 3D VISUALIZATIONS
# ============================================================

def create_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str | None = None,
    max_points: int = 3000,
) -> tuple[go.Figure | None, dict]:
    """
    Generate an interactive 3D scatter plot using actual dataset values.
    Returns (figure, stats_dict).
    """
    required_cols = [x_col, y_col, z_col]
    if color_col and color_col in df.columns:
        required_cols.append(color_col)

    # Coerce numeric coordinates safely
    clean_df = df.copy()
    for c in [x_col, y_col, z_col]:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

    valid_mask = clean_df[[x_col, y_col, z_col]].notna().all(axis=1)
    filtered_df = clean_df.loc[valid_mask, required_cols]

    total_rows = len(df)
    valid_rows = len(filtered_df)
    excluded_rows = total_rows - valid_rows

    stats_meta = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "excluded_rows": excluded_rows,
        "x_col": x_col,
        "y_col": y_col,
        "z_col": z_col,
        "color_col": color_col,
    }

    if valid_rows == 0:
        return None, stats_meta

    if valid_rows > max_points:
        sampled_df = filtered_df.sample(max_points, random_state=42)
    else:
        sampled_df = filtered_df

    kwargs = {
        "data_frame": sampled_df,
        "x": x_col,
        "y": y_col,
        "z": z_col,
        "title": f"3D Scatter: {x_col} × {y_col} × {z_col}",
        "opacity": 0.8,
    }

    if color_col and color_col in sampled_df.columns:
        kwargs["color"] = color_col

    fig = px.scatter_3d(**kwargs)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
    )

    return fig, stats_meta


def create_3d_surface(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    grid_size: int = 30,
) -> tuple[go.Figure | None, dict]:
    """
    Generate a 3D Surface plot by gridding real continuous data points.
    Returns (figure, stats_dict).
    """
    clean_df = df.copy()
    for c in [x_col, y_col, z_col]:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

    valid_df = clean_df.dropna(subset=[x_col, y_col, z_col])

    total_rows = len(df)
    valid_rows = len(valid_df)
    excluded_rows = total_rows - valid_rows

    stats_meta = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "excluded_rows": excluded_rows,
        "x_col": x_col,
        "y_col": y_col,
        "z_col": z_col,
    }

    if valid_rows < 10:
        return None, stats_meta

    try:
        # Create a 2D grid over X and Y and interpolate Z values
        x_min, x_max = valid_df[x_col].min(), valid_df[x_col].max()
        y_min, y_max = valid_df[y_col].min(), valid_df[y_col].max()

        if x_min == x_max or y_min == y_max:
            return None, stats_meta

        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Simple binned average for surface grid Z
        x_bins = pd.cut(valid_df[x_col], bins=grid_size, labels=False)
        y_bins = pd.cut(valid_df[y_col], bins=grid_size, labels=False)

        grid_z = np.full((grid_size, grid_size), np.nan)
        binned = valid_df.groupby([y_bins, x_bins])[z_col].mean()

        for (yb, xb), val in binned.items():
            if pd.notna(yb) and pd.notna(xb) and 0 <= int(yb) < grid_size and 0 <= int(xb) < grid_size:
                grid_z[int(yb), int(xb)] = val

        # Forward/backward fill missing grid cells smoothly
        z_series = pd.DataFrame(grid_z).ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0).values

        fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=z_series, colorscale="Viridis")])
        fig.update_layout(
            title=f"3D Surface Grid: {z_col} over ({x_col}, {y_col})",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        )
        return fig, stats_meta
    except Exception:
        return None, stats_meta


def create_3d_mesh(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str | None = None,
) -> tuple[go.Figure | None, dict]:
    """
    Generate a 3D Mesh plot for continuous spatial / point cloud data.
    Returns (figure, stats_dict).
    """
    clean_df = df.copy()
    for c in [x_col, y_col, z_col]:
        clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

    valid_df = clean_df.dropna(subset=[x_col, y_col, z_col])

    total_rows = len(df)
    valid_rows = len(valid_df)
    excluded_rows = total_rows - valid_rows

    stats_meta = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "excluded_rows": excluded_rows,
        "x_col": x_col,
        "y_col": y_col,
        "z_col": z_col,
    }

    if valid_rows < 4:
        return None, stats_meta

    if valid_rows > 2000:
        plot_df = valid_df.sample(2000, random_state=42)
    else:
        plot_df = valid_df

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=plot_df[x_col],
                y=plot_df[y_col],
                z=plot_df[z_col],
                intensity=plot_df[z_col],
                colorscale="Viridis",
                opacity=0.6,
            )
        ]
    )

    fig.update_layout(
        title=f"3D Mesh: {x_col} × {y_col} × {z_col}",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig, stats_meta


# ============================================================
# AUTOMATIC VISUALIZATION RECOMMENDATIONS
# ============================================================

def recommend_visualizations(df: pd.DataFrame) -> list[dict]:
    """
    Intelligent visualization selector based on dataset characteristics:
    numeric count, categorical count, datetime count, row count, correlations.
    """
    recommendations = []

    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    date_cols = [
        col for col in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower()
    ]

    # 1. 3D Scatter recommendation
    if len(numeric_cols) >= 3:
        recommendations.append({
            "title": "3D Scatter Visualization",
            "type": "3D Scatter",
            "reason": f"Dataset contains {len(numeric_cols)} numeric attributes suitable for interactive 3D spatial relationship analysis.",
            "recommended_cols": numeric_cols[:3],
            "badge": "RECOMMENDED 3D",
        })

    # 2. Correlation Heatmap
    if len(numeric_cols) >= 2:
        recommendations.append({
            "title": "Correlation Heatmap",
            "type": "Heatmap",
            "reason": f"Evaluates pair-wise linear relationships across {len(numeric_cols)} numerical features.",
            "recommended_cols": numeric_cols[:5],
            "badge": "CORRELATION",
        })

    # 3. Categorical Distribution
    if len(cat_cols) > 0:
        recommendations.append({
            "title": "Categorical Frequency Bar Chart",
            "type": "Bar Chart",
            "reason": f"Discovers class frequencies and imbalances across {len(cat_cols)} categorical attribute(s).",
            "recommended_cols": [cat_cols[0]],
            "badge": "CATEGORICAL",
        })

    # 4. Missing Value Analysis
    missing_total = df.isna().sum().sum()
    if missing_total > 0:
        recommendations.append({
            "title": "Missing Value Analysis",
            "type": "Missingness Chart",
            "reason": f"Identifies {missing_total:,} missing value cells across columns to target imputation.",
            "recommended_cols": [],
            "badge": "DATA QUALITY",
        })

    # 5. Temporal Trend
    if date_cols and len(numeric_cols) > 0:
        recommendations.append({
            "title": "Temporal Trend Line Chart",
            "type": "Line Chart",
            "reason": f"Tracks attribute dynamics over date/time dimension '{date_cols[0]}'.",
            "recommended_cols": [date_cols[0], numeric_cols[0]],
            "badge": "TIME SERIES",
        })

    return recommendations


def get_graph_guide(
    chart_type: str,
    x_col: str,
    y_col: str,
    z_col: str | None = None,
    color_col: str | None = None,
) -> dict:
    """
    Returns structured plain-text 📖 GRAPH GUIDE metadata.
    Contains ONLY human-readable plain text without any HTML tags or code strings.
    """
    if chart_type == "3D Scatter":
        return {
            "title": "3D Scatter Spatial Relationship Plot",
            "what_it_represents": f"Visualizes 3D spatial points positioned across three numerical dimensions ({x_col}, {y_col}, {z_col}).",
            "x_axis": f"Primary spatial axis representing '{x_col}'.",
            "y_axis": f"Secondary spatial axis representing '{y_col}'.",
            "z_axis": f"Vertical spatial dimension representing '{z_col}'.",
            "color_meaning": f"Categories/groups encoded by color '{color_col}'." if color_col else "Uniform observation markers.",
            "what_to_look_for": "Look for 3D point clusters, spatial hyperplanes, tight density regions, and isolated spatial outliers.",
            "how_to_interpret": "Dense 3D point clouds reveal multivariate subgrouping; isolated points highlight multi-dimensional anomalies.",
            "data_science_insight": "Spatial clustering in 3D feature space strongly suggests candidate segments for K-Means or DBSCAN clustering.",
        }
    elif chart_type == "3D Surface":
        return {
            "title": "3D Surface Grid Interpolation",
            "what_it_represents": f"Visualizes a continuous elevation surface interpolated across continuous grid dimensions ({x_col}, {y_col}).",
            "x_axis": f"Base grid coordinate representing '{x_col}'.",
            "y_axis": f"Base grid coordinate representing '{y_col}'.",
            "z_axis": f"Interpolated surface elevation representing '{z_col}'.",
            "color_meaning": "Viridis color gradient mapped directly to elevation height.",
            "what_to_look_for": "Look for elevation peaks, valleys, steep gradient slopes, and smooth trend surfaces across continuous space.",
            "how_to_interpret": "High peaks represent local maxima in Z value; steep slopes represent rapid rates of change relative to X and Y.",
            "data_science_insight": "Surface gradients help identify optimal response surfaces and interaction effects between numeric predictors.",
        }
    elif chart_type == "3D Mesh":
        return {
            "title": "3D Mesh Point Cloud Geometry",
            "what_it_represents": f"Visualizes volumetric point cloud geometry enclosing 3D data points ({x_col}, {y_col}, {z_col}).",
            "x_axis": f"Dimension X coordinate '{x_col}'.",
            "y_axis": f"Dimension Y coordinate '{y_col}'.",
            "z_axis": f"Dimension Z coordinate '{z_col}'.",
            "color_meaning": f"Surface intensity gradient mapped to '{z_col}'.",
            "what_to_look_for": "Look for boundary convex hulls, volumetric density, and spatial enclosure shapes.",
            "how_to_interpret": "The outer mesh envelope highlights the boundary limits of valid observation space.",
            "data_science_insight": "Boundary mesh shapes help define operational bounds and detect out-of-distribution inputs.",
        }
    elif chart_type in ["Histogram", "Distribution"]:
        return {
            "title": "Frequency Distribution Histogram",
            "what_it_represents": f"Shows how frequently numeric values occur across binned ranges for '{x_col}'.",
            "x_axis": f"Binned numerical value ranges for '{x_col}'.",
            "y_axis": "Number of dataset observations falling into each bin.",
            "what_to_look_for": "Look for unimodal or bimodal peaks, skewness (tail extending left or right), and zero-frequency gaps.",
            "how_to_interpret": "Taller bars indicate ranges containing more observations. A narrow peak indicates low variation around the center.",
            "data_science_insight": "A highly skewed distribution may require logarithmic or Box-Cox transformation before applying linear ML models.",
        }
    elif chart_type == "Boxplot":
        return {
            "title": "Box & Whisker Quantile Plot",
            "what_it_represents": f"Shows the median, quartiles, overall spread, and potential statistical outliers for '{y_col}'.",
            "x_axis": f"Grouping attribute or overall dataset ('{x_col if x_col else 'All Data'}').",
            "y_axis": f"Numerical scale of '{y_col}'.",
            "what_to_look_for": "Look for median line position, IQR box height, whisker length, and individual outlier points beyond whiskers.",
            "how_to_interpret": "The center line marks the Median (50th percentile). The box covers the Interquartile Range (Q1 to Q3). Points beyond 1.5*IQR are outliers.",
            "data_science_insight": "Comparing box heights across categories reveals variance heterogeneity (heteroscedasticity).",
        }
    elif chart_type == "Heatmap":
        return {
            "title": "Pairwise Correlation Heatmap",
            "what_it_represents": "Shows the strength and direction of linear relationships between numerical feature pairs.",
            "x_axis": "Numerical attribute 1.",
            "y_axis": "Numerical attribute 2.",
            "color_meaning": "Correlation scale ranging from -1.0 (strong inverse) to +1.0 (strong positive).",
            "what_to_look_for": "Look for dark green cells (+1.0), dark red cells (-1.0), and near-zero neutral regions.",
            "how_to_interpret": "Values near +1 indicate features increase together. Values near -1 indicate one increases as the other decreases.",
            "data_science_insight": "Highly correlated feature pairs (|r| > 0.85) indicate multicollinearity and candidates for feature reduction. Note: Correlation does not prove causation.",
        }
    elif chart_type == "Scatter":
        return {
            "title": "2D Relationship Scatter Plot",
            "what_it_represents": f"Each point represents one observation showing the relationship between '{x_col}' and '{y_col}'.",
            "x_axis": f"Independent feature '{x_col}'.",
            "y_axis": f"Dependent measure '{y_col}'.",
            "color_meaning": f"Category breakdown by '{color_col}'." if color_col else "Standard observation series.",
            "what_to_look_for": "Look for upward or downward linear/non-linear trends, point clusters, and isolated outlier points.",
            "how_to_interpret": "An upward sloping trend indicates a positive relationship; downward slope indicates a negative relationship.",
            "data_science_insight": "Scatter plots reveal non-linear patterns (e.g. quadratic curves) that correlation coefficients miss. Note: Correlation does not prove causation.",
        }
    elif chart_type == "Bar Chart":
        return {
            "title": "Categorical Frequency / Aggregate Bar Chart",
            "what_it_represents": f"Compares frequency counts or aggregate metric values across categories in '{x_col}'.",
            "x_axis": f"Category labels for '{x_col}'.",
            "y_axis": f"Frequency / Metric magnitude ('{y_col}').",
            "what_to_look_for": "Look for the tallest bar (dominant category), smallest bar (rare category), and group differences.",
            "how_to_interpret": "Bar height corresponds directly to total count or metric average for that category.",
            "data_science_insight": "Extreme category dominance (>80%) indicates class imbalance that may require resampling for ML classification.",
        }
    else:
        return {
            "title": "Analytical Chart View",
            "what_it_represents": f"Analytical visualization of attributes '{x_col}' and '{y_col}'.",
            "x_axis": f"Attribute '{x_col}'.",
            "y_axis": f"Attribute '{y_col}'.",
            "what_to_look_for": "Examine overall pattern, central tendency, and extreme observations.",
            "how_to_interpret": "Evaluate value distributions and structural patterns.",
            "data_science_insight": "Compare observed patterns against business domain expectations.",
        }