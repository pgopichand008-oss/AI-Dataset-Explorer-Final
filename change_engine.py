import pandas as pd


def _similar_values(old_series, new_series):
    """Check whether two columns contain mostly similar values."""

    old_values = set(old_series.dropna().astype(str).head(1000))
    new_values = set(new_series.dropna().astype(str).head(1000))

    if not old_values or not new_values:
        return 0.0

    intersection = len(old_values.intersection(new_values))
    union = len(old_values.union(new_values))

    return intersection / union if union else 0.0


def compare_datasets(old_df, new_df):
    """
    Compare the previous and updated datasets
    and detect structural changes.
    """

    old_columns = set(old_df.columns)
    new_columns = set(new_df.columns)

    added_columns = sorted(new_columns - old_columns)
    removed_columns = sorted(old_columns - new_columns)

    # Detect data-type changes
    type_changes = []

    for column in old_columns.intersection(new_columns):
        old_type = str(old_df[column].dtype)
        new_type = str(new_df[column].dtype)

        if old_type != new_type:
            type_changes.append({
                "column": column,
                "old_type": old_type,
                "new_type": new_type
            })

    # Detect possible renamed columns
    possible_renames = []

    for old_column in removed_columns:
        for new_column in added_columns:

            old_type = str(old_df[old_column].dtype)
            new_type = str(new_df[new_column].dtype)

            # Only compare columns with the same data type
            if old_type != new_type:
                continue

            similarity = _similar_values(
                old_df[old_column],
                new_df[new_column]
            )

            if similarity >= 0.5:
                possible_renames.append({
                    "old_column": old_column,
                    "new_column": new_column,
                    "confidence": round(similarity * 100, 1)
                })

    return {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
        "possible_renames": possible_renames
    }


def compare_statistics(old_df, new_df):
    """
    Compare basic numerical statistics between
    the previous and updated datasets.

    Only columns present in both datasets and
    numeric in both versions are compared.
    """

    common_columns = [
        column
        for column in old_df.columns
        if column in new_df.columns
    ]

    statistical_changes = []

    for column in common_columns:

        old_numeric = pd.api.types.is_numeric_dtype(old_df[column])
        new_numeric = pd.api.types.is_numeric_dtype(new_df[column])

        # Compare only columns that are numeric in both datasets
        if not old_numeric or not new_numeric:
            continue

        old_series = pd.to_numeric(
            old_df[column],
            errors="coerce"
        ).dropna()

        new_series = pd.to_numeric(
            new_df[column],
            errors="coerce"
        ).dropna()

        # Skip columns with no valid numerical values
        if old_series.empty or new_series.empty:
            continue

        old_mean = float(old_series.mean())
        new_mean = float(new_series.mean())

        old_median = float(old_series.median())
        new_median = float(new_series.median())

        old_std = float(old_series.std()) if len(old_series) > 1 else 0.0
        new_std = float(new_series.std()) if len(new_series) > 1 else 0.0

        old_min = float(old_series.min())
        new_min = float(new_series.min())

        old_max = float(old_series.max())
        new_max = float(new_series.max())

        mean_change = new_mean - old_mean
        median_change = new_median - old_median
        std_change = new_std - old_std
        min_change = new_min - old_min
        max_change = new_max - old_max

        statistical_changes.append({
            "column": column,
            "old_mean": round(old_mean, 4),
            "new_mean": round(new_mean, 4),
            "mean_change": round(mean_change, 4),
            "old_median": round(old_median, 4),
            "new_median": round(new_median, 4),
            "median_change": round(median_change, 4),
            "old_std": round(old_std, 4),
            "new_std": round(new_std, 4),
            "std_change": round(std_change, 4),
            "old_min": round(old_min, 4),
            "new_min": round(new_min, 4),
            "min_change": round(min_change, 4),
            "old_max": round(old_max, 4),
            "new_max": round(new_max, 4)
        })

    return statistical_changes