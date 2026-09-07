"""
engine/calculation_engine.py — Reliable calculation engine for the Calculation Lab.

Performs deterministic mathematical/statistical calculations using Pandas, NumPy, and SciPy.
Tracks valid vs excluded observations and provides explanations and visualization metadata.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd


CATEGORIES: dict[str, list[str]] = {
    "Basic Statistics": [
        "Count", "Sum", "Mean", "Median", "Mode", "Minimum", "Maximum", "Range", "Variance", "Standard Deviation"
    ],
    "Position / Spread": [
        "Quartiles", "Percentiles", "IQR", "Coefficient of Variation"
    ],
    "Distribution": [
        "Skewness", "Kurtosis", "Value Distribution"
    ],
    "Relationships": [
        "Pearson Correlation", "Covariance", "Grouped Statistics"
    ],
    "Categorical Analysis": [
        "Value Counts", "Category Percentages", "Most Frequent Category", "Least Frequent Category"
    ],
    "Data Quality": [
        "Missing Count", "Missing Percentage", "Duplicate Count", "Duplicate Percentage", "Unique Count", "Cardinality"
    ],
    "Dataset Comparison": [
        "Mean Change", "Median Change", "Missing-Value Change", "Row-Count Change", "Column-Count Change"
    ]
}


def perform_calculation(
    df: pd.DataFrame,
    df_prev: pd.DataFrame | None = None,
    category: str = "Basic Statistics",
    operation: str = "Mean",
    primary_col: str | None = None,
    secondary_col: str | None = None,
    extra_params: dict | None = None,
) -> dict:
    """
    Executes selected calculation deterministically.
    Returns structured result with mathematical metadata and visual chart specs.
    """
    extra_params = extra_params or {}
    total_rows = len(df) if df is not None else 0

    if df is None or df.empty:
        return {
            "error": "Dataset is empty or unavailable.",
            "category": category,
            "operation": operation,
        }

    # ---------------------------------------------------------
    # 1. BASIC STATISTICS
    # ---------------------------------------------------------
    if category == "Basic Statistics":
        if not primary_col or primary_col not in df.columns:
            return {"error": "Please select a valid numerical column."}

        series = pd.to_numeric(df[primary_col], errors="coerce")
        valid_series = series.dropna()
        valid_n = len(valid_series)
        excl_n = total_rows - valid_n

        if valid_n == 0:
            return {"error": f"Column '{primary_col}' contains no valid numerical values."}

        if operation == "Count":
            val = valid_n
            fmt = f"{val:,}"
            method = "Count of non-null valid observations"
            expl = f"There are {val:,} valid numerical observations in column '{primary_col}'."
            chart_type = "metric"

        elif operation == "Sum":
            val = float(valid_series.sum())
            fmt = f"{val:,.4f}" if isinstance(val, float) else f"{val:,}"
            method = "Summation (∑ x_i)"
            expl = f"The total sum of all {valid_n:,} valid values in '{primary_col}' is {fmt}."
            chart_type = "distribution"

        elif operation == "Mean":
            val = float(valid_series.mean())
            fmt = f"{val:,.4f}"
            method = "Arithmetic Sample Mean (∑ x_i / N)"
            expl = f"The arithmetic mean (average) of '{primary_col}' is {fmt}."
            chart_type = "mean_vs_median"

        elif operation == "Median":
            val = float(valid_series.median())
            fmt = f"{val:,.4f}"
            method = "50th Percentile / Median"
            expl = f"The median (middle value) of '{primary_col}' is {fmt}."
            chart_type = "mean_vs_median"

        elif operation == "Mode":
            mode_s = valid_series.mode()
            val = float(mode_s.iloc[0]) if not mode_s.empty else np.nan
            fmt = f"{val:,.4f}" if pd.notna(val) else "N/A"
            method = "Most frequent value (Mode)"
            expl = f"The most frequently occurring value in '{primary_col}' is {fmt}."
            chart_type = "distribution"

        elif operation == "Minimum":
            val = float(valid_series.min())
            fmt = f"{val:,.4f}"
            method = "Minimum value (Min)"
            expl = f"The smallest value recorded in '{primary_col}' is {fmt}."
            chart_type = "box"

        elif operation == "Maximum":
            val = float(valid_series.max())
            fmt = f"{val:,.4f}"
            method = "Maximum value (Max)"
            expl = f"The largest value recorded in '{primary_col}' is {fmt}."
            chart_type = "box"

        elif operation == "Range":
            val = float(valid_series.max() - valid_series.min())
            fmt = f"{val:,.4f}"
            method = "Range (Max - Min)"
            expl = f"The total numerical spread between min and max in '{primary_col}' is {fmt}."
            chart_type = "box"

        elif operation == "Variance":
            val = float(valid_series.var()) if valid_n > 1 else 0.0
            fmt = f"{val:,.4f}"
            method = "Sample Variance (s², ddof=1)"
            expl = f"The sample variance measuring squared dispersion in '{primary_col}' is {fmt}."
            chart_type = "distribution"

        elif operation == "Standard Deviation":
            val = float(valid_series.std()) if valid_n > 1 else 0.0
            fmt = f"{val:,.4f}"
            method = "Sample Standard Deviation (s, ddof=1)"
            expl = f"The average distance of observations from the mean in '{primary_col}' is ±{fmt}."
            chart_type = "distribution"

        else:
            return {"error": f"Unknown operation '{operation}'."}

        return {
            "category": category,
            "operation": operation,
            "primary_column": primary_col,
            "secondary_column": secondary_col,
            "result_value": val,
            "formatted_result": fmt,
            "observations_used": valid_n,
            "observations_excluded": excl_n,
            "total_rows": total_rows,
            "method": method,
            "explanation": expl,
            "chart_type": chart_type,
        }

    # ---------------------------------------------------------
    # 2. POSITION / SPREAD
    # ---------------------------------------------------------
    elif category == "Position / Spread":
        if not primary_col or primary_col not in df.columns:
            return {"error": "Please select a valid numerical column."}

        series = pd.to_numeric(df[primary_col], errors="coerce").dropna()
        valid_n = len(series)
        excl_n = total_rows - valid_n

        if valid_n == 0:
            return {"error": f"Column '{primary_col}' contains no valid numerical values."}

        if operation == "Quartiles":
            q1 = float(series.quantile(0.25))
            q2 = float(series.quantile(0.50))
            q3 = float(series.quantile(0.75))
            val = {"Q1 (25%)": q1, "Q2 (Median 50%)": q2, "Q3 (75%)": q3}
            fmt = f"Q1: {q1:,.2f} | Q2: {q2:,.2f} | Q3: {q3:,.2f}"
            method = "Quantiles (25th, 50th, 75th percentiles)"
            expl = f"Quartiles divide '{primary_col}' into four equal 25% quarters."
            chart_type = "box"

        elif operation == "Percentiles":
            p10 = float(series.quantile(0.10))
            p25 = float(series.quantile(0.25))
            p50 = float(series.quantile(0.50))
            p75 = float(series.quantile(0.75))
            p90 = float(series.quantile(0.90))
            val = {"P10": p10, "P25": p25, "P50": p50, "P75": p75, "P90": p90}
            fmt = f"P10: {p10:,.2f} | P50: {p50:,.2f} | P90: {p90:,.2f}"
            method = "Percentiles (P10, P25, P50, P75, P90)"
            expl = f"Percentiles show value thresholds below which specific portions of '{primary_col}' lie."
            chart_type = "box"

        elif operation == "IQR":
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            val = q3 - q1
            fmt = f"{val:,.4f}"
            method = "Interquartile Range (Q3 - Q1)"
            expl = f"The middle 50% of values in '{primary_col}' span a range of {fmt}."
            chart_type = "box"

        elif operation == "Coefficient of Variation":
            mean_val = float(series.mean())
            std_val = float(series.std()) if valid_n > 1 else 0.0
            val = (std_val / mean_val * 100) if mean_val != 0 else 0.0
            fmt = f"{val:.2f}%"
            method = "Relative Dispersion (Std / Mean * 100)"
            expl = f"The coefficient of variation for '{primary_col}' is {fmt}, measuring relative variability."
            chart_type = "distribution"

        else:
            return {"error": f"Unknown operation '{operation}'."}

        return {
            "category": category,
            "operation": operation,
            "primary_column": primary_col,
            "result_value": val,
            "formatted_result": fmt,
            "observations_used": valid_n,
            "observations_excluded": excl_n,
            "total_rows": total_rows,
            "method": method,
            "explanation": expl,
            "chart_type": chart_type,
        }

    # ---------------------------------------------------------
    # 3. DISTRIBUTION
    # ---------------------------------------------------------
    elif category == "Distribution":
        if not primary_col or primary_col not in df.columns:
            return {"error": "Please select a valid numerical column."}

        series = pd.to_numeric(df[primary_col], errors="coerce").dropna()
        valid_n = len(series)
        excl_n = total_rows - valid_n

        if valid_n == 0:
            return {"error": f"Column '{primary_col}' contains no valid numerical values."}

        if operation == "Skewness":
            val = float(series.skew()) if valid_n > 2 else 0.0
            fmt = f"{val:,.4f}"
            method = "Fisher-Pearson Skewness coefficient"
            if abs(val) < 0.5:
                shape = "fairly symmetric"
            elif abs(val) < 1.0:
                shape = "moderately skewed"
            else:
                shape = "substantially skewed"
            expl = f"The skewness coefficient is {fmt}, indicating a {shape} distribution."
            chart_type = "distribution"

        elif operation == "Kurtosis":
            val = float(series.kurtosis()) if valid_n > 3 else 0.0
            fmt = f"{val:,.4f}"
            method = "Excess Kurtosis (Normal distribution = 0)"
            shape = "leptokurtic (heavy-tailed)" if val > 0.5 else ("platykurtic (light-tailed)" if val < -0.5 else "mesokurtic (normal-like)")
            expl = f"The excess kurtosis is {fmt}, meaning the distribution is {shape}."
            chart_type = "distribution"

        elif operation == "Value Distribution":
            counts = series.value_counts(bins=10).reset_index()
            val = counts.to_dict(orient="records")
            fmt = f"10-bin histogram breakdown computed"
            method = "Frequency binning (10 equal intervals)"
            expl = f"Shows value frequency density across equal value ranges for '{primary_col}'."
            chart_type = "distribution"

        else:
            return {"error": f"Unknown operation '{operation}'."}

        return {
            "category": category,
            "operation": operation,
            "primary_column": primary_col,
            "result_value": val,
            "formatted_result": fmt,
            "observations_used": valid_n,
            "observations_excluded": excl_n,
            "total_rows": total_rows,
            "method": method,
            "explanation": expl,
            "chart_type": chart_type,
        }

    # ---------------------------------------------------------
    # 4. RELATIONSHIPS
    # ---------------------------------------------------------
    elif category == "Relationships":
        if operation in ["Pearson Correlation", "Covariance"]:
            if not primary_col or not secondary_col or primary_col not in df.columns or secondary_col not in df.columns:
                return {"error": "Please select two distinct numerical columns."}

            s1 = pd.to_numeric(df[primary_col], errors="coerce")
            s2 = pd.to_numeric(df[secondary_col], errors="coerce")
            valid_df = pd.DataFrame({primary_col: s1, secondary_col: s2}).dropna()
            valid_n = len(valid_df)
            excl_n = total_rows - valid_n

            if valid_n < 2:
                return {"error": "Not enough matching valid observations between the selected columns."}

            if operation == "Pearson Correlation":
                val = float(valid_df[primary_col].corr(valid_df[secondary_col]))
                fmt = f"{val:,.4f}"
                method = "Pearson Linear Correlation Coefficient r"
                strength = "strong" if abs(val) >= 0.7 else ("moderate" if abs(val) >= 0.4 else "weak")
                direction = "positive" if val > 0 else "negative"
                expl = f"The Pearson correlation is {fmt}, showing a {strength} {direction} linear relationship."
                chart_type = "scatter"

            elif operation == "Covariance":
                val = float(valid_df[primary_col].cov(valid_df[secondary_col]))
                fmt = f"{val:,.4f}"
                method = "Sample Covariance Cov(X, Y)"
                direction = "positive" if val > 0 else "negative"
                expl = f"The sample covariance is {fmt}, indicating a joint {direction} directional trend."
                chart_type = "scatter"

            return {
                "category": category,
                "operation": operation,
                "primary_column": primary_col,
                "secondary_column": secondary_col,
                "result_value": val,
                "formatted_result": fmt,
                "observations_used": valid_n,
                "observations_excluded": excl_n,
                "total_rows": total_rows,
                "method": method,
                "explanation": expl,
                "chart_type": chart_type,
            }

        elif operation == "Grouped Statistics":
            if not primary_col or not secondary_col or primary_col not in df.columns or secondary_col not in df.columns:
                return {"error": "Please select a categorical Group column and a numerical Measure column."}

            cat_s = df[primary_col].astype(str)
            num_s = pd.to_numeric(df[secondary_col], errors="coerce")
            valid_df = pd.DataFrame({"Group": cat_s, "Measure": num_s}).dropna()
            valid_n = len(valid_df)
            excl_n = total_rows - valid_n

            if valid_n == 0:
                return {"error": "No valid observations found for grouped statistics."}

            grp = valid_df.groupby("Group")["Measure"].agg(["count", "mean", "median", "std"]).reset_index()
            val = grp.to_dict(orient="records")
            fmt = f"{len(grp)} unique category groups computed"
            method = f"Grouped Aggregation: '{secondary_col}' by '{primary_col}'"
            expl = f"Calculated mean, median, count, and std of '{secondary_col}' broken down across '{primary_col}' groups."
            chart_type = "grouped_bar"

            return {
                "category": category,
                "operation": operation,
                "primary_column": primary_col,
                "secondary_column": secondary_col,
                "result_value": val,
                "grouped_df": grp,
                "formatted_result": fmt,
                "observations_used": valid_n,
                "observations_excluded": excl_n,
                "total_rows": total_rows,
                "method": method,
                "explanation": expl,
                "chart_type": chart_type,
            }

    # ---------------------------------------------------------
    # 5. CATEGORICAL ANALYSIS
    # ---------------------------------------------------------
    elif category == "Categorical Analysis":
        if not primary_col or primary_col not in df.columns:
            return {"error": "Please select a valid categorical column."}

        series = df[primary_col].dropna().astype(str)
        valid_n = len(series)
        excl_n = total_rows - valid_n

        if valid_n == 0:
            return {"error": f"Column '{primary_col}' contains no non-null values."}

        counts = series.value_counts()

        if operation == "Value Counts":
            val = counts.to_dict()
            top_k = list(counts.items())[:3]
            fmt = f"Top: {top_k[0][0]} ({top_k[0][1]:,})" if top_k else "N/A"
            method = "Value Frequency Counting"
            expl = f"Frequency counts of all unique values in '{primary_col}'."
            chart_type = "bar"

        elif operation == "Category Percentages":
            pcts = (counts / valid_n * 100).round(2)
            val = pcts.to_dict()
            top_k = list(pcts.items())[:3]
            fmt = f"Top: {top_k[0][0]} ({top_k[0][1]}%)" if top_k else "N/A"
            method = "Category Proportion % (Count / Valid N * 100)"
            expl = f"Relative percentage distribution of categories in '{primary_col}'."
            chart_type = "bar"

        elif operation == "Most Frequent Category":
            top_cat = counts.index[0]
            top_cnt = int(counts.iloc[0])
            top_pct = round(top_cnt / valid_n * 100, 2)
            val = {"category": top_cat, "count": top_cnt, "percentage": top_pct}
            fmt = f"'{top_cat}' ({top_cnt:,} | {top_pct}%)"
            method = "Dominant Category Identifier"
            expl = f"The most dominant category in '{primary_col}' is '{top_cat}' with {top_cnt:,} occurrences ({top_pct}%)."
            chart_type = "bar"

        elif operation == "Least Frequent Category":
            bot_cat = counts.index[-1]
            bot_cnt = int(counts.iloc[-1])
            bot_pct = round(bot_cnt / valid_n * 100, 2)
            val = {"category": bot_cat, "count": bot_cnt, "percentage": bot_pct}
            fmt = f"'{bot_cat}' ({bot_cnt:,} | {bot_pct}%)"
            method = "Rare Category Identifier"
            expl = f"The least frequent category in '{primary_col}' is '{bot_cat}' with {bot_cnt:,} occurrences ({bot_pct}%)."
            chart_type = "bar"

        else:
            return {"error": f"Unknown operation '{operation}'."}

        return {
            "category": category,
            "operation": operation,
            "primary_column": primary_col,
            "result_value": val,
            "formatted_result": fmt,
            "observations_used": valid_n,
            "observations_excluded": excl_n,
            "total_rows": total_rows,
            "method": method,
            "explanation": expl,
            "chart_type": chart_type,
        }

    # ---------------------------------------------------------
    # 6. DATA QUALITY
    # ---------------------------------------------------------
    elif category == "Data Quality":
        if not primary_col or primary_col not in df.columns:
            return {"error": "Please select a valid column."}

        series = df[primary_col]
        valid_n = total_rows

        if operation == "Missing Count":
            val = int(series.isna().sum())
            fmt = f"{val:,} missing cells"
            method = "Null / NaN count"
            expl = f"Column '{primary_col}' contains {val:,} missing (NaN) cells."
            chart_type = "metric"

        elif operation == "Missing Percentage":
            val = round(series.isna().mean() * 100, 2)
            fmt = f"{val:.2f}%"
            method = "Null Proportion (Missing / Total Rows * 100)"
            expl = f"'{primary_col}' is missing {val:.2f}% of its total records."
            chart_type = "metric"

        elif operation == "Duplicate Count":
            val = int(series.duplicated().sum())
            fmt = f"{val:,} duplicate values"
            method = "Series Duplicate Count"
            expl = f"'{primary_col}' has {val:,} non-unique duplicate entries."
            chart_type = "metric"

        elif operation == "Duplicate Percentage":
            val = round(series.duplicated().mean() * 100, 2)
            fmt = f"{val:.2f}%"
            method = "Duplicate Ratio (Duplicates / Total Rows * 100)"
            expl = f"'{primary_col}' contains {val:.2f}% repeated values."
            chart_type = "metric"

        elif operation == "Unique Count":
            val = int(series.nunique(dropna=True))
            fmt = f"{val:,} distinct unique values"
            method = "Cardinality (Distinct count)"
            expl = f"'{primary_col}' contains {val:,} distinct unique categories/values."
            chart_type = "metric"

        elif operation == "Cardinality":
            uniq = int(series.nunique(dropna=True))
            val = round(uniq / total_rows * 100, 2) if total_rows > 0 else 0.0
            fmt = f"{val:.2f}% unique ({uniq:,} distinct)"
            method = "Cardinality Ratio (Unique / Total * 100)"
            expl = f"The cardinality ratio for '{primary_col}' is {fmt}."
            chart_type = "metric"

        else:
            return {"error": f"Unknown operation '{operation}'."}

        return {
            "category": category,
            "operation": operation,
            "primary_column": primary_col,
            "result_value": val,
            "formatted_result": fmt,
            "observations_used": valid_n,
            "observations_excluded": 0,
            "total_rows": total_rows,
            "method": method,
            "explanation": expl,
            "chart_type": chart_type,
        }

    # ---------------------------------------------------------
    # 7. DATASET COMPARISON (When df_prev exists)
    # ---------------------------------------------------------
    elif category == "Dataset Comparison":
        if df_prev is None or df_prev.empty:
            return {"error": "Dataset Comparison requires both a Previous and an Updated dataset uploaded."}

        if operation in ["Mean Change", "Median Change"]:
            if not primary_col or primary_col not in df.columns or primary_col not in df_prev.columns:
                return {"error": f"Column '{primary_col}' must exist in both Previous and Updated datasets."}

            s_prev = pd.to_numeric(df_prev[primary_col], errors="coerce").dropna()
            s_curr = pd.to_numeric(df[primary_col], errors="coerce").dropna()

            if s_prev.empty or s_curr.empty:
                return {"error": f"Column '{primary_col}' has insufficient numeric values in one of the datasets."}

            if operation == "Mean Change":
                val_prev = float(s_prev.mean())
                val_curr = float(s_curr.mean())
                diff = val_curr - val_prev
                pct = (diff / val_prev * 100) if val_prev != 0 else 0.0
                fmt = f"{diff:+,.4f} ({pct:+,.2f}%)"
                method = "Mean Shift Delta (Mean_curr - Mean_prev)"
                expl = f"Mean shifted from {val_prev:,.2f} to {val_curr:,.2f} (change: {fmt})."

            elif operation == "Median Change":
                val_prev = float(s_prev.median())
                val_curr = float(s_curr.median())
                diff = val_curr - val_prev
                pct = (diff / val_prev * 100) if val_prev != 0 else 0.0
                fmt = f"{diff:+,.4f} ({pct:+,.2f}%)"
                method = "Median Shift Delta (Median_curr - Median_prev)"
                expl = f"Median shifted from {val_prev:,.2f} to {val_curr:,.2f} (change: {fmt})."

            return {
                "category": category,
                "operation": operation,
                "primary_column": primary_col,
                "result_value": diff,
                "formatted_result": fmt,
                "observations_used": len(s_curr),
                "observations_excluded": len(df) - len(s_curr),
                "total_rows": total_rows,
                "method": method,
                "explanation": expl,
                "chart_type": "comparison",
                "val_prev": val_prev,
                "val_curr": val_curr,
            }

        elif operation == "Missing-Value Change":
            if not primary_col or primary_col not in df.columns or primary_col not in df_prev.columns:
                return {"error": f"Column '{primary_col}' must exist in both datasets."}

            m_prev = int(df_prev[primary_col].isna().sum())
            m_curr = int(df[primary_col].isna().sum())
            diff = m_curr - m_prev
            fmt = f"{diff:+,d} missing cells"
            method = "Missingness Delta (Missing_curr - Missing_prev)"
            expl = f"Missing cell count changed from {m_prev:,} to {m_curr:,} ({fmt})."

            return {
                "category": category,
                "operation": operation,
                "primary_column": primary_col,
                "result_value": diff,
                "formatted_result": fmt,
                "observations_used": total_rows,
                "observations_excluded": 0,
                "total_rows": total_rows,
                "method": method,
                "explanation": expl,
                "chart_type": "comparison",
                "val_prev": m_prev,
                "val_curr": m_curr,
            }

        elif operation in ["Row-Count Change", "Column-Count Change"]:
            if operation == "Row-Count Change":
                val_prev = len(df_prev)
                val_curr = len(df)
                diff = val_curr - val_prev
                fmt = f"{diff:+,d} rows"
                method = "Row Delta (Rows_curr - Rows_prev)"
                expl = f"Total dataset rows changed from {val_prev:,} to {val_curr:,} ({fmt})."

            elif operation == "Column-Count Change":
                val_prev = len(df_prev.columns)
                val_curr = len(df.columns)
                diff = val_curr - val_prev
                fmt = f"{diff:+,d} columns"
                method = "Column Delta (Cols_curr - Cols_prev)"
                expl = f"Total dataset columns changed from {val_prev:,} to {val_curr:,} ({fmt})."

            return {
                "category": category,
                "operation": operation,
                "result_value": diff,
                "formatted_result": fmt,
                "observations_used": total_rows,
                "observations_excluded": 0,
                "total_rows": total_rows,
                "method": method,
                "explanation": expl,
                "chart_type": "comparison",
                "val_prev": val_prev,
                "val_curr": val_curr,
            }

    return {"error": "Unsupported calculation selection."}
