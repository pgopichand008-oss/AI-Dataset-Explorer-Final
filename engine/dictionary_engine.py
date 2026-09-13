"""
General-purpose AI Data Dictionary Engine.

Responsibilities
----------------
- Infer column data types from values rather than relying only on pandas dtype.
- Infer semantic roles such as identifier, categorical feature, numerical measure,
  datetime, text, and target candidate.
- Detect common missing-value placeholders.
- Detect mixed/invalid values without modifying the original dataframe.
- Estimate confidence in every inference.
- Provide analytical and ML usefulness guidance.
- Return JSON-serializable output.

Design principles
-----------------
1. Never mutate the input dataframe.
2. Never hard-code one dataset's column names.
3. Prefer evidence from actual values over column-name guesses.
4. Use conservative semantic inference when evidence is weak.
5. Preserve compatibility with build_data_dictionary(df).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

MISSING_LIKE_VALUES = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "nil",
    "missing",
    "unknown",
    "not available",
    "not applicable",
    "not_applicable",
    "-",
    "--",
    "?",
}


def _to_serializable(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def _normalize_column_name(name: Any) -> str:
    """
    Normalize a column name for semantic analysis.

    This is intentionally generic. It does not identify one particular
    dataset; it recognizes broad concepts such as id, name, phone, etc.
    """

    text = str(name).strip().lower()

    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s$%]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _is_missing_like(value: Any) -> bool:
    """Detect explicit missing values and common textual placeholders."""

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip().lower()

    return text in MISSING_LIKE_VALUES


def _get_non_missing_values(series: pd.Series) -> pd.Series:
    """Return values that are not missing or missing-like."""

    if series.empty:
        return series

    mask = ~series.map(_is_missing_like)
    return series.loc[mask]


# ---------------------------------------------------------------------
# Value-pattern analysis
# ---------------------------------------------------------------------

def _numeric_conversion_ratio(series: pd.Series) -> float:
    """Percentage of non-missing values that can reasonably become numeric."""

    values = _get_non_missing_values(series)

    if values.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(values):
        return 1.0

    converted = pd.to_numeric(
        values.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )

    return float(converted.notna().mean())


def _datetime_conversion_ratio(series: pd.Series) -> float:
    """
    Estimate whether values represent dates.

    Uses conservative parsing to avoid interpreting arbitrary numbers or
    short categorical strings as dates.
    """

    values = _get_non_missing_values(series)

    if values.empty:
        return 0.0

    if pd.api.types.is_datetime64_any_dtype(values):
        return 1.0

    # Existing numeric columns should not automatically become dates.
    if pd.api.types.is_numeric_dtype(values):
        return 0.0

    text_values = values.astype(str).str.strip()

    # Require some visible date/time structure.
    date_pattern = text_values.str.contains(
        r"[-/:]|[A-Za-z]{3,}",
        regex=True,
        na=False,
    )

    if date_pattern.mean() < 0.50:
        return 0.0

    parsed = pd.to_datetime(
        text_values,
        errors="coerce",
        format="mixed",
    )

    return float(parsed.notna().mean())


def _boolean_conversion_ratio(series: pd.Series) -> float:
    """Detect common boolean representations."""

    values = _get_non_missing_values(series)

    if values.empty:
        return 0.0

    if pd.api.types.is_bool_dtype(values):
        return 1.0

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    boolean_values = {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
        "t",
        "f",
        "1",
        "0",
    }

    return float(normalized.isin(boolean_values).mean())


def _identifier_pattern_ratio(series: pd.Series) -> float:
    """
    Detect identifier-like value patterns.

    Examples of generic patterns:
    - EMP001
    - ABC-12345
    - UUID-like values
    - account/reference numbers
    - long digit strings
    """

    values = _get_non_missing_values(series)

    if values.empty:
        return 0.0

    text_values = values.astype(str).str.strip()

    patterns = [
        # UUID-like
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",

        # Prefix + number, e.g. ABC123 / EMP-1001
        r"^[A-Za-z]{2,8}[-_]?\d{2,}$",

        # Long numeric identifiers
        r"^\d{6,}$",

        # Alphanumeric reference codes
        r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_-]{6,}$",
    ]

    matched = pd.Series(False, index=text_values.index)

    for pattern in patterns:
        matched |= text_values.str.match(
            pattern,
            case=False,
            na=False,
        )

    return float(matched.mean())


def _text_characteristics(series: pd.Series) -> Dict[str, Any]:
    """Calculate generic text characteristics."""

    values = _get_non_missing_values(series)

    if values.empty:
        return {
            "avg_length": 0.0,
            "max_length": 0,
            "space_ratio": 0.0,
            "alphabetic_ratio": 0.0,
            "email_ratio": 0.0,
            "phone_ratio": 0.0,
        }

    text = values.astype(str).str.strip()

    lengths = text.str.len()

    avg_length = float(lengths.mean())
    max_length = int(lengths.max())

    space_ratio = float(text.str.contains(r"\s", regex=True).mean())

    alphabetic_ratio = float(
        text.str.contains(r"[A-Za-z]", regex=True).mean()
    )

    email_ratio = float(
        text.str.match(
            r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            na=False,
        ).mean()
    )

    phone_ratio = float(
        text.str.match(
            r"^\+?[\d\s().-]{7,}$",
            na=False,
        ).mean()
    )

    return {
        "avg_length": avg_length,
        "max_length": max_length,
        "space_ratio": space_ratio,
        "alphabetic_ratio": alphabetic_ratio,
        "email_ratio": email_ratio,
        "phone_ratio": phone_ratio,
    }


# ---------------------------------------------------------------------
# Semantic column-name signals
# ---------------------------------------------------------------------

def _name_signal(column_name: str, groups: Dict[str, List[str]]) -> float:
    """
    Calculate a generic name-based signal.

    Exact token/phrase matches receive stronger evidence than loose
    substring matches.
    """

    normalized = _normalize_column_name(column_name)
    tokens = set(normalized.split())

    best_score = 0.0

    for terms in groups.values():
        for term in terms:
            term_norm = _normalize_column_name(term)

            if not term_norm:
                continue

            term_tokens = set(term_norm.split())

            if term_norm == normalized:
                best_score = max(best_score, 1.0)

            elif term_tokens and term_tokens.issubset(tokens):
                best_score = max(best_score, 0.95)

            elif term_norm in normalized:
                best_score = max(best_score, 0.75)

    return best_score


IDENTIFIER_NAME_GROUPS = {
    "identifier": [
        "id",
        "identifier",
        "reference",
        "ref",
        "code",
        "key",
        "record number",
        "record no",
        "number",
        "account number",
        "account no",
        "customer number",
        "customer no",
        "employee number",
        "employee no",
        "user id",
        "user number",
        "transaction id",
        "transaction number",
    ]
}


NAME_FIELD_GROUPS = {
    "name": [
        "name",
        "first name",
        "last name",
        "middle name",
        "full name",
        "given name",
        "surname",
        "family name",
        "display name",
    ]
}


CONTACT_GROUPS = {
    "email": [
        "email",
        "email address",
        "mail",
        "e mail",
    ],
    "phone": [
        "phone",
        "phone number",
        "mobile",
        "mobile number",
        "telephone",
        "telephone number",
        "contact number",
    ],
}


CATEGORY_NAME_GROUPS = {
    "category": [
        "category",
        "type",
        "class",
        "classification",
        "group",
        "department",
        "division",
        "team",
        "region",
        "location",
        "country",
        "state",
        "city",
        "status",
        "gender",
        "role",
        "level",
        "segment",
        "category",
    ]
}


TARGET_NAME_GROUPS = {
    "target": [
        "target",
        "label",
        "outcome",
        "response",
        "prediction",
        "predicted",
        "score",
        "result",
        "dependent variable",
    ]
}


# ---------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------

def _detect_column_type(
    series: pd.Series,
    column_name: str,
) -> Dict[str, Any]:
    """
    Infer the most appropriate data type.

    Returns:
        {
            "type": ...,
            "confidence": ...,
            "evidence": {...}
        }
    """

    values = _get_non_missing_values(series)

    total_count = len(series)
    non_missing_count = len(values)

    if non_missing_count == 0:
        return {
            "type": "all missing",
            "confidence": 100,
            "evidence": {
                "non_missing_count": 0,
                "missing_count": total_count,
            },
        }

    missing_count = total_count - non_missing_count

    numeric_ratio = _numeric_conversion_ratio(series)
    datetime_ratio = _datetime_conversion_ratio(series)
    boolean_ratio = _boolean_conversion_ratio(series)
    identifier_ratio = _identifier_pattern_ratio(series)
    text_info = _text_characteristics(series)

    unique_count = int(values.astype(str).nunique())
    cardinality_ratio = unique_count / max(non_missing_count, 1)

    normalized_name = _normalize_column_name(column_name)

    identifier_name_score = _name_signal(
        normalized_name,
        IDENTIFIER_NAME_GROUPS,
    )

    name_field_score = _name_signal(
        normalized_name,
        NAME_FIELD_GROUPS,
    )

    contact_score = _name_signal(
        normalized_name,
        CONTACT_GROUPS,
    )

    category_name_score = _name_signal(
        normalized_name,
        CATEGORY_NAME_GROUPS,
    )

    # -------------------------------------------------------------
    # Existing pandas dtypes
    # -------------------------------------------------------------

    if pd.api.types.is_bool_dtype(series):
        return {
            "type": "boolean",
            "confidence": 98,
            "evidence": {
                "boolean_ratio": 1.0,
                "pandas_dtype": str(series.dtype),
            },
        }

    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "type": "datetime",
            "confidence": 99,
            "evidence": {
                "datetime_ratio": 1.0,
                "pandas_dtype": str(series.dtype),
            },
        }

    if pd.api.types.is_numeric_dtype(series):
        return {
            "type": "numerical",
            "confidence": 96,
            "evidence": {
                "numeric_ratio": 1.0,
                "pandas_dtype": str(series.dtype),
                "unique_count": unique_count,
            },
        }

    # -------------------------------------------------------------
    # Strong content-based semantic patterns
    # -------------------------------------------------------------

    if text_info["email_ratio"] >= 0.80:
        return {
            "type": "text",
            "confidence": 96,
            "evidence": {
                "email_ratio": text_info["email_ratio"],
                "contact_signal": contact_score,
            },
        }

    if (
        text_info["phone_ratio"] >= 0.80
        and contact_score >= 0.50
    ):
        return {
            "type": "identifier",
            "confidence": 94,
            "evidence": {
                "phone_ratio": text_info["phone_ratio"],
                "contact_signal": contact_score,
            },
        }

    # -------------------------------------------------------------
    # Identifier detection
    # -------------------------------------------------------------

    identifier_score = max(
        identifier_name_score,
        identifier_ratio,
    )

    if identifier_score >= 0.90 and cardinality_ratio >= 0.70:
        return {
            "type": "identifier",
            "confidence": 95,
            "evidence": {
                "identifier_name_score": identifier_name_score,
                "identifier_pattern_ratio": identifier_ratio,
                "uniqueness_ratio": cardinality_ratio,
            },
        }

    # High uniqueness + identifier pattern can also indicate IDs.
    if (
        identifier_ratio >= 0.80
        and cardinality_ratio >= 0.90
    ):
        return {
            "type": "identifier",
            "confidence": 92,
            "evidence": {
                "identifier_pattern_ratio": identifier_ratio,
                "uniqueness_ratio": cardinality_ratio,
            },
        }

    # -------------------------------------------------------------
    # Datetime detection
    # -------------------------------------------------------------

    if datetime_ratio >= 0.85:
        return {
            "type": "datetime",
            "confidence": 96,
            "evidence": {
                "datetime_ratio": datetime_ratio,
            },
        }

    # -------------------------------------------------------------
    # Boolean detection
    # -------------------------------------------------------------

    if boolean_ratio >= 0.90:
        return {
            "type": "boolean",
            "confidence": 95,
            "evidence": {
                "boolean_ratio": boolean_ratio,
            },
        }

    # -------------------------------------------------------------
    # Numeric-like columns with a few invalid values
    # -------------------------------------------------------------

    if numeric_ratio >= 0.80:
        confidence = 96 if numeric_ratio >= 0.95 else 90

        return {
            "type": "numerical",
            "confidence": confidence,
            "evidence": {
                "numeric_conversion_ratio": numeric_ratio,
                "invalid_ratio": 1.0 - numeric_ratio,
            },
        }

    # -------------------------------------------------------------
    # Names and human-readable fields
    # -------------------------------------------------------------

    if name_field_score >= 0.75:
        return {
            "type": "text",
            "confidence": 94,
            "evidence": {
                "name_field_score": name_field_score,
                "average_length": text_info["avg_length"],
                "space_ratio": text_info["space_ratio"],
            },
        }

    # Text columns with mostly alphabetic values and moderate
    # cardinality are usually names, descriptions, labels, etc.
    if (
        text_info["alphabetic_ratio"] >= 0.80
        and text_info["avg_length"] >= 2
        and text_info["avg_length"] <= 80
    ):
        # Low cardinality is more likely categorical.
        if cardinality_ratio <= 0.35 and unique_count <= 30:
            return {
                "type": "categorical",
                "confidence": 90,
                "evidence": {
                    "alphabetic_ratio": text_info["alphabetic_ratio"],
                    "uniqueness_ratio": cardinality_ratio,
                    "unique_count": unique_count,
                },
            }

        return {
            "type": "text",
            "confidence": 88,
            "evidence": {
                "alphabetic_ratio": text_info["alphabetic_ratio"],
                "average_length": text_info["avg_length"],
                "uniqueness_ratio": cardinality_ratio,
            },
        }

    # -------------------------------------------------------------
    # Explicit category signals
    # -------------------------------------------------------------

    if category_name_score >= 0.70:
        return {
            "type": "categorical",
            "confidence": 91,
            "evidence": {
                "category_name_score": category_name_score,
                "unique_count": unique_count,
                "uniqueness_ratio": cardinality_ratio,
            },
        }

    # -------------------------------------------------------------
    # Low-cardinality fallback
    # -------------------------------------------------------------

    if unique_count <= 20 and cardinality_ratio <= 0.50:
        return {
            "type": "categorical",
            "confidence": 84,
            "evidence": {
                "unique_count": unique_count,
                "uniqueness_ratio": cardinality_ratio,
            },
        }

    # -------------------------------------------------------------
    # Long / variable text fallback
    # -------------------------------------------------------------

    if (
        text_info["avg_length"] >= 50
        or (
            text_info["avg_length"] >= 25
            and text_info["space_ratio"] >= 0.30
        )
    ):
        return {
            "type": "text",
            "confidence": 85,
            "evidence": {
                "average_length": text_info["avg_length"],
                "maximum_length": text_info["max_length"],
                "space_ratio": text_info["space_ratio"],
            },
        }

    # -------------------------------------------------------------
    # Generic object/string fallback
    # -------------------------------------------------------------

    if pd.api.types.is_string_dtype(series):
        return {
            "type": "text",
            "confidence": 75,
            "evidence": {
                "pandas_dtype": str(series.dtype),
                "unique_count": unique_count,
                "uniqueness_ratio": cardinality_ratio,
            },
        }

    return {
        "type": "unknown",
        "confidence": 50,
        "evidence": {
            "pandas_dtype": str(series.dtype),
            "unique_count": unique_count,
            "uniqueness_ratio": cardinality_ratio,
        },
    }


# ---------------------------------------------------------------------
# Semantic role detection
# ---------------------------------------------------------------------

def _determine_semantic_role(
    series: pd.Series,
    column_name: str,
    detected_type: str,
) -> str:
    """Determine the most useful analytical role."""

    normalized_name = _normalize_column_name(column_name)

    values = _get_non_missing_values(series)

    if values.empty:
        return "unknown"

    unique_count = int(values.astype(str).nunique())
    non_missing_count = len(values)

    uniqueness_ratio = unique_count / max(non_missing_count, 1)

    identifier_name_score = _name_signal(
        normalized_name,
        IDENTIFIER_NAME_GROUPS,
    )

    name_field_score = _name_signal(
        normalized_name,
        NAME_FIELD_GROUPS,
    )

    category_name_score = _name_signal(
        normalized_name,
        CATEGORY_NAME_GROUPS,
    )

    target_name_score = _name_signal(
        normalized_name,
        TARGET_NAME_GROUPS,
    )

    text_info = _text_characteristics(series)

    # -------------------------------------------------------------
    # Identifier
    # -------------------------------------------------------------

    if detected_type == "identifier":
        return "identifier"

    if (
        identifier_name_score >= 0.90
        and uniqueness_ratio >= 0.70
    ):
        return "identifier"

    # -------------------------------------------------------------
    # Name/contact fields are descriptive text, not identifiers
    # -------------------------------------------------------------

    if name_field_score >= 0.75:
        return "text"

    if (
        text_info["email_ratio"] >= 0.80
        or text_info["phone_ratio"] >= 0.80
    ):
        return "contact information"

    # -------------------------------------------------------------
    # Target candidate
    # -------------------------------------------------------------

    if target_name_score >= 0.85:
        return "target candidate"

    # -------------------------------------------------------------
    # Datetime
    # -------------------------------------------------------------

    if detected_type == "datetime":
        return "datetime"

    # -------------------------------------------------------------
    # Numerical
    # -------------------------------------------------------------

    if detected_type == "numerical":
        return "numerical measure"

    # -------------------------------------------------------------
    # Boolean
    # -------------------------------------------------------------

    if detected_type == "boolean":
        return "categorical feature"

    # -------------------------------------------------------------
    # Categorical
    # -------------------------------------------------------------

    if detected_type == "categorical":
        return "categorical feature"

    # -------------------------------------------------------------
    # Text
    # -------------------------------------------------------------

    if detected_type == "text":
        return "text"

    # -------------------------------------------------------------
    # Name-based category signal can rescue ambiguous object fields
    # -------------------------------------------------------------

    if category_name_score >= 0.70:
        return "categorical feature"

    return "unknown"


# ---------------------------------------------------------------------
# Possible meaning
# ---------------------------------------------------------------------

def _generate_possible_meaning(
    column_name: str,
    detected_type: str,
    semantic_role: str,
    series: pd.Series,
) -> str:
    """Generate a concise, evidence-based interpretation."""

    values = _get_non_missing_values(series)

    if values.empty:
        return "Column contains no usable non-missing values."

    unique_count = int(values.astype(str).nunique())

    if semantic_role == "identifier":
        return (
            "Possible interpretation: identifier-like field based on "
            "uniqueness and/or identifier patterns."
        )

    if semantic_role == "datetime":
        return (
            "Possible interpretation: date or time field supported "
            "by date-like observations."
        )

    if semantic_role == "numerical measure":
        return (
            "Possible interpretation: quantitative field containing "
            "numerical observations."
        )

    if semantic_role == "target candidate":
        return (
            "Possible interpretation: potential prediction target "
            "based on target-oriented naming or evidence."
        )

    if semantic_role == "categorical feature":
        return (
            f"Possible interpretation: categorical field with "
            f"{unique_count} observed level(s)."
        )

    if semantic_role == "contact information":
        return (
            "Possible interpretation: contact-information field "
            "such as email or telephone data."
        )

    if semantic_role == "text":
        return (
            "Possible interpretation: free-form or variable-length "
            "textual field."
        )

    if detected_type == "boolean":
        return (
            "Possible interpretation: binary field representing "
            "two logical states."
        )

    return (
        "Possible interpretation: column type or role could not be "
        "determined confidently from available evidence."
    )


# ---------------------------------------------------------------------
# Quality concerns
# ---------------------------------------------------------------------

def _collect_quality_concerns(
    series: pd.Series,
    detected_type: str,
) -> List[str]:
    """Return generalized data-quality observations."""

    concerns: List[str] = []

    total = len(series)

    if total == 0:
        return concerns

    missing_mask = series.map(_is_missing_like)
    missing_count = int(missing_mask.sum())

    if missing_count > 0:
        percentage = (missing_count / total) * 100

        concerns.append(
            f"Contains {missing_count} missing or missing-like value(s) "
            f"({percentage:.1f}%)."
        )

    if detected_type == "numerical":
        numeric_ratio = _numeric_conversion_ratio(series)

        if numeric_ratio < 1.0:
            values = _get_non_missing_values(series)

            non_numeric = pd.to_numeric(
                values.astype(str)
                .str.replace(",", "", regex=False)
                .str.strip(),
                errors="coerce",
            ).isna()

            examples = (
                values.loc[non_numeric]
                .astype(str)
                .head(5)
                .tolist()
            )

            invalid_count = int(non_numeric.sum())

            concerns.append(
                f"Contains {invalid_count} invalid or mixed value(s). "
                f"Examples: {examples}."
            )

    if detected_type == "identifier":
        values = _get_non_missing_values(series)

        if not values.empty:
            uniqueness = (
                values.astype(str).nunique()
                / len(values)
            )

            if uniqueness >= 0.95:
                concerns.append(
                    f"Identifier-like field with "
                    f"{uniqueness * 100:.1f}% unique values; may cause "
                    "leakage or memorization if used directly in ML."
                )

    return concerns


# ---------------------------------------------------------------------
# Analytical usefulness
# ---------------------------------------------------------------------

def _determine_analytical_usefulness(
    detected_type: str,
    semantic_role: str,
) -> str:

    if semantic_role == "identifier":
        return (
            "Useful for record identification and joins, but usually "
            "has limited direct analytical value."
        )

    if semantic_role == "numerical measure":
        return (
            "Useful for descriptive statistics, distributions, "
            "comparisons, correlations and quantitative analysis."
        )

    if semantic_role == "datetime":
        return (
            "Useful for temporal trends, ordering, time intervals, "
            "seasonality and aggregation."
        )

    if semantic_role == "categorical feature":
        return (
            "Useful for grouping, segmentation, frequency analysis "
            "and categorical comparisons."
        )

    if semantic_role == "contact information":
        return (
            "Useful for contact-data profiling and quality checks, "
            "but usually not as a direct analytical variable."
        )

    if semantic_role == "text":
        return (
            "Useful for text profiling, keyword analysis, search "
            "and natural-language processing."
        )

    if semantic_role == "target candidate":
        return (
            "Useful as a potential prediction target after validating "
            "the intended modeling objective."
        )

    return (
        "Limited analytical usefulness until the column type "
        "is validated."
    )


# ---------------------------------------------------------------------
# ML usefulness
# ---------------------------------------------------------------------

def _determine_ml_usefulness(
    detected_type: str,
    semantic_role: str,
) -> str:

    if semantic_role == "identifier":
        return (
            "Use caution; identifier-like fields can cause data "
            "leakage or memorization."
        )

    if semantic_role == "numerical measure":
        return (
            "Potential numerical ML feature after appropriate "
            "validation and preprocessing."
        )

    if semantic_role == "categorical feature":
        return (
            "Potential categorical ML feature after appropriate encoding."
        )

    if semantic_role == "datetime":
        return (
            "Can be transformed into temporal features such as "
            "year, month, day, duration or elapsed time."
        )

    if semantic_role == "text":
        return (
            "Requires text preprocessing such as vectorization "
            "or embeddings before ML use."
        )

    if semantic_role == "contact information":
        return (
            "Usually requires feature extraction and privacy-aware "
            "preprocessing before ML use."
        )

    if semantic_role == "target candidate":
        return (
            "Potential target variable based on available schema "
            "and target evidence."
        )

    return (
        "Validate type and data quality before using the column "
        "in machine learning."
    )


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def _extract_column_statistics(
    series: pd.Series,
    detected_type: str,
) -> Dict[str, Any]:
    """Extract safe, useful statistics without changing the data."""

    values = _get_non_missing_values(series)

    stats: Dict[str, Any] = {
        "count": int(len(series)),
        "non_missing": int(len(values)),
        "missing": int(len(series) - len(values)),
        "unique": int(values.astype(str).nunique())
        if not values.empty
        else 0,
    }

    if values.empty:
        return stats

    if detected_type == "numerical":
        numeric = pd.to_numeric(
            values.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).dropna()

        if not numeric.empty:
            stats.update(
                {
                    "mean": _to_serializable(numeric.mean()),
                    "median": _to_serializable(numeric.median()),
                    "min": _to_serializable(numeric.min()),
                    "max": _to_serializable(numeric.max()),
                    "std": _to_serializable(numeric.std()),
                }
            )

    elif detected_type == "categorical":
        counts = (
            values.astype(str)
            .value_counts()
            .head(10)
            .to_dict()
        )

        stats["top_values"] = {
            str(k): int(v)
            for k, v in counts.items()
        }

    elif detected_type == "text":
        text = values.astype(str)

        stats.update(
            {
                "average_length": _to_serializable(
                    text.str.len().mean()
                ),
                "maximum_length": int(text.str.len().max()),
            }
        )

    elif detected_type == "datetime":
        parsed = pd.to_datetime(
            values.astype(str),
            errors="coerce",
            format="mixed",
        ).dropna()

        if not parsed.empty:
            stats.update(
                {
                    "earliest": _to_serializable(parsed.min()),
                    "latest": _to_serializable(parsed.max()),
                }
            )

    return stats


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------

def build_data_dictionary(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build a generalized AI Data Dictionary.

    Parameters
    ----------
    df:
        Input dataframe.

    Returns
    -------
    dict
        JSON-serializable dictionary describing every column.

    The input dataframe is never modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_data_dictionary expects a pandas DataFrame.")

    result: Dict[str, Any] = {
        "dataset_summary": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "columns": [],
    }

    for column in df.columns:

        series = df[column]

        detection = _detect_column_type(
            series,
            str(column),
        )

        detected_type = detection["type"]
        confidence = int(round(detection["confidence"]))

        semantic_role = _determine_semantic_role(
            series,
            str(column),
            detected_type,
        )

        meaning = _generate_possible_meaning(
            str(column),
            detected_type,
            semantic_role,
            series,
        )

        quality_concerns = _collect_quality_concerns(
            series,
            detected_type,
        )

        analytical_usefulness = _determine_analytical_usefulness(
            detected_type,
            semantic_role,
        )

        ml_usefulness = _determine_ml_usefulness(
            detected_type,
            semantic_role,
        )

        statistics = _extract_column_statistics(
            series,
            detected_type,
        )

        result["columns"].append(
            {
                "column": str(column),
                "name": str(column),
                "column_name": str(column),
                "type": detected_type,
                "detected_type": detected_type,
                "data_type": detected_type,
                "semantic_role": semantic_role,
                "analytical_role": semantic_role,
                "confidence": confidence,
                "possible_meaning": meaning,
                "quality_concerns": quality_concerns,
                "analytical_usefulness": analytical_usefulness,
                "ml_usefulness": ml_usefulness,
                "statistics": statistics,
            }
        )

    return result


__all__ = [
    "build_data_dictionary",
]