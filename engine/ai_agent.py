"""
engine/ai_agent.py — Adaptive dataset change intelligence agent.
Implements the 6-stage Mystery Mission workflow:
DETECT -> ANALYZE -> COMPARE -> RECONSIDER -> RE-PLAN -> REPORT
"""

from __future__ import annotations


def reconsider_previous_finding(finding: str, changes: dict, quality: dict) -> dict:
    """
    Check whether a previous analysis finding is still valid
    after the dataset has been updated.
    """
    if not finding:
        return {
            "status": "NO_PREVIOUS_FINDING",
            "warnings": [],
            "message": "No previous finding was provided for reconsideration."
        }

    finding_text = finding.lower()
    warnings = []

    # Removed columns
    for column in changes.get("removed_columns", []):
        if column.lower() in finding_text:
            warnings.append(
                f"'{column}' was removed from the updated dataset."
            )

    # Data type changes
    for change in changes.get("type_changes", []):
        column = change["column"]
        if column.lower() in finding_text:
            warnings.append(
                f"'{column}' changed type from {change['old_type']} to {change['new_type']}."
            )

    # Increased missing values
    for change in quality.get("missing_changes", []):
        column = change["column"]
        if column.lower() in finding_text and change["change"] > 0:
            warnings.append(
                f"Missing values in '{column}' increased by {change['change']} percentage points."
            )

    if warnings:
        return {
            "status": "RECONSIDER",
            "warnings": warnings,
            "message": "⚠ Previous finding is affected by dataset changes and requires reconsideration."
        }

    return {
        "status": "STILL_VALID",
        "warnings": [],
        "message": "✓ Previous finding remains valid under updated dataset conditions."
    }


def assess_impact(changes: dict, quality: dict, statistics: list | None = None) -> list[dict]:
    """
    Assess how significant the dataset changes are.
    """
    impacts = []

    # Added columns
    if changes.get("added_columns"):
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "MEDIUM",
            "message": f"{len(changes['added_columns'])} column(s) were added."
        })

    # Removed columns
    if changes.get("removed_columns"):
        impacts.append({
            "area": "STRUCTURAL",
            "severity": "HIGH",
            "message": f"{len(changes['removed_columns'])} column(s) were removed."
        })

    # Data type changes
    if changes.get("type_changes"):
        high_conflicts = [
            change for change in changes["type_changes"]
            if change.get("conflict_level") == "HIGH"
        ]
        low_changes = [
            change for change in changes["type_changes"]
            if change.get("conflict_level") == "LOW"
        ]

        if high_conflicts:
            impacts.append({
                "area": "DATA TYPE",
                "severity": "HIGH",
                "message": f"{len(high_conflicts)} high-severity data type conflict(s) detected."
            })
        if low_changes:
            impacts.append({
                "area": "DATA TYPE",
                "severity": "LOW",
                "message": f"{len(low_changes)} low-severity data type change(s) detected."
            })

    # Invalid values
    invalid_columns = [
        col for col, summary in changes.get("invalid_values", {}).items()
        if summary.get("invalid_count", 0) > 0
    ]

    if invalid_columns:
        impacts.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": f"Invalid values detected in {len(invalid_columns)} column(s)."
        })

    # Missing values
    increased_missing = [
        item for item in quality.get("missing_changes", [])
        if item["change"] > 0
    ]

    if increased_missing:
        impacts.append({
            "area": "DATA QUALITY",
            "severity": "MEDIUM",
            "message": f"Missing values increased in {len(increased_missing)} column(s)."
        })

    # Duplicate rows
    if quality.get("duplicate_change", 0) > 0:
        impacts.append({
            "area": "DUPLICATES",
            "severity": "MEDIUM",
            "message": "Duplicate rows increased in the updated dataset."
        })

    # Statistical changes
    if statistics:
        for stat in statistics:
            column_name = stat["column"].lower()
            if column_name == "id" or column_name.endswith("_id") or column_name.endswith("id"):
                continue
            if stat["old_mean"] == 0:
                continue

            mean_change_percent = abs(
                stat["mean_change"] / stat["old_mean"] * 100
            )

            if mean_change_percent >= 20:
                impacts.append({
                    "area": "STATISTICAL",
                    "severity": "HIGH",
                    "message": f"'{stat['column']}' mean changed by {round(mean_change_percent, 1)}%."
                })
            elif mean_change_percent >= 10:
                impacts.append({
                    "area": "STATISTICAL",
                    "severity": "MEDIUM",
                    "message": f"'{stat['column']}' mean changed by {round(mean_change_percent, 1)}%."
                })

    return impacts


def assess_ml_readiness(changes: dict, quality: dict, statistics: list | None = None) -> dict:
    """
    Assess how dataset changes affect machine-learning readiness.
    """
    issues = []

    if changes.get("removed_columns"):
        issues.append("Removed columns may affect previously selected ML features.")

    for change in changes.get("type_changes", []):
        if change.get("conflict_level") == "HIGH":
            issues.append(
                f"High-severity data-type conflict detected in '{change['column']}'; preprocessing required."
            )
        elif change.get("conflict_level") == "LOW":
            issues.append(
                f"Low-severity data-type change detected in '{change['column']}'; preprocessing suggested."
            )

    for column, summary in changes.get("invalid_values", {}).items():
        if summary.get("invalid_count", 0) > 0:
            issues.append(f"Invalid values detected in '{column}'.")

    for change in quality.get("missing_changes", []):
        if change["change"] >= 10:
            issues.append(f"Missing values increased in '{change['column']}'.")

    if statistics:
        for stat in statistics:
            column_name = stat["column"].lower()
            if column_name == "id" or column_name.endswith("_id") or column_name.endswith("id"):
                continue
            if stat["old_mean"] == 0:
                continue
            change_percent = abs(stat["mean_change"] / stat["old_mean"] * 100)
            if change_percent >= 20:
                issues.append(f"'{stat['column']}' shows a significant statistical shift ({round(change_percent, 1)}%).")

    if not issues:
        return {
            "status": "READY FOR FURTHER ANALYSIS",
            "issues": []
        }

    high_severity_present = any(
        "High-severity" in issue or "Removed columns" in issue for issue in issues
    )

    if high_severity_present:
        return {
            "status": "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION",
            "issues": issues
        }

    return {
        "status": "NEEDS ATTENTION BEFORE ANALYSIS",
        "issues": issues
    }


def identify_limitations(changes: dict, quality: dict, statistics: list | None = None) -> list[str]:
    """
    Identify limitations to communicate before analysis.
    """
    limitations = []

    if changes.get("possible_renames"):
        limitations.append(
            "Some column changes may represent renaming; rename detection is probabilistic."
        )
    if changes.get("removed_columns"):
        limitations.append("Previous analyses involving removed columns may no longer apply.")
    if changes.get("type_changes"):
        limitations.append("Changed data types may affect statistical and machine-learning modeling.")

    for column, summary in changes.get("invalid_values", {}).items():
        if summary.get("invalid_count", 0) > 0:
            limitations.append(
                f"Invalid values detected in '{column}'; valid observations filtered where appropriate."
            )

    for change in quality.get("missing_changes", []):
        if change["change"] > 0:
            limitations.append(f"Missing values changed in '{change['column']}'.")

    if statistics:
        for stat in statistics:
            column_name = stat["column"].lower()
            if column_name == "id" or column_name.endswith("_id") or column_name.endswith("id"):
                continue
            if stat["old_mean"] == 0:
                continue
            change_percent = abs(stat["mean_change"] / stat["old_mean"] * 100)
            if change_percent >= 10:
                limitations.append(f"'{stat['column']}' shows a notable statistical shift.")

    return limitations


def generate_recommendation(changes: dict, quality: dict) -> str:
    """
    Generate exactly one final deterministic recommendation.
    """
    for type_change in changes.get("type_changes", []):
        if type_change.get("conflict_level") == "HIGH":
            return "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"

    for column, summary in changes.get("invalid_values", {}).items():
        invalid_count = summary.get("invalid_count", 0)
        valid_count = summary.get("valid_count", 0)

        if invalid_count > 0 and valid_count > 0:
            return "NEEDS ATTENTION BEFORE ANALYSIS"
        if invalid_count > 0 and valid_count == 0:
            return "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION"

    possible_rename_old = {
        item["old_column"] for item in changes.get("possible_renames", [])
    }
    truly_removed = [
        col for col in changes.get("removed_columns", [])
        if col not in possible_rename_old
    ]

    if truly_removed:
        return "NEEDS ATTENTION BEFORE ANALYSIS"

    for change in quality.get("missing_changes", []):
        if change["change"] >= 20:
            return "NEEDS ATTENTION BEFORE ANALYSIS"

    if quality.get("duplicate_change", 0) > 0:
        return "NEEDS ATTENTION BEFORE ANALYSIS"

    return "READY FOR FURTHER ANALYSIS"


def determine_next_action(changes: dict, quality: dict, reconsideration: dict, recommendation: str) -> str:
    """
    Determine the single next best action based on detected evidence.
    Possible actions:
    - Continue analysis
    - Validate affected column
    - Recalculate statistics
    - Review missing values
    - Review type conflict
    - Retrain model
    - Do not proceed with ML
    """
    if recommendation == "NOT SUITABLE WITHOUT ADDITIONAL CORRECTION":
        if any(c.get("conflict_level") == "HIGH" for c in changes.get("type_changes", [])):
            return "Review type conflict"
        return "Do not proceed with ML"

    if reconsideration.get("status") == "RECONSIDER":
        if changes.get("removed_columns"):
            return "Retrain model"
        return "Recalculate statistics"

    if changes.get("type_changes"):
        return "Review type conflict"

    if changes.get("invalid_values"):
        return "Validate affected column"

    if any(item.get("change", 0) >= 10 for item in quality.get("missing_changes", [])):
        return "Review missing values"

    if changes.get("removed_columns"):
        return "Retrain model"

    return "Continue analysis"


def build_step_statuses(
    changes: dict,
    quality: dict,
    statistics: list,
    reconsideration: dict,
    next_action: str,
) -> list[dict]:
    """
    Build real execution states for the 6 Mystery Mission stages:
    DETECT -> ANALYZE -> COMPARE -> RECONSIDER -> RE-PLAN -> REPORT
    """
    has_structural = bool(
        changes.get("added_columns")
        or changes.get("removed_columns")
        or changes.get("type_changes")
        or changes.get("possible_renames")
    )

    detect_status = {
        "step": "DETECT",
        "title": "Structural Detection",
        "detail": "✓ Structural changes detected" if has_structural else "✓ Structure unchanged",
        "kind": "warning" if changes.get("removed_columns") else "success",
        "active": True
    }

    analyze_status = {
        "step": "ANALYZE",
        "title": "Quality & Validity",
        "detail": "✓ Quality & validity analyzed",
        "kind": "warning" if changes.get("invalid_values") else "success",
        "active": True
    }

    compare_status = {
        "step": "COMPARE",
        "title": "Statistical Drift",
        "detail": f"✓ {len(statistics)} numerical comparison(s)" if statistics else "✓ No numeric changes",
        "kind": "success",
        "active": True
    }

    recon_status = reconsideration.get("status", "")
    if recon_status == "RECONSIDER":
        reconsider_detail = "⚠ Previous finding affected"
        reconsider_kind = "warning"
    elif recon_status == "STILL_VALID":
        reconsider_detail = "✓ Finding still valid"
        reconsider_kind = "success"
    else:
        reconsider_detail = "ℹ No prior finding"
        reconsider_kind = "info"

    reconsider_status = {
        "step": "RECONSIDER",
        "title": "Adaptive Reconsideration",
        "detail": reconsider_detail,
        "kind": reconsider_kind,
        "active": True
    }

    replan_status = {
        "step": "RE-PLAN",
        "title": "Next Action Plan",
        "detail": f"🎯 {next_action}",
        "kind": "info",
        "active": True
    }

    report_status = {
        "step": "REPORT",
        "title": "Executive Report",
        "detail": "📄 Report generated",
        "kind": "success",
        "active": True
    }

    return [
        detect_status,
        analyze_status,
        compare_status,
        reconsider_status,
        replan_status,
        report_status,
    ]


def run_intelligence_analysis(
    old_df,
    new_df,
    previous_finding=""
) -> dict:
    """
    Run the complete dataset intelligence workflow.
    """
    from change_engine import (
        compare_datasets,
        compare_statistics
    )
    from quality_engine import compare_quality

    # 1. DETECT
    changes = compare_datasets(old_df, new_df)

    # 2. ANALYZE & COMPARE
    statistics = compare_statistics(old_df, new_df)
    quality = compare_quality(old_df, new_df)

    # Impact & ML assessment
    impact = assess_impact(changes, quality, statistics)
    ml_readiness = assess_ml_readiness(changes, quality, statistics)
    limitations = identify_limitations(changes, quality, statistics)

    # 3. RECONSIDER
    if previous_finding:
        reconsideration = reconsider_previous_finding(previous_finding, changes, quality)
    else:
        reconsideration = {
            "status": "NO_PREVIOUS_FINDING",
            "warnings": [],
            "message": "No previous finding was supplied."
        }

    # 4. RE-PLAN
    recommendation = generate_recommendation(changes, quality)
    next_action = determine_next_action(changes, quality, reconsideration, recommendation)

    # 5. REPORT / STAGE STATUSES
    step_statuses = build_step_statuses(changes, quality, statistics, reconsideration, next_action)

    return {
        "workflow": [
            "Detect",
            "Analyze",
            "Compare",
            "Reconsider",
            "Re-plan",
            "Report"
        ],
        "step_statuses": step_statuses,
        "changes": changes,
        "quality": quality,
        "statistics": statistics,
        "impact": impact,
        "ml_readiness": ml_readiness,
        "limitations": limitations,
        "reconsideration": reconsideration,
        "recommendation": recommendation,
        "next_action": next_action,
    }