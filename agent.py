"""
agent.py — Main entrance / wrapper for the dataset intelligence agent functions.
"""

from engine.ai_agent import (
    run_intelligence_analysis,
    reconsider_previous_finding,
    assess_impact,
    assess_ml_readiness,
    generate_recommendation,
    determine_next_action,
    build_step_statuses,
)

from engine.evidence import build_evidence
from engine.health import calculate_health
from engine.priority_engine import build_prioritized_insights
from engine.story_engine import build_data_story
from engine.recommendation_engine import recommend_next_analysis
from engine.anomaly_engine import investigate_anomalies
from engine.dictionary_engine import build_data_dictionary
from engine.reconsideration import reconsider_dataset
from engine.workflow import run_workflow
from engine.executive_engine import build_executive_intelligence
from engine.ask_engine import ask_dataset

__all__ = [
    "run_intelligence_analysis",
    "reconsider_previous_finding",
    "assess_impact",
    "assess_ml_readiness",
    "generate_recommendation",
    "determine_next_action",
    "build_step_statuses",
    "build_evidence",
    "calculate_health",
    "build_prioritized_insights",
    "build_data_story",
    "recommend_next_analysis",
    "investigate_anomalies",
    "build_data_dictionary",
    "reconsider_dataset",
    "run_workflow",
    "build_executive_intelligence",
    "ask_dataset",
]

