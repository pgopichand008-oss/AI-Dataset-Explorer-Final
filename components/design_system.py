"""
components/design_system.py — Compact Data Scientist Workspace design system.
Enforces compact information-dense typography, dark-mode laboratory theme, and deep CSS styling to eliminate white panels.
"""

from __future__ import annotations
import streamlit as st

DARK_TOKENS: dict[str, str] = {
    "bg-app": "#0b0e14",
    "bg-surface": "#121721",
    "bg-surface-alt": "#1a202c",
    "bg-inverse": "#0b0e14",
    "border": "#232d3f",
    "border-strong": "#2d3748",
    "text-primary": "#edf2f7",
    "text-secondary": "#a0aec0",
    "text-inverse": "#edf2f7",
    "accent": "#6366f1",
    "accent-soft": "rgba(99, 102, 241, 0.12)",
    "accent-2": "#38bdf8",
    "success": "#10b981",
    "success-soft": "rgba(16, 185, 129, 0.12)",
    "warning": "#f59e0b",
    "warning-soft": "rgba(245, 158, 11, 0.12)",
    "danger": "#ef4444",
    "danger-soft": "rgba(239, 68, 68, 0.12)",
    "shadow": "0 2px 4px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.4)",
}


def get_tokens(theme: str = "dark") -> dict[str, str]:
    # Dark Theme Only — non-negotiable architecture requirement
    return DARK_TOKENS


def _build_css(t: dict[str, str]) -> str:
    return f"""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: -0.01em;
        font-size: 13px;
    }}

    .stApp {{
        background: {t['bg-app']} !important;
        color: {t['text-primary']} !important;
    }}

    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 2.5rem;
        max-width: 1480px;
    }}

    /* ---------------- COMPACT TYPOGRAPHY ---------------- */
    h1 {{ font-size: 1.35rem !important; font-weight: 800 !important; letter-spacing: -0.02em !important; margin-bottom: 0.4rem !important; color: {t['text-primary']} !important; }}
    h2 {{ font-size: 1.15rem !important; font-weight: 750 !important; letter-spacing: -0.015em !important; margin-top: 0.8rem !important; margin-bottom: 0.3rem !important; color: {t['text-primary']} !important; }}
    h3 {{ font-size: 1.02rem !important; font-weight: 700 !important; margin-top: 0.6rem !important; margin-bottom: 0.25rem !important; color: {t['text-primary']} !important; }}
    h4 {{ font-size: 0.92rem !important; font-weight: 650 !important; margin-top: 0.5rem !important; margin-bottom: 0.2rem !important; color: {t['text-primary']} !important; }}
    p, span, label, div {{ font-size: 0.85rem; line-height: 1.45; }}

    /* ---------------- TOP BANNER ---------------- */
    .top-app-banner {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, {t['bg-surface']}, {t['bg-surface-alt']});
        border: 1px solid {t['border']};
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 14px;
        box-shadow: {t['shadow']};
    }}
    .top-app-title {{
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {t['text-primary']};
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .top-app-subtitle {{
        font-size: 0.76rem;
        color: {t['text-secondary']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .top-status-pills {{
        display: flex;
        gap: 6px;
        align-items: center;
    }}

    /* ---------------- HERO / COMMAND CENTER ---------------- */
    .command-center-hero {{
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(56, 189, 248, 0.05));
        border: 1px solid {t['border-strong']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }}
    .command-center-hero::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, {t['accent']}, {t['accent-2']});
    }}
    .hero-title-text {{
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {t['text-primary']};
        margin-bottom: 2px;
    }}
    .hero-subtitle-text {{
        font-size: 0.84rem;
        color: {t['text-secondary']};
        margin-bottom: 10px;
    }}

    /* ---------------- DATA INPUT UPLOAD PANEL ---------------- */
    .data-input-panel {{
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid {t['accent']}44;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 14px;
    }}
    .data-input-header {{
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        color: {t['accent-2']};
        letter-spacing: 0.06em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 5px;
    }}

    /* ---------------- CARDS & CONTAINERS ---------------- */
    .info-card {{
        padding: 12px 14px;
        border-radius: 10px;
        background: {t['bg-surface']};
        border: 1px solid {t['border']};
        box-shadow: {t['shadow']};
        margin-bottom: 10px;
    }}
    .info-card-title {{
        font-size: 0.68rem;
        font-weight: 700;
        color: {t['text-secondary']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }}
    .info-card-value {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {t['text-primary']};
        letter-spacing: -0.02em;
    }}
    .info-card-description {{
        font-size: 0.78rem;
        color: {t['text-secondary']};
        margin-top: 2px;
    }}

    /* ---------------- FINDING CARDS ---------------- */
    .finding-card {{
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border: 1px solid {t['border']};
        font-size: 0.82rem;
        line-height: 1.45;
        background: {t['bg-surface']};
    }}
    .finding-success {{ background: {t['success-soft']}; border-color: {t['success']}44; color: {t['text-primary']}; }}
    .finding-warning {{ background: {t['warning-soft']}; border-color: {t['warning']}44; color: {t['text-primary']}; }}
    .finding-danger  {{ background: {t['danger-soft']};  border-color: {t['danger']}44;  color: {t['text-primary']}; }}
    .finding-info    {{ background: {t['accent-soft']};  border-color: {t['accent']}44;  color: {t['text-primary']}; }}

    /* ---------------- SECTION HEADERS ---------------- */
    .section-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: {t['text-primary']};
        letter-spacing: -0.02em;
        margin-top: 4px;
        margin-bottom: 2px;
    }}
    .section-subtitle {{
        color: {t['text-secondary']};
        font-size: 0.82rem;
        margin-bottom: 12px;
    }}

    /* ---------------- STATUS PILLS ---------------- */
    .status-pill {{
        display: inline-block;
        padding: 3px 9px;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 700;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid transparent;
    }}
    .status-success {{ background: {t['success-soft']}; color: {t['success']}; border-color: {t['success']}44; }}
    .status-warning {{ background: {t['warning-soft']}; color: {t['warning']}; border-color: {t['warning']}44; }}
    .status-error   {{ background: {t['danger-soft']};  color: {t['danger']};  border-color: {t['danger']}44; }}
    .status-info    {{ background: {t['accent-soft']};   color: {t['accent-2']}; border-color: {t['accent']}44; }}

    /* ---------------- STREAMLIT NATIVE OVERRIDES (FIX WHITE PANELS) ---------------- */
    div[data-testid="stMetric"] {{
        background: {t['bg-surface']} !important;
        border: 1px solid {t['border']} !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        box-shadow: {t['shadow']} !important;
    }}
    div[data-testid="stMetricLabel"] {{ color: {t['text-secondary']} !important; font-weight: 600; font-size: 0.75rem !important; }}
    div[data-testid="stMetricValue"] {{ color: {t['text-primary']} !important; font-weight: 800; font-size: 1.15rem !important; }}

    /* DATAFRAMES AND TABLES DARK OVERRIDES */
    div[data-testid="stDataFrame"], div[data-testid="stTable"], .dataframe, table {{
        background-color: {t['bg-surface']} !important;
        color: {t['text-primary']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
    }}
    thead tr th, tbody tr td, tbody tr th {{
        background-color: {t['bg-surface']} !important;
        color: {t['text-primary']} !important;
        border-color: {t['border']} !important;
    }}

    .stButton > button {{
        border-radius: 8px;
        font-weight: 650;
        font-size: 0.82rem !important;
        padding: 5px 12px !important;
        border: 1px solid {t['border-strong']} !important;
        background: {t['bg-surface']} !important;
        color: {t['text-primary']} !important;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: {t['accent-2']} !important;
        color: {t['accent-2']} !important;
    }}
    button[kind="primary"] {{
        background: {t['accent']} !important;
        border-color: {t['accent']} !important;
        color: white !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        padding: 4px;
        background: #121721 !important;
        border-radius: 12px;
        border: 1px solid #232d3f !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.82rem !important;
        font-weight: 650;
        color: #a0aec0 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 0.05) !important;
        color: #edf2f7 !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: #1a202c !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2) !important;
    }}

    div[data-testid="stFileUploader"] {{
        border-radius: 10px;
        background: {t['bg-surface']} !important;
        border: 1px dashed {t['border-strong']} !important;
        padding: 8px !important;
    }}

    .stDataFrame {{ border-radius: 10px; overflow: hidden; border: 1px solid {t['border']}; }}

    section[data-testid="stSidebar"] {{
        background: {t['bg-app']} !important;
        border-right: 1px solid {t['border-strong']} !important;
        padding-top: 1rem !important;
    }}
    section[data-testid="stSidebar"] * {{ color: {t['text-primary']} !important; }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.08) !important; margin: 10px 0 !important; }}

    /* EXPANDER & SELECTBOX DARK OVERRIDES */
    div[data-testid="stExpander"] {{
        background: {t['bg-surface']} !important;
        border: 1px solid {t['border']} !important;
        border-radius: 10px !important;
    }}

    .footer {{
        text-align: center;
        padding: 16px;
        color: {t['text-secondary']};
        font-size: 0.78rem;
    }}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {t['border-strong']}; border-radius: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    """


def inject_design_system(theme: str = "dark") -> None:
    tokens = get_tokens("dark")
    st.markdown(f"<style>{_build_css(tokens)}</style>", unsafe_allow_html=True)
