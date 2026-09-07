# FILE: config.py

import os
from google import genai


def get_gemini_api_key():
    """Resolves Gemini API key securely from environment variables or Streamlit Secrets."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
    if key:
        return key

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]
            if "GEMINI_KEY" in st.secrets:
                return st.secrets["GEMINI_KEY"]
    except Exception:
        pass

    return None


def get_gemini_client():
    """Initializes Google GenAI Client if API key is present."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def get_gemini_model():
    """Returns primary supported Google GenAI model name."""
    return "gemini-3.6-flash"
