"""
Custom CSS styles and theme definitions for the PDF Insight Agent.
Consistent blue/purple accent palette, clean typography, responsive layout.
"""

from __future__ import annotations


def get_custom_css() -> str:
    """Returns CSS string with blue/purple modern accent design."""
    return """
<style>
    /* Global accents and font smoothing */
    :root {
        --primary-accent: #4F46E5;
        --secondary-accent: #7C3AED;
        --accent-blue: #2563EB;
        --surface-bg: #F8FAFC;
        --card-border: #E2E8F0;
        --text-primary: #0F172A;
        --text-secondary: #475569;
    }

    /* Streamlit Main Container Spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Header styling */
    .main-title-container {
        background: linear-gradient(135deg, #4338CA 0%, #6366F1 50%, #8B5CF6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.2);
    }
    .main-title-container h1 {
        color: white !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-title-container p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin: 0.35rem 0 0 0;
    }

    /* Badges & Pills */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 0.4rem;
    }
    .badge-primary {
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
    }
    .badge-purple {
        background-color: #F5F3FF;
        color: #7C3AED;
        border: 1px solid #DDD6FE;
    }
    .badge-success {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }
    .badge-gray {
        background-color: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
    }

    /* Cards */
    .ui-card {
        background: #FFFFFF;
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .ui-card:hover {
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);
        border-color: #C7D2FE;
    }

    .ui-card-highlight {
        background: linear-gradient(180deg, #FAFAFE 0%, #FFFFFF 100%);
        border-left: 4px solid var(--primary-accent);
    }

    /* Insight Block */
    .insight-quote {
        background: #F8FAFC;
        border-left: 3px solid #6366F1;
        padding: 0.75rem 1rem;
        border-radius: 0 6px 6px 0;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #1E293B;
    }

    /* Citation Box */
    .citation-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #3B82F6;
        border-radius: 6px;
        padding: 0.65rem 0.85rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
    }

    /* Metric card custom styling */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #C7D2FE;
    }

    /* Sidebar navigation styling */
    .sidebar-section-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
"""


def render_header(title: str, subtitle: str) -> None:
    """Renders the top banner header."""
    import streamlit as st

    html = f"""
    <div class="main-title-container">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
