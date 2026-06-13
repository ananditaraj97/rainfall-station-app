"""
style.py - shared branding/theme helpers for all pages of the
Bhima Basin Climate & SWAT Data Toolkit.
"""

import streamlit as st

NAVY = "#1f3a5f"
ACCENT = "#c1622d"
LIGHT_BG = "#f5f6f8"


def inject_css():
    st.markdown(
        f"""
        <style>
        h1, h2, h3 {{
            color: {NAVY};
        }}
        [data-testid="stSidebar"] {{
            background-color: {NAVY};
        }}
        [data-testid="stSidebar"] * {{
            color: #ffffff !important;
        }}
        [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {{
            color: #c9d4e3 !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: #3b5a82;
        }}
        .stButton > button, .stDownloadButton > button {{
            border-color: {ACCENT};
        }}
        [data-testid="stMetricValue"] {{
            color: {ACCENT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_branding():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding:0.4rem 0 0.8rem 0;">
              <div style="font-size:1.15rem; font-weight:700; color:#ffffff;">
                Bhima Basin Climate Toolkit
              </div>
              <div style="font-size:0.78rem; color:#c9d4e3; letter-spacing:0.04em;">
                IMD &middot; CMIP6 &middot; SWAT DATA PLATFORM
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")


def footer():
    st.markdown("---")
    st.caption("Developed by: Ms. Anandita Raj & Dr. Raj Mohan Singh — Department of Civil Engineering, MNNIT Allahabad")
