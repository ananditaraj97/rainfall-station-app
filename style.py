"""
style.py - shared branding/theme helpers for all pages of the
Bhima Basin Climate & SWAT Data Toolkit.
"""

import streamlit as st

NAVY = "#1f3a5f"
ACCENT = "#c1622d"
LIGHT_BG = "#f5f6f8"

# Edit these two lines to change the sidebar branding text.
APP_TITLE = "Bhima Basin Climate Toolkit"
APP_SUBTITLE = "IMD &middot; CMIP6 &middot; SWAT DATA PLATFORM"

# (label shown in custom nav, path to the page file)
NAV_PAGES = [
    ("Home / IMD Station Extraction", "app.py"),
    ("CMIP6 Models", "pages/2_CMIP6_Models.py"),
    ("Model Evaluation & Ranking", "pages/3_Model_Evaluation_Ranking.py"),
    ("Future Climate Projections", "pages/4_Future_Climate_Projections.py"),
    ("Ensemble & Uncertainty", "pages/5_Ensemble_Uncertainty.py"),
    ("SWAT Weather Files", "pages/6_SWAT_Weather_Files.py"),
    ("Documentation", "pages/7_Documentation.py"),
]


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
        /* hide the default auto-generated page list - replaced by custom nav below branding */
        [data-testid="stSidebarNav"] {{
            display: none;
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
                {APP_TITLE}
              </div>
              <div style="font-size:0.78rem; color:#c9d4e3; letter-spacing:0.04em;">
                {APP_SUBTITLE}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        for label, path in NAV_PAGES:
            st.page_link(path, label=label)
        st.markdown("---")


def footer():
    st.markdown("---")
    st.caption("Developed by: Ms. Anandita Raj & Dr. Raj Mohan Singh — Department of Civil Engineering, MNNIT Allahabad")
