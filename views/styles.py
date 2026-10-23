import streamlit as st

def inject_custom_styles():
    """Injects high-contrast dark theme CSS into the Streamlit app."""
    st.markdown(
        """
        <style>
          .stApp { background-color: #0d1117; color: #c9d1d9; }
          [data-testid="stSidebar"] { background-color: #111820; }
          .kpi-card {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 9px; padding: 14px; text-align: center;
          }
          .kpi-label { color: #9da7b3; font-size: 0.85rem; font-weight: 600; }
          .kpi-value { color: #f0f6fc; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
          .kpi-delta { font-size: 0.82rem; font-weight: 600; margin-top: 2px; }
          .delta-red { color: #ff5555; }
          .delta-green { color: #2ecc71; }
          
          .action-panel {
            background-color: #161b22; border: 1px solid #30363d;
            border-radius: 10px; padding: 18px;
          }
          .batt-card {
            background-color: #555e14; border: 1px solid #7d8b1e;
            border-radius: 8px; padding: 12px; margin-bottom: 12px; color: #fff;
          }
          .dr-card {
            background-color: #1a4d75; border: 1px solid #236fa8;
            border-radius: 8px; padding: 12px; margin-bottom: 12px; color: #fff;
          }
          .advisory-box {
            background-color: #11202e; border-left: 4px solid #00d4ff;
            padding: 12px; border-radius: 6px; font-size: 0.85rem; color: #c9d1d9; line-height: 1.45;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
