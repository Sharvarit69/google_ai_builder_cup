import streamlit as st

from dramagraph.config import get_settings

st.set_page_config(page_title="DramaGraph", page_icon="🎬")
st.title("DramaGraph")
st.write("Continuity checker for AI-generated short videos. Build in progress.")

s = get_settings()
st.subheader("Setup status")
st.write("Gemini key:", "set" if s.gemini_api_key else "missing")
st.write("Storage:", f"bucket {s.gcs_bucket}" if s.gcs_bucket else "local folder")
st.write("Veo budget (seconds):", s.veo_budget_seconds)
