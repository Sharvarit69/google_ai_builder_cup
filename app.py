import streamlit as st

from pages_ui import check_only, episodes

st.set_page_config(page_title="DramaGraph", page_icon="🎬", layout="wide")
st.sidebar.title("DramaGraph")
st.sidebar.caption("Continuity checker for AI-generated short videos")
screen = st.sidebar.radio("Screen", ["Check clips", "Saved checks"])

if screen == "Check clips":
    check_only.render()
else:
    episodes.render()
