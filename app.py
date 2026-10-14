import streamlit as st

from dramagraph import ration
from pages_ui import check_only, episodes, new_episode

st.set_page_config(page_title="DramaGraph", page_icon="🎬", layout="wide")

MAKE, CHECK, SAVED = "Make an episode", "Check my own clips", "Saved episodes"
with st.sidebar:
    st.title("DramaGraph")
    st.caption("Catches continuity mistakes in AI-made video, and fixes only the shots that break.")
    screen = st.radio("Screen", [MAKE, CHECK, SAVED], key="screen", label_visibility="collapsed")
    st.divider()
    st.caption(f"Video budget left: {ration.remaining()} seconds")

if screen == MAKE:
    new_episode.render()
elif screen == CHECK:
    check_only.render()
else:
    episodes.render()
