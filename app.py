import hmac

import streamlit as st

from dramagraph import ration
from dramagraph.config import get_settings
from pages_ui import check_only, episodes, new_episode

st.set_page_config(page_title="DramaGraph", page_icon="🎬", layout="wide")
st.sidebar.title("DramaGraph")
st.sidebar.caption("Continuity checker for AI-generated short videos")
screen = st.sidebar.radio("Screen", ["New episode", "Check clips", "Saved"])

settings = get_settings()
st.sidebar.divider()
st.sidebar.metric("Veo seconds left", ration.remaining())
if st.session_state.get("gen_unlocked"):
    st.sidebar.success("Generation unlocked")
elif not settings.gen_passcode:
    st.sidebar.caption("Generation is switched off (no passcode configured).")
else:
    entered = st.sidebar.text_input("Passcode to generate video", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), settings.gen_passcode.encode()):
            st.session_state["gen_unlocked"] = True
            st.rerun()
        else:
            st.sidebar.error("Wrong passcode")

if screen == "New episode":
    new_episode.render()
elif screen == "Check clips":
    check_only.render()
else:
    episodes.render()
