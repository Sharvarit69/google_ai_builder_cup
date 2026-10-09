import streamlit as st

from dramagraph import storage
from pages_ui.shot_cards import show_episode


def render() -> None:
    st.header("Saved")
    rows = storage.list_episodes()
    if not rows:
        st.info("Nothing saved yet.")
        return
    labels = {
        r["episode_id"]: f"{r['title']} · {r['created_at'][:10]} · {r['status']}" for r in rows
    }
    chosen = st.selectbox("Open", list(labels), format_func=labels.get)
    if chosen:
        show_episode(storage.load_episode(chosen))
