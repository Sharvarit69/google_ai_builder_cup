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
        episode = storage.load_episode(chosen)
        if episode.mode == "GENERATE":
            def reopen(episode_id=chosen):
                st.session_state["episode_id"] = episode_id
                st.session_state["screen"] = "New episode"

            st.button("Continue working on this episode", on_click=reopen,
                      help="Opens it in the episode screen to re-check, repair or export.")
        show_episode(episode)
