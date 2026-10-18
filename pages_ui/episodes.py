import streamlit as st

from dramagraph import storage
from pages_ui.shot_cards import show_episode


def visible(rows: list[dict], allowed: tuple) -> list[dict]:
    """In the public demo, list only the demo episodes (all if none are named)."""
    return [r for r in rows if not allowed or r["episode_id"] in allowed]


def render() -> None:
    from dramagraph.config import get_settings

    st.header("Saved")
    settings = get_settings()
    rows = storage.list_episodes()
    if settings.judge_mode:
        rows = visible(rows, settings.judge_episodes)
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
                st.session_state["screen"] = "Make an episode"
                st.session_state.pop("stage", None)

            st.button("Continue working on this episode", on_click=reopen,
                      help="Opens it in the episode screen to re-check, repair or export.")
        show_episode(episode)
