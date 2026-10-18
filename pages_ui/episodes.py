import streamlit as st

from dramagraph import storage
from pages_ui.shot_cards import show_episode


def visible(rows: list[dict], allowed: tuple) -> list[dict]:
    """In judge mode, only the episodes chosen for visitors (all if none are named)."""
    return [r for r in rows if not allowed or r["episode_id"] in allowed]


def show_plan(episode) -> None:
    """What the critic checked against: the character, setting, props and rules."""
    c = episode.canon
    if c is None:
        return
    with st.expander("The plan the critic checked every clip against"):
        st.markdown(f"**Character:** {c.character.name}. {c.character.appearance}. "
                    f"Wearing {c.character.wardrobe}.")
        st.markdown(f"**Setting:** {c.location}, {c.time_of_day}.")
        for prop in c.props:
            states = ", ".join(f"shot {sid[1:]}: {state}" for sid, state in prop.state_by_shot.items())
            st.markdown(f"**{prop.name}:** {prop.description}" + (f" (end of {states})" if states else ""))
        for shot in episode.shots:
            st.markdown(f"**Shot {shot.spec.shot_id[1:]}:** {shot.spec.action}")
        if c.rules:
            st.markdown("**Rules learned from the creator:**")
            for rule in c.rules:
                st.markdown(f"- {rule.text}")


def render(judge: bool = False) -> None:
    st.header("Saved episodes" if judge else "Saved")
    rows = storage.list_episodes()
    if judge:
        from dramagraph.config import get_settings

        rows = visible(rows, get_settings().judge_episodes)
    if not rows:
        st.info("Nothing saved yet.")
        return
    labels = {
        r["episode_id"]: f"{r['title']} · {r['created_at'][:10]} · {r['status']}" for r in rows
    }
    chosen = st.selectbox("Open", list(labels), format_func=labels.get)
    if chosen:
        episode = storage.load_episode(chosen)
        if judge:
            show_plan(episode)
        elif episode.mode == "GENERATE":
            def reopen(episode_id=chosen):
                st.session_state["episode_id"] = episode_id
                st.session_state["screen"] = "Make an episode"
                st.session_state.pop("stage", None)

            st.button("Continue working on this episode", on_click=reopen,
                      help="Opens it in the episode screen to re-check, repair or export.")
        show_episode(episode)
