import streamlit as st

from dramagraph import ration
from dramagraph.config import get_settings
from pages_ui import check_only, episodes, new_episode

st.set_page_config(page_title="DramaGraph", page_icon="🎬", layout="wide")

MAKE, CHECK, SAVED = "Make an episode", "Check my own clips", "Saved episodes"

if get_settings().judge_mode:
    # Public demo: browse saved episodes, and a few clip checks a day for all visitors.
    # Nothing else can call Gemini or Veo, or save anything (see dramagraph/judge.py).
    from dramagraph import judge

    settings = get_settings()
    with st.sidebar:
        st.title("DramaGraph")
        st.caption("Catches continuity mistakes in AI-made video, and fixes only the shots that break.")
        screen = st.radio("Screen", [SAVED, CHECK], key="judge_screen", label_visibility="collapsed")
        st.divider()
        n = settings.judge_checks_per_day
        st.info(f"**Public demo.** Browse a saved episode, or check up to {n} "
                f"clip{'s' if n != 1 else ''} a day (shared by all visitors). "
                "Making new video is switched off to protect API costs; the demo video shows it.")
        st.caption(f"Free clip checks left today: {judge.remaining()} of {settings.judge_checks_per_day}")
    if screen == CHECK:
        check_only.render(judge_mode=True)
    else:
        episodes.render(judge=True)
    st.stop()

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
