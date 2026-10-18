import streamlit as st

from dramagraph import storage
from dramagraph.check_only import MAX_CLIPS, make_canon, run_check_only
from dramagraph.config import get_settings
from pages_ui.shot_cards import show_episode

MAX_MB = 30


def render() -> None:
    st.header("Check clips")
    st.write(
        "Upload up to four clips in story order and say what should be true in them. "
        "DramaGraph reports where each clip breaks continuity."
    )
    settings = get_settings()
    ready = bool(settings.gemini_api_key and settings.video_understanding_model)
    if not ready:
        st.error("Setup is incomplete: GEMINI_API_KEY and VIDEO_UNDERSTANDING_MODEL must be set.")

    files = st.file_uploader(
        f"Clips (MP4, up to {MAX_CLIPS}, {MAX_MB} MB each)",
        type=["mp4", "mov", "webm"], accept_multiple_files=True,
    )
    files = files or []
    if len(files) > MAX_CLIPS:
        st.error(f"Please upload at most {MAX_CLIPS} clips.")
    too_big = [f.name for f in files if f.size > MAX_MB * 1024 * 1024]
    if too_big:
        st.error("Too large: " + ", ".join(too_big))

    with st.form("expected"):
        title = st.text_input("Title", "The Last Samosa")
        st.subheader("What should be true in every clip")
        name = st.text_input("Character name", "Priya")
        appearance = st.text_input("Appearance", "Adult woman, black hair in a low ponytail")
        wardrobe = st.text_input("Wardrobe", "plain mustard-yellow kurta")
        location = st.text_input("Location", "office desk")
        time_of_day = st.text_input("Time of day", "daytime")
        prop_name = st.text_input("Prop (leave empty for none)", "steel lunch box")
        prop_look = st.text_input("What the prop looks like (shape, parts, colour)",
                                  "round single-tier stainless-steel tin with a flat lid")
        st.subheader("Prop state at the end of each clip")
        states = [
            st.text_input(f"Clip {i + 1}: {f.name}", "closed", key=f"state_{i}")
            for i, f in enumerate(files[:MAX_CLIPS])
        ]
        go = st.form_submit_button("Check clips", type="primary")

    if go:
        if not files or len(files) > MAX_CLIPS or too_big or not ready:
            st.error("Fix the problems above, then try again.")
        else:
            canon = make_canon(name, appearance, wardrobe, prop_name, states,
                               location, time_of_day, prop_look)
            bar = st.progress(0.0, "Starting")
            try:
                episode = run_check_only(
                    title, canon, [f.getvalue() for f in files],
                    on_progress=lambda msg, frac: bar.progress(frac, msg),
                )
                st.session_state["last_check_id"] = episode.episode_id
            except Exception as error:
                st.error(f"The check failed: {error}")
            bar.empty()

    last = st.session_state.get("last_check_id")
    if last and storage.exists(storage.episode_path(last)):
        st.divider()
        show_episode(storage.load_episode(last))
