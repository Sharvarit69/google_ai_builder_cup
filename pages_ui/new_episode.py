import uuid
from datetime import datetime, timezone

import streamlit as st

from dramagraph import assembler, generator, pipeline, ration, storage
from dramagraph.canon import build_canon, validate_reference
from dramagraph.config import get_settings
from dramagraph.models import Episode, SeededError, Shot
from dramagraph.parser import parse_script
from dramagraph.planner import MAX_SHOTS, apply_rules, plan_shots, replan
from pages_ui.shot_cards import report_button, show_final, show_shot

SAMPLE = """Priya sits at her office desk in the morning, guarding her closed steel lunch box.
She glances around the office and pulls the lunch box closer.
Priya checks the wall clock. It is finally noon.
Priya opens the lunch box. It is empty, except for a note that says: Thanks for lunch. Boss."""


def _current() -> Episode | None:
    eid = st.session_state.get("episode_id")
    if eid and storage.exists(storage.episode_path(eid)):
        return storage.load_episode(eid)
    return None


def _run(label: str, work) -> bool:
    """Run a slow step with a spinner; show any problem instead of crashing."""
    try:
        with st.spinner(label):
            work()
        return True
    except Exception as error:
        st.error(str(error))
        return False


def _script_step(ep) -> None:
    with st.expander("1. Script", expanded=ep is None):
        title = st.text_input("Title", ep.title if ep else "The Last Samosa")
        uploaded = st.file_uploader("Upload a .txt script, or type below", type=["txt"])
        text = st.text_area("Script (English, one moment per line)",
                            "\n".join(ep.script) if ep else SAMPLE, height=160)
        if st.button("Read script", type="primary" if ep is None else "secondary"):
            source = uploaded.getvalue().decode("utf-8", "ignore") if uploaded else text

            def work():
                lines, beats = parse_script(source)
                new = Episode(
                    episode_id=uuid.uuid4().hex[:12], title=title.strip() or "Untitled",
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    script=lines, beats=beats,
                )
                storage.save_episode(new)
                st.session_state["episode_id"] = new.episode_id

            if _run("Reading the script", work):
                st.rerun()


def _beats_step(ep, locked) -> None:
    st.subheader("2. Pick the beats")
    usable = [b for b in ep.beats if not b.needs_second_person]
    for b in ep.beats:
        if b.needs_second_person:
            st.caption(f"Not supported (needs a second person on screen): {b.action}")
    labels = {b.beat_id: f"{b.beat_id}: {b.action}" for b in usable}
    default = [i for i in ep.picked_beat_ids if i in labels] or list(labels)[:MAX_SHOTS]
    picked_ids = st.multiselect(f"Up to {MAX_SHOTS} beats, one shot each", list(labels),
                                default=default, format_func=labels.get,
                                max_selections=MAX_SHOTS, disabled=locked)
    if st.button("Build details and shot plan", disabled=locked or not picked_ids):
        picked = [b for b in usable if b.beat_id in picked_ids]

        def work():
            ep.picked_beat_ids = [b.beat_id for b in picked]
            ep.canon = build_canon(ep.script, picked)
            ep.shots = [Shot(spec=s) for s in plan_shots(picked, ep.canon)]
            ep.status = "PLANNED"
            storage.save_episode(ep)

        if _run("Writing the character details and shot plan", work):
            st.rerun()


def _details_step(ep, locked) -> None:
    st.subheader("3. Character, prop and setting")
    st.caption("The critic checks each clip against exactly these details, so keep them few and "
               "simple. Video models cannot reliably draw clock times or readable writing; "
               "leave those out. A prop state is what you see at the end of that shot.")
    if locked:
        st.caption("You can still edit these after generating, then use "
                   "'Re-check existing clips' below. That costs no video.")
    c = ep.canon
    with st.form("details"):
        name = st.text_input("Character name", c.character.name)
        appearance = st.text_input("Appearance", c.character.appearance)
        wardrobe = st.text_input("Wardrobe", c.character.wardrobe)
        location = st.text_input("Location", c.location)
        time_of_day = st.text_input("Time of day", c.time_of_day)
        states, remove = {}, {}
        for pi, prop in enumerate(c.props):
            st.markdown(f"**Prop: {prop.name}**")
            cols = st.columns(len(ep.shots))
            for col, shot in zip(cols, ep.shots):
                sid = shot.spec.shot_id
                states[(pi, sid)] = col.text_input(
                    f"At the end of shot {sid[1:]}", prop.state_by_shot.get(sid, ""),
                    key=f"st_{ep.episode_id}_{pi}_{sid}")
            remove[pi] = st.checkbox(f"Remove '{prop.name}' (stop checking it)",
                                     key=f"rm_{ep.episode_id}_{pi}")
        picture = st.file_uploader("Reference picture of the character (optional)",
                                   type=["png", "jpg", "jpeg"])
        consent = st.checkbox("This picture is AI-generated, or shows an adult who gave permission.")
        if st.form_submit_button("Save details"):
            def work():
                if picture is not None:
                    validate_reference(picture.getvalue(), picture.name, consent)
                    storage.save_bytes(generator.reference_path(ep.episode_id), picture.getvalue())
                c.character.name, c.character.appearance = name.strip(), appearance.strip()
                c.character.wardrobe = wardrobe.strip()
                c.location, c.time_of_day = location.strip(), time_of_day.strip()
                for (pi, sid), value in states.items():
                    if value.strip():
                        c.props[pi].state_by_shot[sid] = value.strip()
                    else:
                        c.props[pi].state_by_shot.pop(sid, None)
                c.props = [p for pi, p in enumerate(c.props) if not remove.get(pi)]
                apply_rules([s.spec for s in ep.shots], c)
                storage.save_episode(ep)

            if _run("Saving", work):
                st.rerun()
    if storage.exists(generator.reference_path(ep.episode_id)):
        st.caption("A reference picture is saved for this episode.")


def _plan_step(ep, locked) -> None:
    st.subheader("4. Shot plan")
    for shot in ep.shots:
        s = shot.spec
        with st.container(border=True):
            st.markdown(f"**Shot {s.shot_id[1:]}** · {s.shot_type}, {s.camera_motion}")
            st.write(s.action)
            st.caption("Must show: " + "; ".join(s.must_show))
    instruction = st.text_input("Change the plan (for example: make shot 2 a close-up of her hands)",
                                disabled=locked)
    if st.button("Re-plan", disabled=locked or not instruction.strip()):
        picked = [b for b in ep.beats if b.beat_id in ep.picked_beat_ids]

        def work():
            specs = replan(picked, ep.canon, [s.spec for s in ep.shots], instruction)
            ep.shots = [Shot(spec=s) for s in specs]
            storage.save_episode(ep)

        if _run("Re-planning", work):
            st.rerun()


def _generate_step(ep, started) -> None:
    st.subheader("5. Generate and check")
    settings = get_settings()
    unlocked = st.session_state.get("gen_unlocked", False)
    todo = [s for s in ep.shots if s.status in ("PLANNED", "GENERATING", "SKIPPED")]
    needed = sum(s.spec.duration_seconds for s in todo)
    left = ration.remaining()

    if not started:
        with st.expander("Test the critic: plant an error on purpose"):
            st.caption("One shot's first clip is asked for with a deliberate mistake, to show "
                       "the critic catching it and the repair fixing it. It is labelled as a "
                       "seeded test everywhere.")
            ids = ["None"] + [s.spec.shot_id for s in ep.shots]
            seed_shot = st.selectbox("Shot", ids, format_func=lambda i: i if i == "None" else f"Shot {i[1:]}")
            seed_kind = st.radio("What to get wrong", ["prop", "wardrobe"], horizontal=True)
            seed_value = st.text_input("Wrong value", "open, with the lid off on the desk"
                                       if seed_kind == "prop" else "plain bright red kurta")
    else:
        seed_shot = "None"

    if not settings.veo_model:
        st.error("Setup is incomplete: VEO_MODEL must be set.")
    elif not unlocked:
        st.info("Enter the passcode in the sidebar to generate video.")
    elif todo and left < needed:
        st.warning(f"These shots need {needed} seconds of video but only {left} remain in the ration.")

    if todo:
        label = "Continue generating" if started else f"Generate {len(todo)} shots (about {needed} seconds of video)"
        if st.button(label, type="primary",
                     disabled=not (unlocked and settings.veo_model and left >= ep.shots[0].spec.duration_seconds)):
            if not started and seed_shot != "None" and seed_value.strip():
                ep.seeded_error = SeededError(shot_id=seed_shot, kind=seed_kind, value=seed_value.strip())
            bar = st.progress(0.0, "Starting. Each shot takes a minute or two. Keep this page open.")
            try:
                pipeline.run_generation(ep, on_progress=lambda m, f: bar.progress(min(f, 1.0), m))
            except Exception as error:
                st.error(f"Generation stopped: {error}")
            st.rerun()

    if not any(s.attempts for s in ep.shots):
        return
    if st.button("Re-check existing clips (no new video)",
                 help="Runs the critic again on the clips you already have, using the "
                      "current details. Use it after editing section 3."):
        bar = st.progress(0.0, "Re-checking")
        try:
            pipeline.recheck(ep, on_progress=lambda m, f: bar.progress(min(f, 1.0), m))
        except Exception as error:
            st.error(f"Re-check stopped: {error}")
        st.rerun()
    for shot in ep.shots:
        with st.container(border=True):
            show_shot(shot)
            if shot.status != "FLAGGED":
                continue
            sid = shot.spec.shot_id
            a, r, b, c = st.columns(4)
            if a.button("Accept anyway", key=f"ok_{sid}", disabled=shot.chosen_attempt is None):
                if _run("Saving", lambda: pipeline.overrule(ep, sid)):
                    st.rerun()
            if r.button("Repair automatically", key=f"auto_{sid}",
                        disabled=not unlocked or shot.chosen_attempt is None):
                if _run("Generating and checking. This takes a minute or two.",
                        lambda: pipeline.auto_repair(ep, sid)):
                    st.rerun()
            if c.button("Drop shot", key=f"drop_{sid}"):
                if _run("Saving", lambda: pipeline.drop(ep, sid)):
                    st.rerun()
            fix = st.text_input("Fix with my instruction", key=f"fix_{sid}",
                                placeholder="for example: the lunch box lid must stay on")
            if b.button("Regenerate with my fix", key=f"go_{sid}",
                        disabled=not (unlocked and fix.strip())):
                if _run("Generating and checking. This takes a minute or two.",
                        lambda: pipeline.repair_with_instruction(ep, sid, fix)):
                    st.rerun()

    _export_step(ep)
    report_button(ep)


def _export_step(ep) -> None:
    st.subheader("6. Export")
    ok, reason = pipeline.can_export(ep)
    if not ok:
        st.info(reason)
        return
    kept = [s for s in ep.shots if s.status != "DROPPED"]
    beats = {b.beat_id: b for b in ep.beats}
    with st.form("export"):
        st.caption("Captions are burned into the video. Leave one empty for no caption.")
        texts = {
            s.spec.beat_id: st.text_input(f"Caption for shot {s.spec.shot_id[1:]}",
                                          beats[s.spec.beat_id].caption if s.spec.beat_id in beats else "",
                                          key=f"cap_{ep.episode_id}_{s.spec.shot_id}")
            for s in kept
        }
        if st.form_submit_button("Export video", type="primary"):
            def work():
                for beat_id, text in texts.items():
                    if beat_id in beats:
                        beats[beat_id].caption = text.strip()
                assembler.assemble(ep)

            if _run("Stitching the shots into one video", work):
                st.rerun()
    show_final(ep)


def render() -> None:
    st.header("New episode")
    ep = _current()
    if ep is not None and st.button("Start a new episode"):
        st.session_state.pop("episode_id", None)
        st.rerun()
    _script_step(ep)
    if ep is None:
        return
    started = any(s.attempts for s in ep.shots)
    if started:
        st.caption("Generation has started, so the plan is locked. Start a new episode to change it.")
    _beats_step(ep, started)
    if ep.canon is None or not ep.shots:
        return
    _details_step(ep, started)
    _plan_step(ep, started)
    _generate_step(ep, started)
