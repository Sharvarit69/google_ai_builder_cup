"""Make an episode, in four steps shown one at a time: Script, Plan, Shots, Export."""
import hmac
import uuid
from datetime import datetime, timezone

import streamlit as st

from dramagraph import assembler, generator, learning, pipeline, ration, storage
from dramagraph.canon import MAX_PROPS, build_canon, validate_reference
from dramagraph.config import get_settings
from dramagraph.models import Episode, LearnedRule, Prop, SeededError, Shot
from dramagraph.parser import parse_script
from dramagraph.planner import MAX_SHOTS, apply_rules, plan_shots, replan
from pages_ui.shot_cards import STATUS, chosen, play, report_button, show_final, show_shot

SAMPLE = """Priya sits at her office desk in the morning, guarding her closed steel lunch box.
She glances around the office and pulls the lunch box closer.
Priya checks the wall clock. It is finally noon.
Priya opens the lunch box. It is empty, except for a note that says: Thanks for lunch. Boss."""

SCRIPT, PLAN, SHOTS, EXPORT = "1 · Script", "2 · Plan", "3 · Shots", "4 · Export"
ICON = {"ACCEPTED": "✅", "OVERRULED": "☑️", "FLAGGED": "⚠️", "DROPPED": "🗑️",
        "SKIPPED": "⏭️", "PLANNED": "○", "GENERATING": "…"}


# ---------- helpers ----------

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


def _go(stage: str) -> None:
    """Move to a step on the next run (the step switch cannot change once drawn)."""
    st.session_state["stage_next"] = stage


def _started(ep) -> bool:
    return bool(ep and any(s.attempts for s in ep.shots))


def _unlocked() -> bool:
    return bool(st.session_state.get("gen_unlocked"))


def _unlock_box(key: str) -> bool:
    """Ask for the passcode where it is needed, instead of in the sidebar."""
    if _unlocked():
        return True
    code = get_settings().gen_passcode
    if not code:
        st.caption("Video generation is switched off on this copy of the app.")
        return False
    entered = st.text_input("Passcode to make video", type="password", key=f"pass_{key}",
                            help="Making video costs money, so it needs a passcode.")
    if entered:
        if hmac.compare_digest(entered.encode(), code.encode()):
            st.session_state["gen_unlocked"] = True
            st.rerun()
        st.error("That passcode is not right.")
    return False


# ---------- step 1: script ----------

def _script_step(ep) -> None:
    if _started(ep):
        st.caption("Video has been made for this episode, so its script is fixed. "
                   "Start a new episode to use a different script.")
        st.text("\n".join(ep.script))
        return

    title = st.text_input("Title", ep.title if ep else "The Last Samosa")
    text = st.text_area("Script: one moment per line, in English",
                        "\n".join(ep.script) if ep else SAMPLE, height=150)
    with st.expander("Or upload a .txt file"):
        uploaded = st.file_uploader("Script file", type=["txt"], label_visibility="collapsed")
    if st.button("Read the script", type="primary" if ep is None else "secondary"):
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

    if ep is None or not ep.beats:
        return
    st.divider()
    usable = [b for b in ep.beats if not b.needs_second_person]
    labels = {b.beat_id: b.action for b in usable}
    default = [i for i in ep.picked_beat_ids if i in labels] or list(labels)[:MAX_SHOTS]
    picked_ids = st.multiselect(f"Moments to film (up to {MAX_SHOTS}, one shot each)", list(labels),
                                default=default, format_func=labels.get, max_selections=MAX_SHOTS)
    skipped = [b.action for b in ep.beats if b.needs_second_person]
    if skipped:
        st.caption("Left out because they need a second person on screen: " + "; ".join(skipped))
    if st.button("Plan the shots", type="primary", disabled=not picked_ids):
        picked = [b for b in usable if b.beat_id in picked_ids]

        def work():
            ep.picked_beat_ids = [b.beat_id for b in picked]
            ep.canon = build_canon(ep.script, picked)
            ep.shots = [Shot(spec=s) for s in plan_shots(picked, ep.canon)]
            ep.status = "PLANNED"
            storage.save_episode(ep)

        if _run("Planning the character, props and shots", work):
            _go(PLAN)
            st.rerun()


# ---------- step 2: plan ----------

def _canon_summary(ep) -> None:
    c = ep.canon
    lines = [f"**{c.character.name}** · {c.character.appearance} · wearing {c.character.wardrobe}",
             f"**Setting** · {c.location}, {c.time_of_day}"]
    for p in c.props:
        states = " → ".join(p.state_by_shot.get(s.spec.shot_id, "?") for s in ep.shots)
        look = f" ({p.description})" if p.description else ""
        lines.append(f"**{p.name}**{look} · {states}")
    if c.rules:
        lines.append("**Your rules** · " + " · ".join(r.text.rstrip(".") for r in c.rules))
    st.markdown("  \n".join(lines))


def _details_form(ep) -> None:
    c = ep.canon
    st.caption("The critic checks every clip against these. Keep them simple: video models "
               "cannot reliably draw clock times or readable writing. A prop state is what "
               "you see at the end of that shot.")
    with st.form("details"):
        name = st.text_input("Character name", c.character.name)
        appearance = st.text_input("Appearance", c.character.appearance)
        wardrobe = st.text_input("Wardrobe", c.character.wardrobe)
        a, b = st.columns(2)
        location = a.text_input("Location", c.location)
        time_of_day = b.text_input("Time of day", c.time_of_day)
        states, remove, names, looks = {}, {}, {}, {}
        slots = list(range(len(c.props))) + ([len(c.props)] if len(c.props) < MAX_PROPS else [])
        for pi in slots:
            prop = c.props[pi] if pi < len(c.props) else None
            st.markdown(f"**Prop: {prop.name}**" if prop else "**Add a prop** (optional)")
            k = f"{ep.episode_id}_{pi}"
            names[pi] = st.text_input("Name", prop.name if prop else "", key=f"pn_{k}")
            looks[pi] = st.text_input(
                "What it looks like (shape, parts, material, colour)",
                prop.description if prop else "", key=f"pd_{k}",
                placeholder="round two-tier stainless-steel tiffin with side clasps")
            cols = st.columns(len(ep.shots))
            for col, shot in zip(cols, ep.shots):
                sid = shot.spec.shot_id
                states[(pi, sid)] = col.text_input(
                    f"End of shot {sid[1:]}", prop.state_by_shot.get(sid, "") if prop else "",
                    key=f"st_{k}_{sid}")
            if prop:
                remove[pi] = st.checkbox(f"Stop checking the {prop.name}", key=f"rm_{k}")
        st.markdown("**Your rules**")
        drop_rule = {}
        for ri, rule in enumerate(c.rules):
            scope = "every shot" if not rule.shot_ids else "shot " + ", ".join(i[1:] for i in rule.shot_ids)
            drop_rule[ri] = st.checkbox(f"Remove: {rule.text} ({scope})", key=f"rr_{ep.episode_id}_{ri}")
        new_rule = st.text_input("Add a rule (applies to every shot)", key=f"nr_{ep.episode_id}",
                                 placeholder="Her watch stays on her left wrist")
        picture = st.file_uploader("Reference picture of the character (optional)",
                                   type=["png", "jpg", "jpeg"])
        consent = st.checkbox("This picture is AI-generated, or shows an adult who gave permission.")
        saved = st.form_submit_button("Save", type="primary")
    if saved:
        def work():
            if picture is not None:
                validate_reference(picture.getvalue(), picture.name, consent)
                storage.save_bytes(generator.reference_path(ep.episode_id), picture.getvalue())
            c.character.name, c.character.appearance = name.strip(), appearance.strip()
            c.character.wardrobe = wardrobe.strip()
            c.location, c.time_of_day = location.strip(), time_of_day.strip()
            kept = []
            for pi in slots:
                if remove.get(pi) or not names[pi].strip():
                    continue
                kept.append(Prop(
                    name=names[pi].strip(), description=looks[pi].strip(),
                    state_by_shot={sid: v.strip() for (qi, sid), v in states.items()
                                   if qi == pi and v.strip()}))
            c.props = kept
            for ri in sorted((i for i, gone in drop_rule.items() if gone), reverse=True):
                learning.forget(c.rules[ri].text)
                del c.rules[ri]
            if new_rule.strip():
                learning.add_rule(c, LearnedRule(text=new_rule.strip(), source="manual"))
            apply_rules([s.spec for s in ep.shots], c)
            storage.save_episode(ep)

        if _run("Saving", work):
            st.session_state[f"edit_{ep.episode_id}"] = False
            st.rerun()


def _plan_step(ep) -> None:
    started = _started(ep)
    st.markdown("##### What must stay the same in every shot")
    editing_key = f"edit_{ep.episode_id}"
    with st.container(border=True):
        if st.session_state.get(editing_key):
            _details_form(ep)
            if st.button("Cancel"):
                st.session_state[editing_key] = False
                st.rerun()
        else:
            _canon_summary(ep)
            if st.button("Edit"):
                st.session_state[editing_key] = True
                st.rerun()
    if started:
        st.caption("After editing, open Shots and use 'Check the clips again'. It costs no video.")

    st.markdown("##### Shots")
    with st.container(border=True):
        for shot in ep.shots:
            s = shot.spec
            st.markdown(f"**{s.shot_id[1:]}.** {s.action}  \n:gray[{s.shot_type}, {s.camera_motion}]")
        if not started:
            with st.expander("Change the plan"):
                instruction = st.text_input("What should change",
                                            placeholder="make shot 2 a close-up of her hands")
                if st.button("Re-plan", disabled=not instruction.strip()):
                    picked = [b for b in ep.beats if b.beat_id in ep.picked_beat_ids]

                    def work():
                        specs = replan(picked, ep.canon, [s.spec for s in ep.shots], instruction)
                        ep.shots = [Shot(spec=s) for s in specs]
                        storage.save_episode(ep)

                    if _run("Re-planning", work):
                        st.rerun()

    _generate_panel(ep)


def _generate_panel(ep) -> None:
    todo = [s for s in ep.shots if s.status in ("PLANNED", "GENERATING", "SKIPPED")]
    if not todo:
        st.button("See the shots", type="primary", on_click=_go, args=(SHOTS,))
        return
    settings = get_settings()
    needed = sum(s.spec.duration_seconds for s in todo)
    left = ration.remaining()
    st.markdown("##### Make the video")
    with st.container(border=True):
        if not settings.veo_model:
            st.error("Setup is incomplete: VEO_MODEL must be set.")
            return
        st.write(f"{len(todo)} shots · about {needed} seconds of video · {left} seconds left in "
                 "the budget. Takes a minute or two per shot; keep this page open.")
        if not _unlock_box("gen"):
            return
        seed = None
        if not _started(ep):
            with st.expander("Demo option: plant a deliberate error"):
                st.caption("The first try of one shot is made wrong on purpose, so you can show the "
                           "critic catching it and the repair fixing it. It is labelled as a test.")
                ids = ["None"] + [s.spec.shot_id for s in ep.shots]
                seed_shot = st.selectbox("Shot", ids, format_func=lambda i: "None" if i == "None" else f"Shot {i[1:]}")
                seed_kind = st.radio("What to get wrong", ["prop", "wardrobe"], horizontal=True)
                seed_value = st.text_input("Wrong version", "open, with the lid off on the desk"
                                           if seed_kind == "prop" else "plain bright red kurta")
                if seed_shot != "None" and seed_value.strip():
                    seed = SeededError(shot_id=seed_shot, kind=seed_kind, value=seed_value.strip())
        if left < needed:
            st.warning(f"Only {left} seconds remain, so some shots may not be made.")
        label = "Continue making shots" if _started(ep) else f"Make {len(todo)} shots"
        if st.button(label, type="primary", disabled=left < ep.shots[0].spec.duration_seconds):
            if seed:
                ep.seeded_error = seed
            bar = st.progress(0.0, "Starting")
            try:
                pipeline.run_generation(ep, on_progress=lambda m, f: bar.progress(min(f, 1.0), m))
            except Exception as error:
                st.error(f"Stopped: {error}")
            _go(SHOTS)
            st.rerun()


# ---------- step 3: shots ----------

def _status_banner(ep) -> None:
    waiting = [s for s in ep.shots if s.status in ("FLAGGED", "SKIPPED", "PLANNED", "GENERATING")]
    kept = [s for s in ep.shots if s.status != "DROPPED"]
    if not waiting:
        st.success(f"All {len(kept)} shots are ready.")
        st.button("Go to Export", type="primary", on_click=_go, args=(EXPORT,))
    else:
        names = ", ".join(s.spec.shot_id[1:] for s in waiting)
        st.warning(f"{len(waiting)} of {len(ep.shots)} shots need your decision: shot {names}.")


def _storyboard(ep) -> str:
    """All shots side by side. Returns the id of the shot to show below."""
    key = f"sel_{ep.episode_id}"
    ids = [s.spec.shot_id for s in ep.shots]
    if st.session_state.get(key) not in ids:
        first_waiting = next((s.spec.shot_id for s in ep.shots if s.status == "FLAGGED"), ids[0])
        st.session_state[key] = first_waiting
    for col, shot in zip(st.columns(len(ep.shots)), ep.shots):
        sid = shot.spec.shot_id
        with col:
            attempt = chosen(shot)
            if attempt and attempt.clip_uri:
                play(attempt.clip_uri)
            else:
                st.caption("No clip")
            selected = st.session_state[key] == sid
            st.button(f"{ICON.get(shot.status, '')} Shot {sid[1:]}", key=f"pick_{sid}",
                      type="primary" if selected else "secondary", use_container_width=True,
                      on_click=lambda s=sid: st.session_state.update({key: s}))
    return st.session_state[key]


def _shot_panel(ep, shot) -> None:
    sid = shot.spec.shot_id

    def use(attempt_no):
        if _run("Saving", lambda: pipeline.use_attempt(ep, sid, attempt_no)):
            st.rerun()

    with st.container(border=True):
        st.caption(shot.spec.action)
        show_shot(shot, key=ep.episode_id, on_use=use, with_video=False)

        if shot.status == "DROPPED":
            if st.button("Restore this shot", key=f"restore_{sid}"):
                if _run("Saving", lambda: pipeline.restore(ep, sid)):
                    st.rerun()
            return
        if not shot.attempts:
            return

        needs_decision = shot.status == "FLAGGED"
        if not needs_decision and not st.toggle("Something wrong with this shot?", key=f"wrong_{sid}"):
            return
        suggestion, _ = pipeline.suggested_fix(shot)
        st.divider()
        fix = st.text_input("What should this shot show instead?", suggestion,
                            key=f"fix_{ep.episode_id}_{sid}_{len(shot.attempts)}",
                            placeholder="the note is already inside the box when the lid comes off",
                            help="Pre-filled with the critic's suggestion when it has one. "
                                 "Your instruction is also remembered as a rule.")
        unlocked = _unlock_box(f"fix_{sid}")
        cols = st.columns(3)
        if cols[0].button(f"Remake this shot ({shot.spec.duration_seconds}s)", key=f"fixgo_{sid}",
                          type="primary", disabled=not (unlocked and fix.strip())):
            if _run("Making and checking the shot. This takes a minute or two.",
                    lambda: pipeline.fix_shot(ep, sid, fix)):
                st.rerun()
        if needs_decision:
            if cols[1].button("Keep it as it is", key=f"ok_{sid}",
                              disabled=pipeline.latest_clip(shot) is None):
                if _run("Saving", lambda: pipeline.overrule(ep, sid)):
                    st.rerun()
        if cols[2].button("Leave this shot out", key=f"drop_{sid}"):
            if _run("Saving", lambda: pipeline.drop(ep, sid)):
                st.rerun()


def _shots_step(ep) -> None:
    if not _started(ep):
        st.info("No shots made yet. Go to Plan and press Make.")
        return
    _status_banner(ep)
    selected = _storyboard(ep)
    shot = next(s for s in ep.shots if s.spec.shot_id == selected)
    _shot_panel(ep, shot)
    with st.expander("More"):
        if st.button("Check the clips again (no new video)",
                     help="Runs the critic again with the current details and rules."):
            bar = st.progress(0.0, "Checking")
            try:
                pipeline.recheck(ep, on_progress=lambda m, f: bar.progress(min(f, 1.0), m))
            except Exception as error:
                st.error(f"Stopped: {error}")
            st.rerun()
        report_button(ep)
        if [s for s in ep.shots if s.status in ("PLANNED", "GENERATING", "SKIPPED")]:
            _generate_panel(ep)


# ---------- step 4: export ----------

def _export_step(ep) -> None:
    ok, reason = pipeline.can_export(ep)
    if not ok:
        st.info(reason)
        st.button("Back to Shots", on_click=_go, args=(SHOTS,))
        return
    show_final(ep)
    kept = [s for s in ep.shots if s.status != "DROPPED"]
    beats = {b.beat_id: b for b in ep.beats}
    with st.form("export"):
        st.markdown("##### Captions")
        st.caption("Shown on screen at the bottom of each shot. Leave one empty for none.")
        texts = {
            s.spec.beat_id: st.text_input(f"Shot {s.spec.shot_id[1:]}",
                                          beats[s.spec.beat_id].caption if s.spec.beat_id in beats else "",
                                          key=f"cap_{ep.episode_id}_{s.spec.shot_id}")
            for s in kept
        }
        label = "Export again" if ep.final_video_uri else "Export the video"
        go = st.form_submit_button(label, type="primary")
    if go:
        def work():
            for beat_id, text in texts.items():
                if beat_id in beats:
                    beats[beat_id].caption = text.strip()
            assembler.assemble(ep)

        if _run("Joining the shots into one video", work):
            st.rerun()
    report_button(ep)


# ---------- page ----------

def _allowed(ep) -> list[str]:
    if ep is None or ep.canon is None or not ep.shots:
        return [SCRIPT]
    if not _started(ep):
        return [SCRIPT, PLAN]
    return [SCRIPT, PLAN, SHOTS, EXPORT]


def _default(ep) -> str:
    allowed = _allowed(ep)
    if not _started(ep):
        return allowed[-1]
    return EXPORT if pipeline.can_export(ep)[0] and ep.final_video_uri else SHOTS


def render() -> None:
    ep = _current()
    head, new = st.columns([4, 1])
    head.header(ep.title if ep else "New episode")
    if ep is not None and new.button("New episode", use_container_width=True):
        st.session_state.pop("episode_id", None)
        st.session_state.pop("stage", None)
        st.rerun()

    allowed = _allowed(ep)
    if "stage_next" in st.session_state:
        st.session_state["stage"] = st.session_state.pop("stage_next")
    if st.session_state.get("stage") not in allowed:
        st.session_state["stage"] = _default(ep)
    stage = st.segmented_control("Step", allowed, key="stage", label_visibility="collapsed")
    if stage is None:   # clicking the selected step again clears it; stay on a sensible step
        stage = _default(ep)

    if stage == SCRIPT:
        _script_step(ep)
    elif stage == PLAN:
        _plan_step(ep)
    elif stage == SHOTS:
        _shots_step(ep)
    else:
        _export_step(ep)
