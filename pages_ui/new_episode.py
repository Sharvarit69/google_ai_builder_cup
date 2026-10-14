"""Make an episode, in four steps shown one at a time: Script, Plan, Shots, Export."""
import hmac
import uuid
from datetime import datetime, timezone

import streamlit as st

from dramagraph import assembler, learning, pipeline, ration, storage
from dramagraph.canon import MAX_PROPS, build_canon
from dramagraph.config import get_settings
from dramagraph.models import Episode, LearnedRule, Prop, SeededError, Shot
from dramagraph.parser import parse_script
from dramagraph.planner import MAX_SHOTS, apply_rules, plan_shots, replan
from pages_ui.shot_cards import chosen, play, report_button, show_attempts, show_final

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

RULE_SOURCE = {"creator_catch": "From your fix", "observation": "From a critic suggestion",
               "manual": "Typed by you", "library": "From an earlier episode"}


def _save(ep, label: str = "Saving", work=None) -> None:
    """Re-apply the continuity requirements, save, and redraw."""
    def go():
        if work:
            work()
        apply_rules([s.spec for s in ep.shots], ep.canon)
        storage.save_episode(ep)

    if _run(label, go):
        st.rerun()


def _card_header(title: str, edit_key: str | None = None, extra=None):
    left, right = st.columns([5, 1])
    left.markdown(f"#### {title}")
    if edit_key is not None:
        editing = st.session_state.get(edit_key, False)
        if right.button("Cancel" if editing else "Edit", key=f"btn_{edit_key}", use_container_width=True):
            st.session_state[edit_key] = not editing
            st.rerun()
    elif extra:
        with right:
            extra()


def _picture_strip(ep, owner: str, pictures_list, consent_needed: bool) -> None:
    """Thumbnails with labels, a remove button each, and an Add picture box."""
    from dramagraph import pictures as pics

    shown = list(pictures_list)
    if owner == "character" and not shown:
        shown = pics.character_pictures(ep)
    cols = st.columns(pics.MAX_PER_ITEM)
    for col, picture in zip(cols, shown):
        with col:
            data = pics.load(picture)
            if data:
                st.image(data, use_container_width=True)
            st.caption(picture.label or "no label")
            if picture in pictures_list and st.button("✕", key=f"rmpic_{picture.uri}",
                                                      help="Remove this picture"):
                _save(ep, work=lambda u=picture.uri: pics.remove_picture(ep, owner, u))
    if len(pictures_list) >= pics.MAX_PER_ITEM:
        return
    with st.popover("Add picture", use_container_width=False):
        upload = st.file_uploader("Picture (PNG or JPG)", type=["png", "jpg", "jpeg"],
                                  key=f"up_{ep.episode_id}_{owner}")
        hint = "front, side, back" if owner == "character" else "a state, e.g. closed or open"
        label = st.text_input(f"Label ({hint})", key=f"lab_{ep.episode_id}_{owner}")
        consent = True
        if consent_needed:
            consent = st.checkbox("AI-generated, or an adult who gave permission",
                                  key=f"ok_{ep.episode_id}_{owner}")
        if st.button("Add", key=f"addpic_{ep.episode_id}_{owner}", disabled=upload is None):
            _save(ep, "Saving the picture", lambda: pics.add_picture(
                ep, owner, upload.getvalue(), upload.name, label, consent))


def _character_card(ep) -> None:
    c = ep.canon.character
    key = f"edit_char_{ep.episode_id}"
    with st.container(border=True):
        _card_header("👤 Character", key)
        if st.session_state.get(key):
            with st.form(f"char_{ep.episode_id}"):
                name = st.text_input("Name", c.name)
                appearance = st.text_area("Appearance", c.appearance, height=80)
                wardrobe = st.text_area("Wardrobe", c.wardrobe, height=80)
                if st.form_submit_button("Save", type="primary"):
                    def work():
                        c.name, c.appearance, c.wardrobe = name.strip(), appearance.strip(), wardrobe.strip()
                    st.session_state[key] = False
                    _save(ep, work=work)
            return
        pics_col, text = st.columns([2, 3])
        with pics_col:
            _picture_strip(ep, "character", c.pictures, consent_needed=True)
        text.markdown(f"**Name**  \n{c.name}")
        text.markdown(f"**Appearance**  \n{c.appearance}")
        text.markdown(f"**Wardrobe**  \n{c.wardrobe}")


def _setting_card(ep) -> None:
    c = ep.canon
    key = f"edit_set_{ep.episode_id}"
    with st.container(border=True):
        _card_header("🏢 Setting", key)
        if st.session_state.get(key):
            with st.form(f"set_{ep.episode_id}"):
                location = st.text_area("Location", c.location, height=80)
                time_of_day = st.text_input("Time of day", c.time_of_day)
                if st.form_submit_button("Save", type="primary"):
                    def work():
                        c.location, c.time_of_day = location.strip(), time_of_day.strip()
                    st.session_state[key] = False
                    _save(ep, work=work)
            return
        st.markdown(f"**Location**  \n{c.location}")
        st.markdown(f"**Time of day**  \n{c.time_of_day}")


def _props_card(ep) -> None:
    c = ep.canon
    shot_ids = [s.spec.shot_id for s in ep.shots]

    def add_prop():
        with st.popover("Add a prop", use_container_width=True, disabled=len(c.props) >= MAX_PROPS):
            name = st.text_input("Name", key=f"np_{ep.episode_id}")
            look = st.text_input("What it looks like (shape, parts, material, colour)",
                                 key=f"nl_{ep.episode_id}")
            if st.button("Add", key=f"npgo_{ep.episode_id}", disabled=not name.strip()):
                _save(ep, work=lambda: c.props.append(Prop(name=name.strip(), description=look.strip())))

    with st.container(border=True):
        _card_header(f"🍱 Props · {len(c.props)} of {MAX_PROPS}", extra=add_prop)
        st.caption("Objects the story depends on. The critic checks what each looks like, "
                   "and its state at the end of every shot.")
        if not c.props:
            st.write("No props are being checked.")
        for pi, prop in enumerate(c.props):
            if pi:
                st.divider()
            key = f"edit_prop_{ep.episode_id}_{pi}"
            pics_col, info = st.columns([2, 5])
            with pics_col:
                _picture_strip(ep, prop.name, prop.pictures, consent_needed=False)
            with info:
                if st.session_state.get(key):
                    with st.form(f"prop_{ep.episode_id}_{pi}"):
                        name = st.text_input("Name", prop.name)
                        look = st.text_input("What it looks like", prop.description)
                        a, b = st.columns(2)
                        save = a.form_submit_button("Save", type="primary")
                        cancel = b.form_submit_button("Cancel")
                    if cancel:
                        st.session_state[key] = False
                        st.rerun()
                    if save:
                        def work(p=prop, n=name, l=look):
                            p.name, p.description = n.strip() or p.name, l.strip()
                        st.session_state[key] = False
                        _save(ep, work=work)
                else:
                    t, e, x = st.columns([5, 1, 1])
                    t.markdown(f"**{prop.name}**  \n{prop.description or ':gray[No description yet]'}")
                    if e.button("Edit", key=f"pe_{key}", use_container_width=True):
                        st.session_state[key] = True
                        st.rerun()
                    if x.button("Remove", key=f"px_{key}", use_container_width=True):
                        _save(ep, work=lambda i=pi: c.props.pop(i))
                row = {"State at the end of": prop.name}
                row.update({f"Shot {sid[1:]}": prop.state_by_shot.get(sid, "") for sid in shot_ids})
                edited = st.data_editor([row], hide_index=True, use_container_width=True,
                                        disabled=["State at the end of"], key=f"states_{ep.episode_id}_{pi}")
                new = {sid: str(edited[0].get(f"Shot {sid[1:]}") or "").strip() for sid in shot_ids}
                if new != {sid: prop.state_by_shot.get(sid, "") for sid in shot_ids}:
                    def work(p=prop, n=new):
                        p.state_by_shot = {sid: v for sid, v in n.items() if v}
                    _save(ep, work=work)
        if c.props:
            st.caption("Click a state to change it. It saves as soon as you leave the cell.")


def _rules_card(ep) -> None:
    c = ep.canon
    with st.container(border=True):
        _card_header(f"📏 Rules learned from you · {len(c.rules)}")
        st.caption("Checks that came from you, not from the built-in list. They are added when you "
                   "fix a shot in your own words, or typed below. General ones carry over to new episodes.")
        if not c.rules:
            st.write(":gray[No rules yet.]")
        for ri, rule in enumerate(c.rules):
            scope = "Every shot" if not rule.shot_ids else "Shot " + ", ".join(i[1:] for i in rule.shot_ids)
            a, b, d, x = st.columns([6, 2, 1.4, 0.7])
            a.write(rule.text)
            b.caption(RULE_SOURCE.get(rule.source, ""))
            d.caption(scope)
            if x.button("✕", key=f"rmrule_{ep.episode_id}_{ri}", help="Remove this rule"):
                def work(i=ri):
                    learning.forget(c.rules[i].text)
                    del c.rules[i]
                _save(ep, work=work)
        a, b = st.columns([5, 1])
        text = a.text_input("Add a rule", key=f"newrule_{ep.episode_id}_{len(c.rules)}",
                            placeholder="for example: her watch stays on her left wrist",
                            label_visibility="collapsed")
        if b.button("Add rule", key=f"addrule_{ep.episode_id}", use_container_width=True,
                    disabled=not text.strip()):
            _save(ep, work=lambda: learning.add_rule(c, LearnedRule(text=text.strip(), source="manual")))


def _shots_card(ep) -> None:
    started = _started(ep)
    seconds = ep.shots[0].spec.duration_seconds if ep.shots else 0
    with st.container(border=True):
        _card_header(f"🎬 Shots · {len(ep.shots)} × {seconds} seconds")
        st.caption("Edit any line directly." + ("" if started else
                   " Or describe a bigger change below and let Gemini re-plan.")
                   + (" Changes apply to the next version made of that shot." if started else ""))
        rows = [{"Shot": int(s.spec.shot_id[1:]), "What happens": s.spec.action,
                 "Camera": ", ".join(x for x in (s.spec.shot_type, s.spec.camera_motion) if x)}
                for s in ep.shots]
        edited = st.data_editor(rows, hide_index=True, use_container_width=True, disabled=["Shot"],
                                key=f"shots_{ep.episode_id}")
        changes = {f"S{r['Shot']}": (str(r["What happens"] or ""), str(r["Camera"] or ""))
                   for r, old in zip(edited, rows)
                   if (r["What happens"], r["Camera"]) != (old["What happens"], old["Camera"])}
        if changes:
            _save(ep, "Saving the shots", lambda: pipeline.edit_shots(ep, changes))
        if not started:
            a, b = st.columns([5, 1])
            instruction = a.text_input("Re-plan", label_visibility="collapsed",
                                       placeholder="Or describe a bigger change, e.g. make shot 2 a close-up of her hands")
            if b.button("Re-plan", use_container_width=True, disabled=not instruction.strip()):
                picked = [x for x in ep.beats if x.beat_id in ep.picked_beat_ids]

                def work():
                    specs = replan(picked, ep.canon, [s.spec for s in ep.shots], instruction)
                    ep.shots = [Shot(spec=s) for s in specs]
                _save(ep, "Re-planning", work)


def _plan_step(ep) -> None:
    st.caption("Gemini filled these in from your script. Everything here is what the critic "
               "checks in every clip, so edit anything that is not right.")
    left, right = st.columns([3, 2])
    with left:
        _character_card(ep)
    with right:
        _setting_card(ep)
    _props_card(ep)
    _rules_card(ep)
    _shots_card(ep)
    if _started(ep):
        st.caption("After changing details or rules, open Shots and use 'Check the clips again'. "
                   "It costs no video.")
    _generate_panel(ep)


def _generate_panel(ep) -> None:
    todo = [s for s in ep.shots if s.status in ("PLANNED", "GENERATING", "SKIPPED")]
    if not todo:
        st.button("See the shots", type="primary", on_click=_go, args=(SHOTS,))
        return
    settings = get_settings()
    needed = sum(s.spec.duration_seconds for s in todo)
    left = ration.remaining()
    with st.container(border=True):
        if not settings.veo_model:
            st.error("Setup is incomplete: VEO_MODEL must be set.")
            return
        st.write(f"**Make the video** · {len(todo)} shots · about {needed} seconds · {left} seconds "
                 "left in the budget. Each shot takes a minute or two; keep this page open.")
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
        label = "Continue making shots" if _started(ep) else f"Make {len(todo)} shots · {needed} seconds of video"
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

SHORT_STATUS = {"ACCEPTED": "Accepted", "FLAGGED": "Needs decision", "OVERRULED": "Kept by you",
                "DROPPED": "Left out", "SKIPPED": "Not made", "PLANNED": "Not made", "GENERATING": "Making"}
TAG_CSS = """<style>
.dg-tag{display:inline-block;padding:1px 9px;border-radius:10px;font-size:0.78rem;font-weight:600;margin-right:6px}
.dg-major{background:#fde2e1;color:#b42318}.dg-minor{background:#fef3c7;color:#92400e}
.dg-time{background:#eef2f6;color:#334155}.dg-rule{background:#e0e7ff;color:#3730a3}
</style>"""


def _status_banner(ep) -> None:
    waiting = [s for s in ep.shots if s.status in ("FLAGGED", "SKIPPED", "PLANNED", "GENERATING")]
    kept = [s for s in ep.shots if s.status != "DROPPED"]
    if not waiting:
        a, b = st.columns([4, 1])
        a.success(f"All {len(kept)} shots are ready.")
        b.button("Go to Export", type="primary", on_click=_go, args=(EXPORT,), use_container_width=True)
    else:
        names = ", ".join(s.spec.shot_id[1:] for s in waiting)
        ready = len(ep.shots) - len(waiting)
        st.warning(f"{ready} of {len(ep.shots)} shots are ready. **Shot {names} "
                   f"need{'s' if len(waiting) == 1 else ''} your decision.**")


def _short_status(shot) -> str:
    fixed = shot.status == "ACCEPTED" and any(
        a.critic and a.critic.decision == "REGENERATE" for a in shot.attempts)
    return "Fixed" if fixed else SHORT_STATUS.get(shot.status, shot.status)


def _storyboard(ep) -> str:
    from dramagraph import problems

    key = f"sel_{ep.episode_id}"
    ids = [s.spec.shot_id for s in ep.shots]
    if st.session_state.get(key) not in ids:
        st.session_state[key] = next((s.spec.shot_id for s in ep.shots if s.status == "FLAGGED"), ids[0])
    for col, shot in zip(st.columns(len(ep.shots)), ep.shots):
        sid = shot.spec.shot_id
        with col:
            attempt = chosen(shot)
            if attempt and attempt.clip_uri:
                play(attempt.clip_uri, muted=True)
            else:
                st.caption("No clip")
            st.button(f"{ICON.get(shot.status, '')} Shot {sid[1:]} · {_short_status(shot)}",
                      key=f"pick_{sid}", use_container_width=True,
                      type="primary" if st.session_state[key] == sid else "secondary",
                      on_click=lambda s=sid: st.session_state.update({key: s}))
            st.caption(problems.one_line(shot))
    return st.session_state[key]


def _fix_key(ep, shot) -> str:
    return f"fix_{ep.episode_id}_{shot.spec.shot_id}_{len(shot.attempts)}"


def _include_key(ep, shot, i: int) -> str:
    return f"inc_{ep.episode_id}_{shot.spec.shot_id}_{len(shot.attempts)}_{i}"


def _ticked(ep, shot) -> list:
    """The problems whose 'Include in remake' box is ticked (majors start ticked)."""
    from dramagraph import problems

    found = problems.for_shot(shot)
    return [p for i, p in enumerate(found)
            if p.fix and st.session_state.get(_include_key(ep, shot, i), p.severity == "major")]


def _problems_tab(ep, shot) -> None:
    from dramagraph import problems

    found = problems.for_shot(shot)
    attempt = chosen(shot)
    if attempt is None or attempt.critic is None:
        st.write("This clip has not been checked yet. Use 'Check the clips again' under More.")
        return
    if not found:
        st.write("No problems found. Every check passed.")
        return
    st.markdown(TAG_CSS, unsafe_allow_html=True)
    sid = shot.spec.shot_id
    for i, p in enumerate(found):
        with st.container(border=True):
            when = f"<span class='dg-tag dg-time'>at 0:{int(p.seconds):02d}</span>" if p.seconds else ""
            mine = "<span class='dg-tag dg-rule'>YOUR RULE</span>" if p.source == "rule" else ""
            st.markdown(f"<span class='dg-tag dg-{p.severity}'>{p.severity.upper()}</span>{mine}{when}"
                        f"&nbsp;**{p.headline}**", unsafe_allow_html=True)
            if p.detail and p.detail.rstrip(".").lower() != p.headline.rstrip(".").lower():
                st.write(p.detail)
            if p.fix:
                label = "Should be" if p.source == "check" else "Suggested fix"
                st.markdown(f"**{label}:** {p.fix}")
            a, b, c, _ = st.columns([1.2, 1.5, 1.4, 0.6])
            a.button(f"▶ Play from 0:{int(p.seconds):02d}", key=f"play_{sid}_{i}", use_container_width=True,
                     on_click=lambda t=p.seconds: st.session_state.update({f"t_{sid}": t}))
            if p.fix:
                b.checkbox("Include in remake", key=_include_key(ep, shot, i), value=p.severity == "major")
            if p.source == "check":
                c.caption("Already checked every time")
            elif p.source == "rule" or (p.fix and learning.has_rule(ep.canon, p.fix)):
                c.button("✓ Rule saved", key=f"saved_{sid}_{i}", disabled=True, use_container_width=True)
            elif p.fix:
                if c.button("Always check this", key=f"rule_{sid}_{i}", use_container_width=True,
                            help="Saves the suggested fix as a rule and checks every clip against it "
                                 "straight away. Makes no new video."):
                    bar = st.progress(0.0, "Saving the rule and checking the clips")

                    def work(f=p.fix):
                        pipeline.save_rule_and_recheck(
                            ep, sid, f, on_progress=lambda m, x: bar.progress(min(x, 1.0), m))
                    rules_before = len(ep.canon.rules)
                    ok = _run("Saving the rule", work)
                    _reset_fix_boxes(ep)
                    if len(ep.canon.rules) > rules_before:
                        _announce_rule(ep, p.fix)
                    if ok:
                        st.rerun()


def _reset_fix_boxes(ep) -> None:
    """After a re-check the suggestions change, so refill the fix boxes."""
    for key in [k for k in st.session_state if str(k).startswith(f"fix_{ep.episode_id}_")]:
        del st.session_state[key]


def _announce_rule(ep, note: str) -> None:
    """Remember to show a confirmation after the page redraws."""
    latest = ep.canon.rules[-1].text if ep.canon and ep.canon.rules else note
    st.session_state["toast"] = f"Rule added: {latest}  ·  see Plan › Rules"


def _checks_tab(shot) -> None:
    from dramagraph.critic import lines as critic_lines

    attempt = chosen(shot)
    if attempt is None or attempt.critic is None:
        st.write("Nothing checked yet.")
        return
    mark = {"match": "✅", "mismatch": "❌", "unclear": "⚠️"}
    for row in critic_lines(attempt.critic):
        st.write(f"{mark[row['verdict']]} **{row['label']}** · {row['observed']}")
    if attempt.critic.seen:
        st.caption(f"What the critic saw: {attempt.critic.seen}")


def _pictures_tab(shot) -> None:
    attempt = chosen(shot)
    used = attempt.references_used if attempt else []
    if used:
        st.write("Pictures sent to Veo for this version:")
        for name in used:
            st.write(f"• {name}")
    else:
        st.write("No reference pictures were used for this version. Add pictures in Plan.")


def _shot_panel(ep, shot) -> None:
    from dramagraph import problems

    sid = shot.spec.shot_id
    attempt = chosen(shot)
    made = [a for a in shot.attempts if a.clip_uri]

    def use(attempt_no):
        if _run("Saving", lambda: pipeline.use_attempt(ep, sid, attempt_no)):
            st.rerun()

    with st.container(border=True):
        left, right = st.columns([1, 2])
        with left:
            if attempt and attempt.clip_uri:
                play(attempt.clip_uri, start=st.session_state.get(f"t_{sid}", 0))
                st.caption(f"Version {made.index(attempt) + 1 if attempt in made else 1} of {len(made)} · "
                           f"{shot.spec.duration_seconds} seconds")
            else:
                failures = [a.error for a in shot.attempts if a.error]
                st.error("No clip could be made. " + (failures[-1] if failures else ""))
        with right:
            st.markdown(f"### Shot {sid[1:]} · {_short_status(shot)}")
            st.caption(f"Should show: {shot.spec.action}")
            count = len(problems.for_shot(shot))
            tabs = st.tabs([f"Problems · {count}", "All checks", f"Versions · {len(made)}", "Pictures used"])
            with tabs[0]:
                _problems_tab(ep, shot)
            with tabs[1]:
                _checks_tab(shot)
            with tabs[2]:
                if len(made) > 1:
                    show_attempts(shot, on_use=use)
                else:
                    st.write("Only one version has been made of this shot.")
            with tabs[3]:
                _pictures_tab(shot)


def _decision(ep, shot) -> None:
    sid = shot.spec.shot_id
    if shot.status == "DROPPED":
        if st.button("Put this shot back", key=f"restore_{sid}"):
            if _run("Saving", lambda: pipeline.restore(ep, sid)):
                st.rerun()
        return
    if not shot.attempts:
        return
    needs = shot.status == "FLAGGED"
    if needs:
        st.markdown(f"##### What do you want to do with shot {sid[1:]}?")
    elif not st.toggle("Something wrong with this shot?", key=f"wrong_{sid}"):
        return
    ticked = _ticked(ep, shot)
    joined = " ".join(p.fix for p in ticked)
    key, last = _fix_key(ep, shot), f"joined_{_fix_key(ep, shot)}"
    if key not in st.session_state or st.session_state.get(last) != joined:
        st.session_state[key] = joined          # the ticks changed, so rebuild the text
        st.session_state[last] = joined
    if ticked:
        st.write(f"Remake shot {sid[1:]} to fix **{len(ticked)} problem{'s' if len(ticked) > 1 else ''}**: "
                 + "; ".join(p.headline for p in ticked))
    else:
        st.write("Tick the problems above to include their fixes, or type below what the shot should show.")
    fix = st.text_input("What the remake will ask for (edit freely)", key=key,
                        placeholder="the note is already inside the box when the lid comes off")
    edited = fix.strip() != joined.strip()
    # Only fixes that go beyond the built-in checks are worth keeping as rules.
    rule_note = fix.strip() if edited else " ".join(p.fix for p in ticked if p.source != "check")
    already = bool(rule_note) and learning.has_rule(ep.canon, rule_note)
    if fix.strip() and not rule_note:
        save_rule = st.checkbox("Also save this as a rule (all shots, future episodes)", value=False,
                                disabled=True, key=f"sr_{key}")
        st.caption("Not saved as a rule: these fixes are already checked on every clip.")
    elif already:
        save_rule = False
        st.caption("✓ This is already one of your rules.")
    else:
        save_rule = st.checkbox("Also save this as a rule (all shots, future episodes)", value=True,
                                key=f"sr_{key}", disabled=not fix.strip())
    unlocked = _unlock_box(f"fix_{sid}")
    a, b, c, _ = st.columns([1.4, 1.2, 1.2, 2])
    if a.button(f"Remake shot {sid[1:]} · {shot.spec.duration_seconds}s", key=f"fixgo_{sid}",
                type="primary", use_container_width=True, disabled=not (unlocked and fix.strip())):
        rules_before = len(ep.canon.rules)
        if _run("Making and checking the shot. This takes a minute or two.",
                lambda: pipeline.fix_shot(ep, sid, fix, save_rule=save_rule, rule_note=rule_note)):
            if len(ep.canon.rules) > rules_before:
                _announce_rule(ep, fix)
            st.rerun()
    if needs and b.button("Keep it as it is", key=f"ok_{sid}", use_container_width=True,
                          disabled=pipeline.latest_clip(shot) is None):
        if _run("Saving", lambda: pipeline.overrule(ep, sid)):
            st.rerun()
    if c.button("Leave it out", key=f"drop_{sid}", use_container_width=True):
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
    _decision(ep, shot)
    with st.expander("More"):
        if st.button("Check the clips again (no new video)",
                     help="Runs the critic again with the current details, pictures and rules."):
            bar = st.progress(0.0, "Checking")
            try:
                pipeline.recheck(ep, on_progress=lambda m, f: bar.progress(min(f, 1.0), m))
                _reset_fix_boxes(ep)
                st.rerun()
            except Exception as error:
                st.error(str(error))
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
    if "toast" in st.session_state:
        st.toast(st.session_state.pop("toast"), icon="📏")
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
