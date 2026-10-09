"""Shared display of results, used by every screen.

A shot card shows a summary first: the clip, one status line, and anything that
needs a decision. Every check, every minor finding and every attempt is one click away.
"""
import streamlit as st

from dramagraph import storage
from dramagraph.critic import lines as critic_lines
from dramagraph.models import Attempt, Episode, Shot
from dramagraph.qa_report import build_report, report_to_markdown

BADGE = {"ACCEPT": "✅ Passed", "REGENERATE": "❌ Error found", "REVIEW": "⚠️ Unclear"}
STATUS = {"ACCEPTED": "✅ Accepted", "FLAGGED": "⚠️ Needs your decision",
          "OVERRULED": "☑️ Chosen by you over the critic", "DROPPED": "🗑️ Dropped",
          "SKIPPED": "⏭️ Not generated yet", "PLANNED": "Planned", "GENERATING": "Generating"}
MARK = {"match": "✅", "mismatch": "❌", "unclear": "⚠️"}


def _clean(text: str) -> str:
    return text.strip().rstrip(".")


@st.cache_data(max_entries=48, show_spinner=False)
def clip_bytes(uri: str) -> bytes:
    """Clips never change once saved, so each is fetched from storage only once."""
    return storage.load_bytes(uri)


def play(uri) -> None:
    if uri and storage.exists(uri):
        st.video(clip_bytes(uri))


def describe_observation(o) -> str:
    text = f"**{'Major' if o.severity == 'major' else 'Minor'}** at {o.timestamp_seconds:.1f}s: {_clean(o.what)}."
    if o.suggested_fix.strip():
        text += f"  \nSuggested fix: {_clean(o.suggested_fix)}."
    return text


def chosen(shot: Shot):
    found = next((a for a in shot.attempts if a.attempt_no == shot.chosen_attempt), None)
    return found or next((a for a in reversed(shot.attempts) if a.clip_uri), None)


def summary_line(shot: Shot) -> str:
    """One sentence that says where the shot stands."""
    attempt = chosen(shot)
    made = sum(1 for a in shot.attempts if a.clip_uri)
    if attempt is None or attempt.critic is None:
        return "No checked clip yet."
    rows = critic_lines(attempt.critic)
    passed = sum(1 for r in rows if r["verdict"] == "match")
    text = f"{passed} of {len(rows)} checks passed"
    if made > 1:
        text += f" · clip {attempt.attempt_no} of {made} made"
    if any(a.seeded for a in shot.attempts):
        text += " · includes a seeded test error"
    return text


def show_summary(shot: Shot) -> None:
    """The clip in use, the status, and only what needs attention."""
    attempt = chosen(shot)
    st.markdown(f"**Shot {shot.spec.shot_id[1:]}** · {STATUS.get(shot.status, shot.status)}")
    left, right = st.columns([1, 2])
    with left:
        play(attempt.clip_uri if attempt else None)
    with right:
        st.write(summary_line(shot))
        if attempt is None:
            failures = [a.error for a in shot.attempts if a.error]
            st.error("No clip could be made. " + (failures[-1] if failures else ""))
            return
        if attempt.critic is None:
            st.error("This clip could not be checked. " + (attempt.error or ""))
            return
        critic = attempt.critic
        for row in critic_lines(critic):
            if row["verdict"] != "match":
                st.write(f"{MARK[row['verdict']]} {row['label']}: {row['observed']}")
        for v in critic.violations:
            st.warning(f"At {v.timestamp_seconds:.1f}s. Expected: {_clean(v.expected)}. "
                       f"Observed: {_clean(v.observed)}.")
        for o in critic.observations:
            if o.severity == "major":
                st.info("The critic also noticed (advisory). " + describe_observation(o))


def show_details(shot: Shot) -> None:
    """Every check and every minor finding for the clip in use."""
    attempt = chosen(shot)
    if attempt is None or attempt.critic is None:
        st.write("Nothing to show yet.")
        return
    st.caption(f"What the critic saw: {attempt.critic.seen}")
    for row in critic_lines(attempt.critic):
        st.write(f"{MARK[row['verdict']]} {row['label']}: {row['observed']}")
    for o in attempt.critic.observations:
        if o.severity != "major":
            st.write(describe_observation(o))
    failures = [a.error for a in shot.attempts if a.error and not a.clip_uri]
    if failures:
        st.caption("Generation problems: " + " | ".join(failures[-2:]))


def attempt_label(shot: Shot, attempt: Attempt) -> str:
    kind = {"INITIAL": "first try", "AUTO_REPAIR": "automatic repair",
            "CUSTOM_REPAIR": "your fix", "UPLOADED": "uploaded"}.get(attempt.kind, attempt.kind)
    notes = [kind]
    if attempt.seeded:
        notes.append("seeded test error")
    if attempt.rejected:
        notes.append("set aside by you")
    if attempt.attempt_no == shot.chosen_attempt:
        notes.append("in use")
    verdict = BADGE.get(attempt.critic.decision, "") if attempt.critic else "not checked"
    return f"Clip {attempt.attempt_no} ({', '.join(notes)}) · {verdict}"


def show_attempts(shot: Shot, on_use=None) -> None:
    """Every clip made for the shot, in order. on_use(attempt_no) lets the creator pick one."""
    clips = [a for a in shot.attempts if a.clip_uri]
    for row in range(0, len(clips), 3):
        for col, attempt in zip(st.columns(3), clips[row:row + 3]):
            with col:
                st.caption(attempt_label(shot, attempt))
                play(attempt.clip_uri)
                if attempt.critic:
                    for r in critic_lines(attempt.critic):
                        if r["verdict"] != "match":
                            st.caption(f"{MARK[r['verdict']]} {r['label']}: {r['observed']}")
                if on_use and attempt.attempt_no != shot.chosen_attempt:
                    if st.button("Use this clip", key=f"use_{shot.spec.shot_id}_{attempt.attempt_no}"):
                        on_use(attempt.attempt_no)


def show_shot(shot: Shot, key: str = "", on_use=None) -> None:
    """Summary, with details and all attempts behind switches so they load only when asked."""
    show_summary(shot)
    made = sum(1 for a in shot.attempts if a.clip_uri)
    sid = shot.spec.shot_id
    one, two = st.columns(2)
    if one.toggle("All checks and minor findings", key=f"det_{key}_{sid}"):
        show_details(shot)
    if made > 1 and two.toggle(f"All {made} clips made for this shot", key=f"att_{key}_{sid}"):
        show_attempts(shot, on_use)


def report_button(episode: Episode) -> None:
    report = episode.report or build_report(episode)
    st.download_button(
        "Download QA report", report_to_markdown(report),
        file_name=f"qa_report_{episode.episode_id}.md", key=f"dl_{episode.episode_id}",
    )


def show_final(episode: Episode) -> None:
    """The stitched video, if it has been exported."""
    uri = episode.final_video_uri
    if not uri or not storage.exists(uri):
        return
    data = storage.load_bytes(uri)
    left, _ = st.columns([1, 2])
    with left:
        st.video(data)
        st.download_button("Download video", data, file_name=f"{episode.episode_id}.mp4",
                           mime="video/mp4", key=f"mp4_{episode.episode_id}")


def show_episode(episode: Episode) -> None:
    show_final(episode)
    for shot in episode.shots:
        with st.container(border=True):
            show_shot(shot, key=episode.episode_id)
    report_button(episode)
