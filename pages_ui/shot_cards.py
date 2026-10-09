"""Shared display of results, used by every screen."""
import streamlit as st

from dramagraph import storage
from dramagraph.critic import lines as critic_lines
from dramagraph.models import Attempt, Episode, Shot
from dramagraph.qa_report import build_report, report_to_markdown

BADGE = {"ACCEPT": "✅ Accept", "REGENERATE": "❌ Error found", "REVIEW": "⚠️ Needs your review"}
STATUS = {"ACCEPTED": "✅ Accepted", "FLAGGED": "⚠️ Needs your decision",
          "OVERRULED": "☑️ Accepted by you over the critic", "DROPPED": "🗑️ Dropped",
          "SKIPPED": "⏭️ Not generated yet", "PLANNED": "Planned", "GENERATING": "Generating"}
MARK = {"match": "✅", "mismatch": "❌", "unclear": "⚠️"}


def _clean(text: str) -> str:
    return text.strip().rstrip(".")


def show_attempt(attempt: Attempt) -> None:
    """One clip with the critic's findings beside it."""
    left, right = st.columns([1, 2])
    with left:
        if attempt.clip_uri and storage.exists(attempt.clip_uri):
            st.video(storage.load_bytes(attempt.clip_uri))
    with right:
        if attempt.seeded:
            st.caption("Seeded test: this clip was asked for with a deliberate error.")
        if attempt.critic is None:
            st.error("No result for this clip. " + (attempt.error or ""))
            return
        critic = attempt.critic
        st.markdown(BADGE.get(critic.decision, critic.decision or ""))
        for row in critic_lines(critic):
            st.write(f"{MARK[row['verdict']]} {row['label']}: {row['observed']}")
        for v in critic.violations:
            st.warning(f"{v.type} at {v.timestamp_seconds:.1f}s. Expected: {_clean(v.expected)}. "
                       f"Observed: {_clean(v.observed)}.")
        show_observations(critic.observations)


def describe_observation(o) -> str:
    text = f"**{'Major' if o.severity == 'major' else 'Minor'}** at {o.timestamp_seconds:.1f}s: {_clean(o.what)}."
    if o.suggested_fix.strip():
        text += f"  \nSuggested fix: {_clean(o.suggested_fix)}."
    return text


def show_observations(observations, skip_major: bool = False) -> None:
    """Advisory findings. Minor ones are folded away so they do not bury the important ones."""
    major = [o for o in observations if o.severity == "major"]
    minor = [o for o in observations if o.severity != "major"]
    if not skip_major:
        for o in major:
            st.info("Also noticed (advisory). " + describe_observation(o))
    if minor:
        with st.expander(f"{len(minor)} minor thing(s) also noticed"):
            for o in minor:
                st.write(describe_observation(o))


def chosen(shot: Shot):
    wanted = shot.chosen_attempt
    found = next((a for a in shot.attempts if a.attempt_no == wanted), None)
    return found or (shot.attempts[-1] if shot.attempts else None)


def show_shot(shot: Shot) -> None:
    """A shot's final clip, plus the first failed clip if it was repaired."""
    final = chosen(shot)
    st.markdown(f"**Shot {shot.spec.shot_id[1:]}** · {STATUS.get(shot.status, shot.status)}")
    if final is None:
        st.info("No clip yet.")
        return
    failed_first = next(
        (a for a in shot.attempts
         if a.clip_uri and a.critic and a.critic.decision == "REGENERATE"
         and a.attempt_no < final.attempt_no), None)
    if failed_first:
        st.caption("Before: the clip the critic rejected")
        show_attempt(failed_first)
        st.caption(f"After: attempt {final.attempt_no}")
    show_attempt(final)
    errors = [a.error for a in shot.attempts if a.error and not a.clip_uri]
    if errors:
        st.caption("Generation problems: " + " | ".join(errors[-2:]))


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
            show_shot(shot)
    report_button(episode)
