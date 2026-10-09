"""Shared display of results, used by the check screen and the saved list."""
import streamlit as st

from dramagraph import storage
from dramagraph.critic import verdicts
from dramagraph.models import Episode
from dramagraph.qa_report import build_report, report_to_markdown

BADGE = {"ACCEPT": "✅ Accept", "REGENERATE": "❌ Error found", "REVIEW": "⚠️ Needs your review"}
MARK = {"match": "✅", "mismatch": "❌", "unclear": "⚠️"}
LABEL = {"wardrobe": "Wardrobe", "prop_state": "Prop", "scene": "Scene",
         "cross_shot": "Matches previous clip"}


def show_episode(episode: Episode) -> None:
    for shot in episode.shots:
        attempt = shot.attempts[-1] if shot.attempts else None
        with st.container(border=True):
            left, right = st.columns([1, 2])
            with left:
                if attempt and attempt.clip_uri and storage.exists(attempt.clip_uri):
                    st.video(storage.load_bytes(attempt.clip_uri))
            with right:
                st.markdown(f"**Clip {shot.spec.shot_id[1:]}**")
                if attempt is None or attempt.critic is None:
                    st.error("Could not check this clip. " + ((attempt and attempt.error) or ""))
                    continue
                critic = attempt.critic
                st.markdown(BADGE.get(critic.decision, critic.decision or ""))
                for name, verdict in verdicts(critic).items():
                    observed = getattr(critic, name).observed
                    st.write(f"{MARK[verdict]} {LABEL[name]}: {observed}")
                for v in critic.violations:
                    st.warning(
                        f"{v.type} at {v.timestamp_seconds:.1f}s. "
                        f"Expected: {v.expected}. Observed: {v.observed}."
                    )
    report = episode.report or build_report(episode)
    st.download_button(
        "Download QA report",
        report_to_markdown(report),
        file_name=f"qa_report_{episode.episode_id}.md",
        key=f"dl_{episode.episode_id}",
    )
