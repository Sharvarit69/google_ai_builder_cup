"""Check-only mode: run the critic over clips the user already has."""
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from . import storage
from .critic import check_clip
from .models import Attempt, Canon, Character, Episode, Prop, Shot, ShotSpec
from .qa_report import build_report

MAX_CLIPS = 4


def make_canon(name, appearance, wardrobe, prop_name, prop_states, location, time_of_day,
               prop_description=""):
    """prop_states is one expected state per clip, in clip order."""
    props = []
    if prop_name.strip():
        props.append(
            Prop(
                name=prop_name.strip(),
                description=prop_description.strip(),
                state_by_shot={
                    f"S{i + 1}": s.strip() for i, s in enumerate(prop_states) if s.strip()
                },
            )
        )
    return Canon(
        character=Character(
            name=name.strip() or "The character",
            appearance=appearance.strip(),
            wardrobe=wardrobe.strip(),
        ),
        props=props,
        location=location.strip(),
        time_of_day=time_of_day.strip(),
    )


def run_check_only(
    title: str,
    canon: Canon,
    clips: list[bytes],
    on_progress: Optional[Callable[[str, float], None]] = None,
    checker=check_clip,
) -> Episode:
    """Save the clips, check each against the canon and the clip before it."""
    if not clips:
        raise ValueError("Add at least one clip.")
    if len(clips) > MAX_CLIPS:
        raise ValueError(f"Check-only mode takes at most {MAX_CLIPS} clips.")

    episode = Episode(
        episode_id=uuid.uuid4().hex[:12],
        title=title.strip() or "Untitled check",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        mode="CHECK_ONLY",
        canon=canon,
        status="GENERATING",
    )
    for i, data in enumerate(clips):
        shot_id = f"S{i + 1}"
        uri = storage.clip_path(episode.episode_id, shot_id, 1)
        storage.save_bytes(uri, data)
        spec = ShotSpec(
            shot_id=shot_id, beat_id=f"B{i + 1}", shot_type="uploaded",
            camera_motion="uploaded", action="uploaded clip",
        )
        episode.shots.append(
            Shot(spec=spec, attempts=[Attempt(attempt_no=1, kind="UPLOADED", clip_uri=uri)],
                 chosen_attempt=1)
        )
    storage.save_episode(episode)

    previous_local = None
    for i, shot in enumerate(episode.shots):
        if on_progress:
            on_progress(f"Checking clip {i + 1} of {len(clips)}", i / len(clips))
        attempt = shot.attempts[0]
        local = storage.local_copy(attempt.clip_uri)
        try:
            attempt.critic = checker(local, shot.spec, canon, previous_local)
            shot.status = "ACCEPTED" if attempt.critic.decision == "ACCEPT" else "FLAGGED"
        except Exception as error:  # one bad clip must not lose the others
            attempt.error = str(error)[:500]
            shot.status = "FLAGGED"
        previous_local = local
        storage.save_episode(episode)

    all_ok = all(s.status == "ACCEPTED" for s in episode.shots)
    episode.status = "READY" if all_ok else "NEEDS_REVIEW"
    episode.report = build_report(episode)
    storage.save_episode(episode)
    if on_progress:
        on_progress("Done", 1.0)
    return episode
