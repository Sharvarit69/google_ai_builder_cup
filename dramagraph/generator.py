"""The only place Veo is called. It cannot run unless the ration has seconds left."""
import time
from typing import Optional

from . import llm, ration, storage
from .config import get_settings
from .models import Attempt, Episode, Shot

POLL_SECONDS = 10
MAX_WAIT_SECONDS = 360
RATION_EMPTY = "Veo ration used up"


class GenerationRefused(Exception):
    """Veo declined to make the clip (usually a safety filter). Retrying won't help."""


class RequestRejected(Exception):
    """Veo rejected the request before starting any work, so nothing was generated."""


def reference_path(episode_id: str) -> str:
    return f"episodes/{episode_id}/reference.png"


def call_veo(prompt: str, reference: Optional[bytes] = None, client=None,
             sleep=time.sleep) -> bytes:
    """Submit one clip to Veo, wait for it and return the MP4 bytes."""
    from google.genai import types

    settings = get_settings()
    if not settings.veo_model:
        raise RequestRejected("VEO_MODEL is not set.")
    client = client or llm.get_client()
    options = {"aspect_ratio": "9:16"}
    if reference:
        options["reference_images"] = [
            types.VideoGenerationReferenceImage(
                image=types.Image(image_bytes=reference, mime_type="image/png"),
                reference_type="asset",
            )
        ]
    try:
        operation = client.models.generate_videos(
            model=settings.veo_model, prompt=prompt,
            config=types.GenerateVideosConfig(**options),
        )
    except Exception as error:
        raise RequestRejected(str(error)) from error

    waited = 0
    while not operation.done:
        if waited >= MAX_WAIT_SECONDS:
            raise TimeoutError(f"Veo did not finish within {MAX_WAIT_SECONDS} seconds.")
        sleep(POLL_SECONDS)
        waited += POLL_SECONDS
        operation = client.operations.get(operation)

    if getattr(operation, "error", None):
        raise RuntimeError(f"Veo failed: {operation.error}")
    response = operation.response
    videos = getattr(response, "generated_videos", None) or []
    if not videos:
        reasons = getattr(response, "rai_media_filtered_reasons", None)
        raise GenerationRefused(
            "Veo declined to generate this shot. " + ("; ".join(reasons) if reasons else "")
        )
    video = videos[0].video
    data = client.files.download(file=video)
    return data or video.video_bytes


def generate_clip(episode: Episode, shot: Shot, prompt: str, kind: str,
                  seeded: bool = False, veo=call_veo) -> Attempt:
    """One Veo call for one shot. Always appends an Attempt, even on failure."""
    attempt = Attempt(attempt_no=len(shot.attempts) + 1, kind=kind, prompt=prompt, seeded=seeded)
    shot.attempts.append(attempt)
    path = storage.clip_path(episode.episode_id, shot.spec.shot_id, attempt.attempt_no)
    if storage.exists(path):  # already paid for, e.g. after a crash
        attempt.clip_uri = path
        return attempt

    seconds = shot.spec.duration_seconds
    if not ration.reserve(seconds, f"{episode.episode_id} {shot.spec.shot_id} a{attempt.attempt_no}"):
        attempt.error = RATION_EMPTY
        return attempt

    ref_path = reference_path(episode.episode_id)
    reference = storage.load_bytes(ref_path) if storage.exists(ref_path) else None
    try:
        try:
            data = veo(prompt, reference)
        except RequestRejected as error:
            if reference is None or "reference" not in str(error).lower():
                raise
            data = veo(prompt, None)  # the model would not take the picture; go without
        storage.save_bytes(path, data)
        attempt.clip_uri = path
        episode.seconds_generated += seconds
    except RequestRejected as error:
        ration.refund(seconds, "request rejected before generation")
        attempt.error = f"Request rejected: {error}"[:500]
    except GenerationRefused as error:
        attempt.error = f"REFUSED: {error}"[:500]
    except Exception as error:
        attempt.error = str(error)[:500]
    return attempt


def generate_with_retries(episode: Episode, shot: Shot, prompt: str, kind: str,
                          seeded: bool = False, veo=call_veo) -> Attempt:
    """Try, then retry up to MAX_GEN_RETRIES times. Refusals and an empty ration stop early."""
    attempt = None
    for _ in range(1 + get_settings().max_gen_retries):
        attempt = generate_clip(episode, shot, prompt, kind, seeded, veo)
        if attempt.clip_uri or attempt.error == RATION_EMPTY:
            break
        if attempt.error and attempt.error.startswith("REFUSED"):
            break
    return attempt
