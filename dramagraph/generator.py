"""The only place Veo is called. It cannot run unless the ration has seconds left."""
import time

from . import llm, ration, storage
from .config import get_settings
from .models import Attempt, Episode, Shot

POLL_SECONDS = 10
MAX_WAIT_SECONDS = 360
RATION_EMPTY = "Veo ration used up"


class GenerationRefused(Exception):
    """Veo declined to make the clip. Veo does not charge for these, and they are
    often temporary (for example an audio problem), so a retry can succeed."""


class RequestRejected(Exception):
    """Veo rejected the request before starting any work, so nothing was generated."""


def reference_path(episode_id: str) -> str:
    return f"episodes/{episode_id}/reference.png"


def call_veo(prompt: str, reference=None, client=None, sleep=time.sleep) -> bytes:
    """Submit one clip to Veo, wait for it and return the MP4 bytes."""
    from google.genai import types

    settings = get_settings()
    if not settings.veo_model:
        raise RequestRejected("VEO_MODEL is not set.")
    from . import judge

    judge.before_video()                    # public demo: no video, ever
    client = client or llm.get_client()
    options = {"aspect_ratio": "9:16"}
    # reference is a list of (bytes, mime type) pairs; plain bytes is accepted too
    if isinstance(reference, (bytes, bytearray)):
        reference = [(bytes(reference), "image/png")]
    if reference:
        options["reference_images"] = [
            types.VideoGenerationReferenceImage(
                image=types.Image(image_bytes=data, mime_type=mime),
                reference_type="asset",
            )
            for data, mime in reference
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

    from . import judge

    judge.before_video()                    # public demo: stop before any budget is taken
    seconds = shot.spec.duration_seconds
    if not ration.reserve(seconds, f"{episode.episode_id} {shot.spec.shot_id} a{attempt.attempt_no}"):
        attempt.error = RATION_EMPTY
        return attempt

    from . import pictures as pics   # imported here to avoid a circular import

    reference, used = [], []
    for owner, picture in pics.pick_for_shot(episode, shot.spec) if episode.canon else []:
        data = pics.load(picture)
        if data:
            reference.append((data, picture.mime))
            used.append(pics.describe(owner, picture))
    try:
        try:
            data = veo(prompt, reference or None)
            attempt.references_used = used
        except RequestRejected as error:
            words = str(error).lower()
            if not reference or not ("reference" in words or "image" in words):
                raise
            # the model would not take the pictures; go without, and say so, because a
            # version made without them is the likely cause of any character drift
            attempt.references_refused = used
            data = veo(prompt, None)
        storage.save_bytes(path, data)
        attempt.clip_uri = path
        episode.seconds_generated += seconds
    except RequestRejected as error:
        ration.refund(seconds, "request rejected before generation")
        attempt.error = f"Request rejected: {error}"[:500]
    except GenerationRefused as error:
        ration.refund(seconds, "Veo declined; not charged")
        attempt.error = f"REFUSED: {error}"[:500]
    except Exception as error:
        attempt.error = str(error)[:500]
    return attempt


def generate_with_retries(episode: Episode, shot: Shot, prompt: str, kind: str,
                          seeded: bool = False, veo=call_veo) -> Attempt:
    """Try, then retry up to MAX_GEN_RETRIES times. An empty ration stops early."""
    attempt = None
    for _ in range(1 + get_settings().max_gen_retries):
        attempt = generate_clip(episode, shot, prompt, kind, seeded, veo)
        if attempt.clip_uri or attempt.error == RATION_EMPTY:
            break
    return attempt
