"""Read and write files. Uses Cloud Storage if GCS_BUCKET is set, else local_data/."""
import json
import os
import tempfile
from pathlib import Path

from . import judge
from .config import get_settings
from .models import Episode

LOCAL_ROOT = Path("local_data")


def _bucket():
    name = get_settings().gcs_bucket
    if not name:
        return None
    from google.cloud import storage  # imported only when needed

    return storage.Client().bucket(name)


def _writable(path: str) -> None:
    """In the public demo, the demo episodes cannot be changed."""
    if judge.protected(path):
        raise PermissionError(judge.PROTECTED)


def save_bytes(path: str, data: bytes) -> str:
    _writable(path)
    bucket = _bucket()
    if bucket is None:
        target = LOCAL_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    else:
        bucket.blob(path).upload_from_string(data)
    return path


def update_json(path: str, change, attempts: int = 10):
    """Read a small JSON file, apply change(data) -> (new data or None, result), and
    write it back only if nobody else changed it in between. Returns the result.

    On Cloud Storage this uses the object's generation number, so two writers at once
    cannot both win; the loser reads again and retries."""
    _writable(path)
    bucket = _bucket()
    if bucket is None:
        target = LOCAL_ROOT / path
        data = json.loads(target.read_bytes()) if target.exists() else None
        new, result = change(data)
        if new is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(json.dumps(new).encode("utf-8"))
        return result
    for _ in range(attempts):
        try:
            blob = bucket.get_blob(path)
            generation = blob.generation if blob is not None else 0   # 0: must not exist yet
            data = (json.loads(blob.download_as_bytes(if_generation_match=generation))
                    if blob is not None else None)
            new, result = change(data)
            if new is None:
                return result
            bucket.blob(path).upload_from_string(json.dumps(new), if_generation_match=generation)
            return result
        except Exception as error:      # someone else wrote in between: read again and retry
            if type(error).__name__ != "PreconditionFailed":
                raise
    raise RuntimeError("Too many people at once. Please try again in a moment.")


def load_bytes(path: str) -> bytes:
    bucket = _bucket()
    if bucket is None:
        return (LOCAL_ROOT / path).read_bytes()
    return bucket.blob(path).download_as_bytes()


def exists(path: str) -> bool:
    bucket = _bucket()
    if bucket is None:
        return (LOCAL_ROOT / path).exists()
    return bucket.blob(path).exists()


def list_paths(prefix: str) -> list[str]:
    bucket = _bucket()
    if bucket is None:
        root = LOCAL_ROOT / prefix
        if not root.exists():
            return []
        return sorted(
            str(p.relative_to(LOCAL_ROOT)).replace(os.sep, "/")
            for p in root.rglob("*")
            if p.is_file()
        )
    return sorted(b.name for b in bucket.list_blobs(prefix=prefix))


def local_copy(path: str) -> str:
    """Return a path on this machine for a stored file (FFmpeg and uploads need one)."""
    if _bucket() is None:
        return str(LOCAL_ROOT / path)
    suffix = Path(path).suffix
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(load_bytes(path))
    return tmp


def episode_path(episode_id: str) -> str:
    return f"episodes/{episode_id}/episode.json"


def clip_path(episode_id: str, shot_id: str, attempt_no: int) -> str:
    return f"episodes/{episode_id}/clips/{shot_id}_a{attempt_no}.mp4"


def save_episode(episode: Episode) -> None:
    data = episode.model_dump_json(indent=2).encode("utf-8")
    save_bytes(episode_path(episode.episode_id), data)


def load_episode(episode_id: str) -> Episode:
    return Episode.model_validate_json(load_bytes(episode_path(episode_id)))


def list_episodes() -> list[dict]:
    """Newest first: id, title, date, mode and status for the saved list."""
    rows = []
    for path in list_paths("episodes/"):
        if not path.endswith("/episode.json"):
            continue
        ep = Episode.model_validate_json(load_bytes(path))
        rows.append(
            {
                "episode_id": ep.episode_id,
                "title": ep.title,
                "created_at": ep.created_at,
                "mode": ep.mode,
                "status": ep.status,
            }
        )
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)
