"""Read and write files. Uses Cloud Storage if GCS_BUCKET is set, else local_data/."""
import os
import tempfile
from pathlib import Path

from .config import get_settings
from .models import Episode

LOCAL_ROOT = Path("local_data")


def _bucket():
    name = get_settings().gcs_bucket
    if not name:
        return None
    from google.cloud import storage  # imported only when needed

    return storage.Client().bucket(name)


def save_bytes(path: str, data: bytes) -> str:
    bucket = _bucket()
    if bucket is None:
        target = LOCAL_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    else:
        bucket.blob(path).upload_from_string(data)
    return path


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
