"""Stitch the accepted shots into one vertical video with a title screen and captions."""
import os
import subprocess
import tempfile
import textwrap

from . import storage
from .models import Episode

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WIDTH, HEIGHT, FPS = 720, 1280, 24
TITLE_SECONDS = 2
ENCODE = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an", "-movflags", "+faststart"]


def _run(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("Video processing failed: " + result.stderr.strip()[-400:])


def _text_file(folder: str, name: str, text: str, width: int) -> str:
    """Text goes in a file so quotes and punctuation cannot break the command."""
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(textwrap.wrap(text.strip(), width)) or " ")
    return path


def _drawtext(text_path: str, size: int, y: str) -> str:
    return (f"drawtext=fontfile={FONT}:textfile={text_path}:expansion=none:fontsize={size}:fontcolor=white:"
            f"line_spacing=10:box=1:boxcolor=black@0.55:boxborderw=18:x=(w-text_w)/2:y={y}")


def make_title_card(folder: str, title: str) -> str:
    out = os.path.join(folder, "title.mp4")
    text = _text_file(folder, "title.txt", title, 18)
    _run(["-f", "lavfi", "-i", f"color=c=0x1b1b2f:s={WIDTH}x{HEIGHT}:d={TITLE_SECONDS}:r={FPS}",
          "-vf", _drawtext(text, 64, "(h-text_h)/2"), *ENCODE, out])
    return out


def prepare_clip(folder: str, index: int, clip_path: str, caption: str) -> str:
    """Same size, frame rate and codec for every clip, sound removed, caption burned in."""
    out = os.path.join(folder, f"part{index}.mp4")
    filters = (f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}")
    if caption.strip():
        text = _text_file(folder, f"cap{index}.txt", caption, 28)
        filters += "," + _drawtext(text, 40, "h-text_h-170")
    _run(["-i", clip_path, "-vf", filters, *ENCODE, out])
    return out


def stitch(folder: str, parts: list[str]) -> str:
    listing = os.path.join(folder, "parts.txt")
    with open(listing, "w", encoding="utf-8") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    out = os.path.join(folder, "final.mp4")
    _run(["-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", out])
    return out


def final_path(episode_id: str) -> str:
    return f"episodes/{episode_id}/final.mp4"


def assemble(episode: Episode) -> str:
    """Build the final video from every shot that was not dropped. Returns its stored path."""
    from .pipeline import can_export, chosen_clip   # imported here to avoid a circular import

    ok, reason = can_export(episode)
    if not ok:
        raise ValueError(reason)
    captions = {b.beat_id: b.caption for b in episode.beats}
    with tempfile.TemporaryDirectory() as folder:
        parts = [make_title_card(folder, episode.title)]
        for i, shot in enumerate(s for s in episode.shots if s.status != "DROPPED"):
            clip = storage.local_copy(chosen_clip(shot))
            parts.append(prepare_clip(folder, i, clip, captions.get(shot.spec.beat_id, "")))
        with open(stitch(folder, parts), "rb") as f:
            storage.save_bytes(final_path(episode.episode_id), f.read())
    episode.final_video_uri = final_path(episode.episode_id)
    episode.status = "EXPORTED"
    storage.save_episode(episode)
    return episode.final_video_uri
