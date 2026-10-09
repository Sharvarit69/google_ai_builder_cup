import json
import shutil
import subprocess

import pytest

from dramagraph import assembler, storage
from dramagraph.models import Attempt, Beat, Episode, Shot, ShotSpec

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "")
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path / "store")


def make_clip(path, size, seconds, with_audio):
    args = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
            f"testsrc=s={size}:d={seconds}:r=30"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=d={seconds}"]
    subprocess.run(args + ["-pix_fmt", "yuv420p", str(path)], check=True)
    return path.read_bytes()


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                          str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def episode(tmp_path, statuses):
    clips = [("720x1280", 3, True), ("1280x720", 2, False), ("720x1280", 2, True)]
    captions = ["It's Priya's lunch: \"hands off\" & 100% hers", "", "Noon at last"]
    ep = Episode(episode_id="ep1", title="The Last Samosa", created_at="2026-10-09T00:00:00")
    for i, ((size, secs, audio), status) in enumerate(zip(clips, statuses)):
        uri = storage.clip_path("ep1", f"S{i+1}", 1)
        storage.save_bytes(uri, make_clip(tmp_path / f"c{i}.mp4", size, secs, audio))
        ep.beats.append(Beat(beat_id=f"B{i+1}", script_lines=[i + 1], action="x", caption=captions[i]))
        spec = ShotSpec(shot_id=f"S{i+1}", beat_id=f"B{i+1}", shot_type="", camera_motion="", action="")
        ep.shots.append(Shot(spec=spec, attempts=[Attempt(attempt_no=1, kind="INITIAL", clip_uri=uri)],
                             chosen_attempt=1, status=status))
    return ep


def test_assemble_makes_one_vertical_silent_video(tmp_path):
    ep = episode(tmp_path, ["ACCEPTED", "OVERRULED", "ACCEPTED"])
    uri = assembler.assemble(ep)
    info = probe(storage.local_copy(uri))
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    assert len(info["streams"]) == 1 and len(video) == 1          # sound removed
    assert (video[0]["width"], video[0]["height"]) == (720, 1280)  # the wide clip was fitted
    assert abs(float(info["format"]["duration"]) - (2 + 3 + 2 + 2)) < 0.3
    assert ep.status == "EXPORTED" and storage.load_episode("ep1").final_video_uri == uri


def test_dropped_shots_are_left_out(tmp_path):
    ep = episode(tmp_path, ["ACCEPTED", "DROPPED", "ACCEPTED"])
    info = probe(storage.local_copy(assembler.assemble(ep)))
    assert abs(float(info["format"]["duration"]) - (2 + 3 + 2)) < 0.3


def test_cannot_export_with_a_flagged_shot(tmp_path):
    ep = episode(tmp_path, ["ACCEPTED", "FLAGGED", "ACCEPTED"])
    with pytest.raises(ValueError, match="S2"):
        assembler.assemble(ep)
