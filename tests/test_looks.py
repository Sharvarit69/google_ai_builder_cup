"""The person and props must look the same in every shot: point-by-point checks,
close looks on sharp stills, and an approved look locked from the first shot that
passed. None of it is tied to one story, so these tests use a different one."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dramagraph import check_only, critic, llm, looks, pictures, pipeline, storage
from dramagraph.models import CheckResult, CriticResult, LookPoint, ShotSpec
from dramagraph.problems import for_critic
from tests.fake_client import FakeClient

UMBRELLA = ("long curved-handle umbrella, one canopy with eight ribs, nylon, glossy, "
            "bright red, as tall as his hip")
SPEC = ShotSpec(shot_id="S1", beat_id="B1", shot_type="", camera_motion="", action="he waits")


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    from dramagraph import generator
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    for k, v in {"GCS_BUCKET": "", "GEMINI_API_KEY": "k", "TEXT_MODEL": "t",
                 "VIDEO_UNDERSTANDING_MODEL": "v", "VEO_MODEL": "veo", "VEO_BUDGET_SECONDS": "200",
                 "SHOT_SECONDS": "8", "MAX_REPAIRS": "2", "MAX_GEN_RETRIES": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(generator, "POLL_SECONDS", 0)
    monkeypatch.setattr("time.sleep", lambda s: None)


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    return fake


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=360x640:d=8:r=24",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def bus_stop():
    return check_only.make_canon("Arun", "adult man, short grey hair", "plain navy wool coat",
                                 "umbrella", ["closed"], "bus stop", "evening",
                                 prop_description=UMBRELLA)


def test_prompts_carry_no_example_from_one_story():
    words = re.compile(r"lunch|tiffin|samosa|priya|kurta", re.I)
    for path in Path("prompts").glob("*.txt"):
        assert not words.search(path.read_text(encoding="utf-8")), path.name


def test_any_point_that_differs_fails_the_look():
    c = bus_stop()
    result = CriticResult(wardrobe=CheckResult(verdict="match", observed=""),
                          prop_look=CheckResult(verdict="match", observed="umbrella"),
                          prop_state=CheckResult(verdict="match", observed=""),
                          scene=CheckResult(verdict="match", observed=""),
                          look_points=[LookPoint(item="umbrella", point="finish", verdict="mismatch",
                                                 observed="matte, not glossy"),
                                       LookPoint(item="umbrella", point="shape", verdict="match")])
    critic.apply_looks(result, c)
    assert result.prop_look.verdict == "mismatch" and "finish: matte" in result.prop_look.observed
    assert critic.decide(result) == "REGENERATE"
    assert {"key": "look_umbrella_finish", "label": "umbrella · finish", "verdict": "mismatch",
            "observed": "matte, not glossy"} in critic.lines(result)


def test_close_look_catches_a_changed_shape_with_close_ups(client, clip):
    client.look_mismatches = {("object 1", "shape"): "folded flat and rectangular"}
    reference = Path(clip).with_suffix(".png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", clip, "-frames:v", "1",
                    str(reference)], check=True)
    result = critic.check_clip(clip, SPEC, bus_stop(),
                               references={"umbrella": ("your picture", reference.read_bytes())})
    compare = client.close_prompts[-1]
    assert "bright red" in compare and "object 1 (umbrella), your picture (the reference)" in compare

    shape = next(p for p in result.look_points if p.item == "umbrella" and p.point == "shape")
    assert shape.close_look and shape.verdict == "mismatch" and shape.timestamp_seconds == 1.2
    assert result.prop_look.verdict == "mismatch" and result.decision == "REGENERATE"
    problem = next(p for p in for_critic(result) if "looks different" in p.headline)
    assert problem.headline == "Umbrella looks different: shape"
    assert problem.image and storage.load_bytes(problem.image).startswith(b"\x89PNG")
    assert not problem.review_only           # look drift is fixed by a remake, unlike a clock time


def test_close_look_checks_the_person_too(client, clip):
    client.look_mismatches = {("person", "clothes"): "a beige raincoat, not a navy wool coat"}
    result = critic.check_clip(clip, SPEC, bus_stop())
    assert result.wardrobe.verdict == "mismatch" and "beige raincoat" in result.wardrobe.observed
    assert any(v.type == "WARDROBE" and v.crop_uri for v in result.violations)


def test_the_first_shot_that_passes_sets_the_approved_look(client, clip):
    from tests.test_pipeline import planned_episode
    ep = planned_episode(2)
    pipeline.run_generation(ep)                 # fake clips cannot be read, so nothing is locked yet
    assert ep.canon.character.approved is None
    for shot in ep.shots:
        for a in shot.attempts:
            shutil.copy(clip, storage.local_copy(a.clip_uri))

    assert looks.lock(ep) == ["character", "steel lunch box"]
    approved = ep.canon.props[0].approved
    assert approved.label == "approved look · shot 1" and storage.exists(approved.uri)
    names = [owner for owner, _ in pictures.pick_for_shot(ep, ep.shots[1].spec)]
    assert names == ["Priya", "steel lunch box"]          # sent to Veo for later shots and repairs
    assert set(pipeline._references(ep)) == {"character", "steel lunch box"}
    assert looks.lock(ep) == []                            # once only

    # the creator removes it: it is not taken again, and their own picture always wins
    pictures.remove_picture(ep, "steel lunch box", approved.uri)
    assert ep.canon.props[0].approved is None and looks.lock(ep) == []
    pictures.add_picture(ep, "character", Path(clip).read_bytes()[:10], "me.png", "front")
    assert pipeline._references(ep)["character"][0] == "your picture"
