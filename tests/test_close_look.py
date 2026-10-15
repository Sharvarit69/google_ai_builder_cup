"""Rules about small details (a clock time, writing) get a close look on sharp, enlarged
still frames, and a wrong reading goes to the creator for review instead of a remake."""
import subprocess

import pytest

from dramagraph import check_only, close_look, critic, llm, pipeline, storage
from dramagraph.models import LearnedRule, ShotSpec
from dramagraph.problems import for_critic
from tests.fake_client import FakeClient

CLOCK = "The wall clock shows 12:00."


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


def canon_with(*rules):
    c = check_only.make_canon("Priya", "adult woman", "yellow kurta", "lunch box",
                              ["closed"], "office desk", "daytime")
    c.rules = [LearnedRule(text=t) for t in rules]
    return c


SPEC = ShotSpec(shot_id="S1", beat_id="B1", shot_type="", camera_motion="", action="she waits")


def test_frames_are_sharp_stills_from_start_middle_and_end(clip):
    stills = close_look.frames(clip)
    assert [round(at, 1) for at, _ in stills] == [1.2, 4.0, 7.2]
    assert all(png.startswith(b"\x89PNG") for _, png in stills)
    image = close_look.crop(stills[0][1], [300, 300, 600, 700])
    assert max(image.size) == close_look.CROP_SIDE                 # enlarged
    assert close_look.crop(stills[0][1], [600, 300, 300, 700]) is None   # nonsense box


def test_clock_rule_gets_a_close_look_and_goes_to_review(client, clip):
    c = canon_with(CLOCK, "Her watch stays on her left wrist.")
    result = critic.check_clip(clip, SPEC, c)

    assert c.rules[0].close_look is True and c.rules[1].close_look is False   # decided once, stored
    assert CLOCK not in critic.build_prompt(SPEC, c)              # not left to the low-detail watch
    clock = next(r for r in result.rules if r.rule == CLOCK)
    assert clock.close_look and clock.verdict == "mismatch"
    assert clock.observed == "Close look shows about 12:15" and clock.timestamp_seconds == 7.2
    assert storage.load_bytes(clock.crop_uri).startswith(b"\x89PNG")
    assert "frame 1" not in client.close_prompts[1] and "Close-up 2 is from 0:07" in client.close_prompts[1]
    assert result.decision == "REVIEW"                            # sent to the creator, not remade

    problem = next(p for p in for_critic(result) if p.source == "rule" and p.severity == "major")
    assert problem.image == clock.crop_uri and problem.review_only and problem.seconds == 7.2


def test_an_unsure_reading_is_unclear_never_a_guess(client, clip):
    client.close_reading = ("2:34", False, "mismatch")
    result = critic.check_clip(clip, SPEC, canon_with(CLOCK))
    clock = result.rules[0]
    assert clock.verdict == "unclear" and clock.observed == "Could not read it clearly; it may show 2:34"
    assert result.decision == "REVIEW"


def test_a_correct_reading_passes(client, clip):
    client.close_reading = ("12:02", True, "match")
    assert critic.check_clip(clip, SPEC, canon_with(CLOCK)).decision == "ACCEPT"


def test_close_look_never_spends_video_on_a_remake(client):
    from tests.test_pipeline import planned_episode
    ep = planned_episode(1)
    pipeline.learn(ep, "S1", "the clock should show 12:00", "creator_catch")
    assert ep.canon.rules[0].close_look is True
    pipeline.run_generation(ep)
    s1 = ep.shots[0]
    assert len(s1.attempts) == 1 and s1.status == "FLAGGED"   # no automatic remake
    clock = s1.attempts[0].critic.rules[0]
    assert s1.attempts[0].critic.decision == "REVIEW" and clock.close_look and clock.verdict == "unclear"


def test_old_rules_and_the_shared_library_keep_the_decision(client):
    from dramagraph import learning
    rule = LearnedRule(text=CLOCK, close_look=True)
    learning.remember(rule)
    assert learning.library()[0].close_look is True
