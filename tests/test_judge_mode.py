"""Public demo (judge mode): the app works as normal, but every Gemini request uses one
unit of a hard daily allowance, video comes out of a hard budget, and the demo episode can
be locked."""
import json

import pytest

from dramagraph import generator, judge, llm, storage
from dramagraph.config import get_settings
from pages_ui.episodes import visible
from tests.fake_client import FakeClient


class Reply:
    text = '{"close_look": true}'


class CountingClient:
    def __init__(self):
        self.models, self.calls = self, 0

    def generate_content(self, **kw):
        self.calls += 1
        return Reply()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    for k, v in {"GCS_BUCKET": "", "GEMINI_API_KEY": "k", "TEXT_MODEL": "t",
                 "JUDGE_MODE": "1", "JUDGE_GEMINI_CALLS_PER_DAY": "3",
                 "JUDGE_EPISODES": "demo1"}.items():
        monkeypatch.setenv(k, v)


def ask(client):
    from dramagraph.close_look import RuleKind
    return llm.ask_json(purpose="x", prompt="p", schema=RuleKind, client=client)


def test_off_unless_switched_on(monkeypatch):
    for value, expected in (("", False), ("0", False), ("1", True), ("true", True), ("ON", True)):
        monkeypatch.setenv("JUDGE_MODE", value)
        assert get_settings().judge_mode is expected


def test_every_gemini_request_uses_the_daily_allowance_then_stops():
    client = CountingClient()
    for _ in range(3):
        ask(client)
    assert judge.remaining() == 0
    with pytest.raises(RuntimeError, match="used up"):
        ask(client)
    assert client.calls == 3                         # the fourth request was never sent


def test_retries_count_too(monkeypatch):
    class Flaky(CountingClient):
        def generate_content(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("503 UNAVAILABLE")
            return Reply()
    client = Flaky()
    from dramagraph.close_look import RuleKind
    llm.ask_json(purpose="x", prompt="p", schema=RuleKind, client=client, sleep=lambda s: None)
    assert client.calls == 2 and judge.remaining() == 1


def test_no_allowance_outside_judge_mode(monkeypatch):
    monkeypatch.setenv("JUDGE_MODE", "0")
    client = CountingClient()
    for _ in range(5):
        ask(client)
    assert client.calls == 5 and not storage.exists(judge.LEDGER)


def test_judges_can_make_video_until_the_video_budget_is_used(monkeypatch):
    from dramagraph import ration
    from tests.test_pipeline import planned_episode
    monkeypatch.setattr(generator, "POLL_SECONDS", 0)
    monkeypatch.setattr("time.sleep", lambda s: None)
    for k, v in {"VIDEO_UNDERSTANDING_MODEL": "v", "VEO_MODEL": "veo", "SHOT_SECONDS": "8",
                 "JUDGE_GEMINI_CALLS_PER_DAY": "100", "VEO_BUDGET_SECONDS": "16"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(llm, "get_client", lambda: FakeClient())
    ep = planned_episode(1)
    first = generator.generate_clip(ep, ep.shots[0], "prompt", "INITIAL")
    second = generator.generate_clip(ep, ep.shots[0], "prompt", "AUTO_REPAIR")
    third = generator.generate_clip(ep, ep.shots[0], "prompt", "CUSTOM_REPAIR")
    assert first.clip_uri and second.clip_uri
    assert not third.clip_uri and third.error == generator.RATION_EMPTY
    assert ration.used() == 16 and ration.remaining() == 0


def test_demo_episode_is_locked_but_new_work_can_be_saved():
    with pytest.raises(PermissionError, match="demo episode"):
        storage.save_bytes("episodes/demo1/episode.json", b"changed")
    storage.save_bytes("episodes/visitor/episode.json", b"a visitor's own episode")
    assert storage.load_bytes("episodes/visitor/episode.json")


def test_only_demo_episodes_are_listed():
    rows = [{"episode_id": "demo1"}, {"episode_id": "old_test"}]
    assert visible(rows, ("demo1",)) == [{"episode_id": "demo1"}]
    assert visible(rows, ()) == rows


def test_all_or_nothing_and_new_day(monkeypatch):
    assert judge.spend(2) is True
    assert judge.spend(2) is False and judge.remaining() == 1
    monkeypatch.setattr(judge, "_today", lambda: "2099-01-01")
    assert judge.remaining() == 3 and judge.spend(3) is True
    assert json.loads(storage.load_bytes(judge.LEDGER)) == {"2099-01-01": 3}


class PreconditionFailed(Exception):
    pass


class FakeBlob:
    def __init__(self, bucket):
        self.bucket = bucket

    @property
    def generation(self):
        return self.bucket.generation

    def download_as_bytes(self, if_generation_match=None):
        return self.bucket.data

    def exists(self):
        return self.bucket.data is not None

    def upload_from_string(self, data, if_generation_match=None):
        if self.bucket.interfere:                    # another visitor writes first
            self.bucket.interfere -= 1
            other = self.bucket.other or {judge._today(): 2}
            self.bucket.data = json.dumps(other).encode()
            self.bucket.generation += 1
            raise PreconditionFailed("someone else wrote")
        assert if_generation_match == self.bucket.generation
        self.bucket.data, self.bucket.generation = data.encode(), self.bucket.generation + 1


class FakeBucket:
    def __init__(self, other=None):
        self.data, self.generation, self.interfere = None, 0, 1
        self.other = other          # what the other visitor writes first

    def get_blob(self, name):
        return FakeBlob(self) if self.data is not None else None

    def blob(self, name):
        return FakeBlob(self)


def test_two_visitors_at_once_cannot_both_take_the_last_units(monkeypatch):
    bucket = FakeBucket()
    monkeypatch.setattr(storage, "_bucket", lambda: bucket)
    # this visitor wants 2 while another takes 2 in the same moment: 2 + 2 > 3
    assert judge.spend(2) is False
    assert json.loads(bucket.data) == {judge._today(): 2}


def test_two_people_cannot_both_take_the_last_video_seconds(monkeypatch):
    from dramagraph import ration
    monkeypatch.setenv("VEO_BUDGET_SECONDS", "96")
    # someone else takes 8 seconds in the same moment, leaving 8 of 96
    bucket = FakeBucket(other={"used": 88, "entries": []})
    monkeypatch.setattr(storage, "_bucket", lambda: bucket)
    assert ration.reserve(16, "this visitor") is False       # 88 + 16 > 96: refused
    assert json.loads(bucket.data)["used"] == 88
    assert ration.reserve(8, "this visitor") is True         # exactly fits
    assert json.loads(bucket.data)["used"] == 96 and ration.remaining() == 0
