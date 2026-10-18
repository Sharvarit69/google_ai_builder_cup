"""Judge mode: visitors can browse saved episodes and check a few clips a day, but nothing
else can reach Gemini or Veo or change what is saved, and the daily count is a hard cap."""
import json

import pytest

from dramagraph import judge, llm, storage
from dramagraph.config import get_settings
from pages_ui.episodes import visible


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    monkeypatch.setenv("GCS_BUCKET", "")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("google.genai.Client", lambda api_key: "a real client")


def test_judge_mode_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    assert get_settings().judge_mode is False
    assert llm.get_client() == "a real client"
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("JUDGE_MODE", value)
        assert get_settings().judge_mode is True
    monkeypatch.setenv("JUDGE_MODE", "0")
    assert get_settings().judge_mode is False


def test_model_calls_only_inside_a_reserved_check(monkeypatch):
    monkeypatch.setenv("JUDGE_MODE", "1")
    with pytest.raises(RuntimeError, match="switched off in the public demo"):
        llm.get_client()
    # every Gemini and Veo call goes through get_client: planning, checking, rules, video
    with pytest.raises(RuntimeError, match="switched off"):
        llm.ask_json(purpose="x", prompt="p", schema=dict, model="m")
    with judge.opened():
        assert llm.get_client() == "a real client"
    with pytest.raises(RuntimeError):
        llm.get_client()                             # closed again afterwards


def test_nothing_is_saved_outside_a_reserved_check(monkeypatch):
    storage.save_bytes("episodes/a/episode.json", b"before")
    monkeypatch.setenv("JUDGE_MODE", "1")
    with pytest.raises(PermissionError):
        storage.save_bytes("episodes/a/episode.json", b"after")
    assert storage.load_bytes("episodes/a/episode.json") == b"before"     # reading still works
    with judge.opened():
        storage.save_bytes("episodes/b/episode.json", b"a visitor's check")


def test_daily_allowance_is_a_hard_cap(monkeypatch):
    monkeypatch.setenv("JUDGE_MODE", "1")
    monkeypatch.setenv("JUDGE_CHECKS_PER_DAY", "5")
    assert judge.remaining() == 5
    assert judge.reserve(4) is True
    assert judge.reserve(2) is False                 # all or nothing: 4 + 2 > 5
    assert judge.reserve(1) is True
    assert judge.reserve(1) is False and judge.remaining() == 0
    assert judge.reserve(0) is False

    # a new day starts a new count, and old days are dropped
    monkeypatch.setattr(judge, "_today", lambda: "2099-01-01")
    assert judge.remaining() == 5 and judge.reserve(5) is True
    assert json.loads(storage.load_bytes(judge.LEDGER)) == {"2099-01-01": 5}


def test_zero_allowance_means_no_checks(monkeypatch):
    monkeypatch.setenv("JUDGE_MODE", "1")
    monkeypatch.setenv("JUDGE_CHECKS_PER_DAY", "0")
    assert judge.reserve(1) is False and judge.remaining() == 0


class PreconditionFailed(Exception):
    pass


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @property
    def generation(self):
        return self.bucket.generation

    def download_as_bytes(self, if_generation_match=None):
        return self.bucket.data

    def upload_from_string(self, data, if_generation_match=None):
        if self.bucket.interfere:                    # another visitor writes first
            self.bucket.interfere -= 1
            self.bucket.data = json.dumps({judge._today(): 3}).encode()
            self.bucket.generation += 1
            raise PreconditionFailed("someone else wrote")
        assert if_generation_match == self.bucket.generation
        self.bucket.data, self.bucket.generation = data.encode(), self.bucket.generation + 1


class FakeBucket:
    def __init__(self):
        self.data, self.generation, self.interfere = None, 0, 1

    def get_blob(self, name):
        return FakeBlob(self, name) if self.data is not None else None

    def blob(self, name):
        return FakeBlob(self, name)


def test_two_visitors_at_once_cannot_both_take_the_last_checks(monkeypatch):
    monkeypatch.setenv("JUDGE_MODE", "1")
    monkeypatch.setenv("JUDGE_CHECKS_PER_DAY", "4")
    bucket = FakeBucket()
    monkeypatch.setattr(storage, "_bucket", lambda: bucket)
    # this visitor asks for 2 while another takes 3 in the same moment: 3 + 2 > 4
    assert judge.reserve(2) is False
    assert json.loads(bucket.data) == {judge._today(): 3}


def test_only_chosen_episodes_are_shown(monkeypatch):
    rows = [{"episode_id": "a"}, {"episode_id": "b"}]
    assert visible(rows, ()) == rows
    monkeypatch.setenv("JUDGE_EPISODES", " b , c ")
    assert get_settings().judge_episodes == ("b", "c")
    assert visible(rows, get_settings().judge_episodes) == [{"episode_id": "b"}]


def test_a_reserved_check_runs_and_saves_but_not_without_one(monkeypatch):
    from dramagraph.check_only import make_canon, run_check_only
    from tests.test_critic import result

    monkeypatch.setenv("JUDGE_MODE", "1")
    canon = make_canon("Arun", "adult man", "navy coat", "umbrella", ["closed"], "bus stop", "evening")
    checker = lambda *args, **kw: result()            # stand-in for the Gemini critic
    with pytest.raises(PermissionError):
        run_check_only("t", canon, [b"clip"], checker=checker)
    assert judge.reserve(1)
    with judge.opened():
        episode = run_check_only("t", canon, [b"clip"], checker=checker)
    assert storage.load_episode(episode.episode_id).shots[0].attempts[0].critic.wardrobe.verdict == "match"
