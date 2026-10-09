import json

import pytest
from pydantic import BaseModel, ValidationError

from dramagraph import llm, storage
from dramagraph.models import (
    Attempt, Canon, Character, CheckResult, CriticResult, Episode, Prop, Shot, ShotSpec,
)


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "")
    monkeypatch.setenv("TEXT_MODEL", "test-model")
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)


def make_episode(episode_id="ep1", created="2026-10-09T10:00:00"):
    canon = Canon(
        character=Character(name="Priya", appearance="adult woman", wardrobe="yellow kurta"),
        props=[Prop(name="lunch box", description="steel", state_by_shot={"S1": "closed"})],
        location="office desk", time_of_day="day",
    )
    spec = ShotSpec(shot_id="S1", beat_id="B1", shot_type="medium",
                    camera_motion="static", action="guards the lunch box")
    critic = CriticResult(
        wardrobe=CheckResult(verdict="match", observed="yellow kurta"),
        prop_state=CheckResult(verdict="mismatch", observed="open"),
        scene=CheckResult(verdict="match", observed="one person"),
    )
    shot = Shot(spec=spec, attempts=[Attempt(attempt_no=1, kind="INITIAL", critic=critic)])
    return Episode(episode_id=episode_id, title="The Last Samosa", created_at=created,
                   canon=canon, shots=[shot])


def test_episode_saves_and_loads_unchanged():
    ep = make_episode()
    storage.save_episode(ep)
    assert storage.load_episode("ep1") == ep


def test_list_episodes_newest_first():
    storage.save_episode(make_episode("old", "2026-10-01T00:00:00"))
    storage.save_episode(make_episode("new", "2026-10-09T00:00:00"))
    assert [r["episode_id"] for r in storage.list_episodes()] == ["new", "old"]


def test_verdict_must_be_one_of_three():
    with pytest.raises(ValidationError):
        CheckResult(verdict="maybe", observed="x")


def test_bytes_round_trip_and_exists():
    assert not storage.exists("a/b.bin")
    storage.save_bytes("a/b.bin", b"hello")
    assert storage.exists("a/b.bin") and storage.load_bytes("a/b.bin") == b"hello"


class Answer(BaseModel):
    word: str
    count: int


class FakeReply:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Stands in for the Gemini client; returns canned replies or raises."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.prompts.append(contents[-1])
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeReply(item)


def test_ask_json_returns_validated_object():
    client = FakeClient(['{"word": "hi", "count": 2}'])
    out = llm.ask_json("test", "say hi", Answer, client=client)
    assert out == Answer(word="hi", count=2)


def test_ask_json_handles_code_fences():
    client = FakeClient(['```json\n{"word": "hi", "count": 2}\n```'])
    assert llm.ask_json("test", "p", Answer, client=client).count == 2


def test_ask_json_retries_once_after_bad_reply():
    client = FakeClient(['{"word": "hi"}', '{"word": "hi", "count": 1}'])
    out = llm.ask_json("test", "p", Answer, client=client)
    assert out.count == 1
    assert "rejected" in client.prompts[1]


def test_ask_json_gives_up_after_two_bad_replies():
    client = FakeClient(['{"word": "hi"}', "not json"])
    with pytest.raises(ValidationError):
        llm.ask_json("test", "p", Answer, client=client)


def test_ask_json_waits_and_retries_on_rate_limit():
    waits = []
    client = FakeClient([RuntimeError("429 RESOURCE_EXHAUSTED"), '{"word": "a", "count": 0}'])
    out = llm.ask_json("test", "p", Answer, client=client, sleep=waits.append)
    assert out.word == "a" and waits == [2]


def test_ask_json_does_not_retry_other_errors():
    client = FakeClient([RuntimeError("API key not valid")])
    with pytest.raises(RuntimeError):
        llm.ask_json("test", "p", Answer, client=client)


def test_calls_are_logged():
    client = FakeClient(['{"word": "hi", "count": 2}'])
    llm.ask_json("logging-test", "p", Answer, client=client)
    logs = storage.list_paths("logs/")
    assert len(logs) == 1
    assert json.loads(storage.load_bytes(logs[0]).splitlines()[0])["purpose"] == "logging-test"
