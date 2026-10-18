"""Browse-only judge mode: visitors can look at saved episodes, but nothing they do can
reach Gemini or Veo, or change what is saved."""
import pytest

from dramagraph import llm, storage
from dramagraph.config import get_settings
from pages_ui.episodes import visible


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    monkeypatch.setenv("GCS_BUCKET", "")
    monkeypatch.setenv("GEMINI_API_KEY", "k")


def test_judge_mode_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    assert get_settings().judge_mode is False
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("JUDGE_MODE", value)
        assert get_settings().judge_mode is True
    monkeypatch.setenv("JUDGE_MODE", "0")
    assert get_settings().judge_mode is False


def test_no_model_call_is_possible_in_judge_mode(monkeypatch):
    monkeypatch.setenv("JUDGE_MODE", "1")
    with pytest.raises(RuntimeError, match="switched off in this demo"):
        llm.get_client()
    # every Gemini and Veo call goes through get_client, so this covers planning,
    # checking, re-checking, rules and video generation
    with pytest.raises(RuntimeError, match="switched off"):
        llm.ask_json(purpose="x", prompt="p", schema=dict, model="m")


def test_nothing_can_be_saved_in_judge_mode(monkeypatch):
    storage.save_bytes("a.txt", b"before")
    monkeypatch.setenv("JUDGE_MODE", "1")
    with pytest.raises(PermissionError):
        storage.save_bytes("a.txt", b"after")
    assert storage.load_bytes("a.txt") == b"before"      # reading still works


def test_only_chosen_episodes_are_shown(monkeypatch):
    rows = [{"episode_id": "a"}, {"episode_id": "b"}]
    assert visible(rows, ()) == rows
    monkeypatch.setenv("JUDGE_EPISODES", " b , c ")
    assert get_settings().judge_episodes == ("b", "c")
    assert visible(rows, get_settings().judge_episodes) == [{"episode_id": "b"}]
