import pytest

from dramagraph import check_only, critic, storage
from dramagraph.models import CheckResult, CriticResult, ShotSpec, Violation
from dramagraph.qa_report import build_report, report_to_markdown
from eval.run_eval import score_row, summarise


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "")
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)


def result(w="match", p="match", s="match", x=None, violations=()):
    return CriticResult(
        wardrobe=CheckResult(verdict=w, observed="w"),
        prop_state=CheckResult(verdict=p, observed="p"),
        scene=CheckResult(verdict=s, observed="s"),
        cross_shot=CheckResult(verdict=x, observed="x") if x else None,
        violations=list(violations),
    )


@pytest.mark.parametrize("args,expected", [
    (dict(), "ACCEPT"),
    (dict(w="mismatch"), "REGENERATE"),
    (dict(p="unclear"), "REVIEW"),
    (dict(p="unclear", s="mismatch"), "REGENERATE"),
    (dict(x="mismatch"), "REGENERATE"),
    (dict(x="unclear"), "REVIEW"),
    (dict(x="match"), "ACCEPT"),
])
def test_decide(args, expected):
    assert critic.decide(result(**args)) == expected


def canon(states=("closed", "open")):
    return check_only.make_canon("Priya", "adult woman", "yellow kurta",
                                 "lunch box", list(states), "office desk", "daytime")


def spec(shot_id="S1", second_person=None):
    return ShotSpec(shot_id=shot_id, beat_id="B1", shot_type="", camera_motion="",
                    action="", second_person=second_person)


def test_prompt_uses_the_prop_state_for_that_shot():
    assert "CLOSED" in critic.build_prompt(spec("S1"), canon())
    assert "OPEN" in critic.build_prompt(spec("S2"), canon())


def test_prompt_cross_shot_and_hands_notes():
    plain = critic.build_prompt(spec(), canon())
    assert "Set cross_shot to null" in plain and "Hands alone" not in plain
    both = critic.build_prompt(spec(second_person="hands only"), canon(), has_previous=True)
    assert "previous shot" in both and "Hands alone" in both


def test_check_clip_overrides_model_decision(monkeypatch):
    monkeypatch.setattr(critic.llm, "upload_video", lambda path, client=None: path)
    model_reply = result(p="mismatch", x="match", violations=[
        Violation(type="PROP_STATE", timestamp_seconds=-2, expected="closed", observed="open")])
    model_reply.decision = "ACCEPT"  # the model is wrong; code must overrule it
    seen = {}

    def fake_ask(**kwargs):
        seen.update(kwargs)
        return model_reply

    monkeypatch.setattr(critic.llm, "ask_json", fake_ask)
    out = critic.check_clip("a.mp4", spec(), canon(), client=object())
    assert out.decision == "REGENERATE"
    assert out.cross_shot is None and len(seen["media"]) == 1
    assert out.violations[0].timestamp_seconds == 0.0


def test_check_only_run_saves_and_compares_with_previous():
    calls = []

    def fake_checker(local, shot_spec, canon_, previous):
        calls.append((shot_spec.shot_id, previous is not None))
        if shot_spec.shot_id == "S2":
            raise RuntimeError("upload failed")
        r = result()
        r.decision = "ACCEPT"
        return r

    ep = check_only.run_check_only("t", canon(), [b"one", b"two", b"three"], checker=fake_checker)
    assert calls == [("S1", False), ("S2", True), ("S3", True)]
    assert [s.status for s in ep.shots] == ["ACCEPTED", "FLAGGED", "ACCEPTED"]
    assert ep.status == "NEEDS_REVIEW"
    saved = storage.load_episode(ep.episode_id)
    assert saved == ep and storage.load_bytes(saved.shots[2].attempts[0].clip_uri) == b"three"
    assert "upload failed" in report_to_markdown(saved.report)


def test_check_only_limits():
    with pytest.raises(ValueError):
        check_only.run_check_only("t", canon(), [])
    with pytest.raises(ValueError):
        check_only.run_check_only("t", canon(), [b"x"] * 5)


def test_report_counts_come_from_the_record():
    def checker(local, shot_spec, canon_, previous):
        bad = shot_spec.shot_id == "S1"
        r = result(w="mismatch" if bad else "match", violations=[
            Violation(type="WARDROBE", timestamp_seconds=1.5, expected="yellow", observed="red")
        ] if bad else [])
        r.decision = critic.decide(r)
        return r

    ep = check_only.run_check_only("t", canon(), [b"a", b"b"], checker=checker)
    report = build_report(ep)
    assert (report["clips_checked"], report["clips_with_errors"], report["violations_found"]) == (2, 1, 1)
    assert "WARDROBE at 1.5s" in report_to_markdown(report)


V = lambda w="match", p="match", s="match": {"wardrobe": w, "prop_state": p, "scene": s}


def test_score_row_outcomes():
    err = {"clip": "c4", "has error": "yes", "error type": "wardrobe"}
    ok = {"clip": "c1", "has error": "no", "error type": ""}
    assert score_row(err, V(w="mismatch"))["outcome"] == "caught"
    assert score_row(err, V(p="mismatch"))["outcome"] == "missed"   # wrong check flagged
    assert score_row(err, V())["outcome"] == "missed"
    assert score_row(ok, V())["outcome"] == "correct_accept"
    assert score_row(ok, V(s="mismatch"))["outcome"] == "false_alarm"
    assert score_row(ok, V(s="unclear"))["outcome"] == "correct_accept"


def test_score_row_judges_only_named_checks_for_stock_clips():
    stock_ok = {"clip": "s1", "has error": "no", "error type": "", "judge": "scene"}
    stock_bad = {"clip": "s3", "has error": "yes", "error type": "scene", "judge": "scene"}
    assert score_row(stock_ok, V(w="mismatch", p="mismatch"))["outcome"] == "correct_accept"
    assert score_row(stock_bad, V(w="mismatch", s="mismatch"))["outcome"] == "caught"


def test_summarise():
    s = summarise([
        {"clip": "a", "outcome": "caught", "sent_to_review": False},
        {"clip": "b", "outcome": "missed", "sent_to_review": False},
        {"clip": "c", "outcome": "correct_accept", "sent_to_review": True},
    ])
    assert s["errors_caught"] == "1 of 2" and s["false_alarms"] == "0 of 1" and s["missed"] == ["b"]
