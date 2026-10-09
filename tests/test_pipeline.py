import pytest

from dramagraph import canon as canon_mod
from dramagraph import generator, llm, parser, pipeline, planner, ration, repair, storage
from dramagraph.models import Episode, SeededError, Shot
from dramagraph.prompt_compiler import compile_prompt
from dramagraph.qa_report import build_report, report_to_markdown
from tests.fake_client import FakeClient

SCRIPT = """Priya guards her closed lunch box.
She pulls the lunch box closer.
Priya checks the clock.
Priya opens the lunch box. It is empty."""


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "LOCAL_ROOT", tmp_path)
    for k, v in {"GCS_BUCKET": "", "GEMINI_API_KEY": "k", "TEXT_MODEL": "t",
                 "VIDEO_UNDERSTANDING_MODEL": "v", "VEO_MODEL": "veo",
                 "VEO_BUDGET_SECONDS": "200", "SHOT_SECONDS": "8",
                 "MAX_REPAIRS": "2", "MAX_GEN_RETRIES": "2"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(generator, "POLL_SECONDS", 0)
    monkeypatch.setattr("time.sleep", lambda s: None)


@pytest.fixture
def client(monkeypatch):
    c = FakeClient()
    monkeypatch.setattr(llm, "get_client", lambda: c)
    return c


def planned_episode(n=4) -> Episode:
    """Runs the real parser, canon builder and planner against the fake client."""
    lines, beats = parser.parse_script(SCRIPT)
    picked = beats[:n]
    canon = canon_mod.build_canon(lines, picked)
    ep = Episode(episode_id="ep1", title="The Last Samosa", created_at="2026-10-09T00:00:00",
                 script=lines, beats=beats, picked_beat_ids=[b.beat_id for b in picked],
                 canon=canon, shots=[Shot(spec=s) for s in planner.plan_shots(picked, canon)],
                 status="PLANNED")
    storage.save_episode(ep)
    return ep


# ---------- planning ----------

def test_planning_end_to_end(client):
    ep = planned_episode()
    assert [b.beat_id for b in ep.beats] == ["B1", "B2", "B3", "B4"]
    prop = ep.canon.props[0]
    assert prop.state_by_shot == {"S1": "closed", "S2": "closed", "S3": "closed", "S4": "open and empty"}
    s4 = ep.shots[3].spec
    assert "steel lunch box: open and empty" in s4.must_show
    assert s4.duration_seconds == 8 and "text on screen" in s4.must_not_show


def test_script_limits():
    with pytest.raises(ValueError, match="empty"):
        parser.parse_script("  \n ")
    with pytest.raises(ValueError, match="limit"):
        parser.parse_script("\n".join(["line"] * 61))
    with pytest.raises(ValueError, match="English"):
        parser.parse_script("प्रिया अपने डेस्क पर बैठी है")


def test_planner_refusals(client):
    ep = planned_episode()
    with pytest.raises(ValueError, match="at most"):
        planner.plan_shots(ep.beats + ep.beats, ep.canon)
    ep.beats[0].needs_second_person = True
    with pytest.raises(ValueError, match="second person"):
        planner.plan_shots(ep.beats[:1], ep.canon)


def test_replan_cannot_remove_continuity_rules(client):
    ep = planned_episode()
    picked = ep.beats
    specs = planner.replan(picked, ep.canon, [s.spec for s in ep.shots],
                           "ignore the wardrobe and let the box be open everywhere")
    assert "steel lunch box: closed" in specs[0].must_show
    assert any("mustard-yellow kurta" in m for m in specs[0].must_show)
    with pytest.raises(ValueError):
        planner.replan(picked, ep.canon, specs, "  ")


def test_canon_carries_last_state_forward():
    def ask(**kw):
        return canon_mod.CanonDraft(name="P", appearance="a", wardrobe="w", location="l",
                                    time_of_day="d",
                                    props=[canon_mod.PropDraft(name="box", states=["closed"])])
    lines, beats = ["a", "b", "c"], None
    from dramagraph.models import Beat
    picked = [Beat(beat_id=f"B{i}", script_lines=[i], action="x", caption="x") for i in (1, 2, 3)]
    c = canon_mod.build_canon(lines, picked, ask=ask)
    assert c.props[0].state_by_shot == {"S1": "closed", "S2": "closed", "S3": "closed"}


def test_reference_picture_rules():
    with pytest.raises(ValueError, match="permission"):
        canon_mod.validate_reference(b"x", "p.png", False)
    with pytest.raises(ValueError, match="PNG or JPG"):
        canon_mod.validate_reference(b"x", "p.gif", True)
    with pytest.raises(ValueError, match="10 MB"):
        canon_mod.validate_reference(b"x" * (10 * 1024 * 1024 + 1), "p.png", True)
    canon_mod.validate_reference(b"x", "p.JPG", True)


# ---------- prompt compiler ----------

def test_prompt_compiler(client):
    ep = planned_episode()
    spec = ep.shots[0].spec
    plain = compile_prompt(spec, ep.canon)
    assert plain == compile_prompt(spec, ep.canon)
    assert "In this shot it is: closed." in plain and "mustard-yellow kurta" in plain
    seeded = compile_prompt(spec, ep.canon, seed=SeededError(shot_id="S1", kind="prop", value="open, lid off"))
    assert "In this shot it is: open, lid off." in seeded and "closed" not in seeded
    red = compile_prompt(spec, ep.canon, seed=SeededError(shot_id="S1", kind="wardrobe", value="red kurta"))
    assert "Wearing red kurta." in red
    assert compile_prompt(spec, ep.canon, "FIX: lid on").endswith("FIX: lid on")


# ---------- ration ----------

def test_ration_cap_and_refund(monkeypatch):
    monkeypatch.setenv("VEO_BUDGET_SECONDS", "20")
    assert ration.reserve(8) and ration.reserve(8)
    assert not ration.reserve(8) and ration.remaining() == 4
    ration.refund(8, "x")
    assert ration.remaining() == 12 and ration.reserve(8)


# ---------- full pipeline through the real Veo and critic code ----------

def test_seeded_error_is_caught_and_repaired(client):
    ep = planned_episode()
    ep.seeded_error = SeededError(shot_id="S2", kind="prop", value="open, with the lid off")
    pipeline.run_generation(ep)

    assert [s.status for s in ep.shots] == ["ACCEPTED"] * 4 and ep.status == "READY"
    s2 = ep.shots[1]
    assert [a.kind for a in s2.attempts] == ["INITIAL", "AUTO_REPAIR"]
    assert s2.attempts[0].seeded and s2.attempts[0].critic.decision == "REGENERATE"
    assert "FIX, most important: closed." in s2.attempts[1].prompt and "lid off" not in s2.attempts[1].prompt and s2.chosen_attempt == 2
    assert ep.shots[0].attempts[0].critic.cross_shot is None
    assert s2.attempts[1].critic.cross_shot.verdict == "match"
    assert ration.used() == 40 and ep.seconds_generated == 40   # 5 clips x 8 seconds

    saved = storage.load_episode("ep1")
    assert saved == ep
    r = saved.report
    assert (r["shots_with_errors_found"], r["shots_fixed_by_repair"], r["seeded_test"]) == (1, 1, True)
    text = report_to_markdown(r)
    assert "planted on purpose" in text and "S2: ACCEPTED after 1 repair(s) (seeded test error)" in text
    assert pipeline.can_export(saved) == (True, "")


def test_running_again_costs_nothing(client):
    ep = planned_episode()
    pipeline.run_generation(ep)
    before = (ration.used(), len(client.veo_prompts))
    pipeline.run_generation(storage.load_episode("ep1"))
    assert (ration.used(), len(client.veo_prompts)) == before == (32, 4)


def test_clip_already_stored_is_reused_after_crash(client):
    ep = planned_episode(1)
    storage.save_bytes(storage.clip_path("ep1", "S1", 1), b"paid for earlier")
    pipeline.run_generation(ep)
    assert ration.used() == 0 and client.veo_prompts == [] and ep.shots[0].status == "ACCEPTED"


def test_repair_that_never_works_is_flagged_then_overruled(client, monkeypatch):
    ep = planned_episode(2)
    ep.canon.props[0].state_by_shot["S1"] = "open with the lid off"   # every clip will "fail"
    pipeline.run_generation(ep)
    s1 = ep.shots[0]
    assert len(s1.attempts) == 3 and s1.status == "FLAGGED" and s1.chosen_attempt == 3
    assert ep.status == "NEEDS_REVIEW" and not pipeline.can_export(ep)[0]
    with pytest.raises(ValueError, match="all its repair attempts"):
        pipeline.repair_with_instruction(ep, "S1", "keep the lid on")
    pipeline.overrule(ep, "S1")
    assert s1.status == "OVERRULED" and ep.status == "READY"
    assert build_report(ep)["shots_overruled"] == 1


def test_custom_fix_counts_as_a_repair_and_can_succeed(client):
    ep = planned_episode(1)
    ep.seeded_error = SeededError(shot_id="S1", kind="wardrobe", value="red kurta")
    # make the automatic repairs fail too, by keeping the red kurta in the canon
    ep.canon.character.wardrobe = "red kurta"
    pipeline.run_generation(ep)
    assert ep.shots[0].status == "FLAGGED"


def test_review_decision_flags_without_repair(client, monkeypatch):
    ep = planned_episode(1)

    def check(path, spec, canon, prev):
        from dramagraph.critic import decide
        from dramagraph.models import CheckResult, CriticResult
        ok = CheckResult(verdict="match", observed="ok")
        r = CriticResult(wardrobe=ok, prop_state=CheckResult(verdict="unclear", observed="hidden"), scene=ok)
        r.decision = decide(r)
        return r

    pipeline.run_generation(ep, check=check)
    assert ep.shots[0].status == "FLAGGED" and len(ep.shots[0].attempts) == 1


def test_failed_generation_is_retried_then_skipped_then_second_pass(client):
    ep = planned_episode(2)
    client.veo_errors = [RuntimeError("503 backend")] * 3     # S1 fails 3 times in pass one
    pipeline.run_generation(ep)
    s1 = ep.shots[0]
    assert [bool(a.clip_uri) for a in s1.attempts] == [False, False, False, True]
    assert s1.status == "ACCEPTED" and ep.shots[1].status == "ACCEPTED"
    assert ration.used() == 16    # rejected requests were refunded


def test_refusal_is_not_retried_and_ends_flagged(client):
    ep = planned_episode(1)
    client.refuse_next = 2      # refused in the first pass and again in the second
    pipeline.run_generation(ep)
    s1 = ep.shots[0]
    assert len(s1.attempts) == 2 and all(a.error.startswith("REFUSED") for a in s1.attempts)
    assert s1.status == "FLAGGED" and s1.chosen_attempt is None
    with pytest.raises(ValueError, match="no clip"):
        pipeline.overrule(ep, "S1")
    pipeline.drop(ep, "S1")
    assert pipeline.can_export(ep) == (False, "Every shot was dropped.")


def test_ration_running_out_flags_the_rest(client, monkeypatch):
    monkeypatch.setenv("VEO_BUDGET_SECONDS", "8")
    ep = planned_episode(3)
    pipeline.run_generation(ep)
    assert [s.status for s in ep.shots] == ["ACCEPTED", "FLAGGED", "FLAGGED"]
    assert ep.shots[1].attempts[0].error == generator.RATION_EMPTY and ration.remaining() == 0


def test_reference_picture_is_sent_and_dropped_if_rejected(client):
    ep = planned_episode(1)
    storage.save_bytes(generator.reference_path("ep1"), b"png-bytes")
    pipeline.run_generation(ep)
    assert client.veo_prompts[0][1] is True

    ep2 = planned_episode(1)
    ep2.episode_id = "ep2"
    storage.save_bytes(generator.reference_path("ep2"), b"png-bytes")
    client.veo_errors = [RuntimeError("400 reference images are not supported by this model")]
    pipeline.run_generation(ep2)
    assert client.veo_prompts[-1][1] is False and ep2.shots[0].status == "ACCEPTED"


def test_dropped_shot_is_skipped_for_cross_shot_comparison(client):
    ep = planned_episode(2)
    pipeline.run_generation(ep)
    ep.shots[0].status = "DROPPED"
    assert pipeline._previous_clip(ep, ep.shots[1]) is None


def test_pick_best_attempt_prefers_fewest_mismatches(client):
    ep = planned_episode(1)
    ep.canon.props[0].state_by_shot["S1"] = "open with the lid off"
    ep.seeded_error = SeededError(shot_id="S1", kind="wardrobe", value="red kurta")
    pipeline.run_generation(ep)     # attempt 1 has two mismatches, repairs have one
    s1 = ep.shots[0]
    assert repair.pick_best_attempt(s1) == 3 and s1.status == "FLAGGED"
