"""Measure the critic against hand-labelled clips.

Usage:  python -m eval.run_eval eval/labels.csv path/to/clips [repeats]

labels.csv columns: clip, has error (yes/no), error type (wardrobe, prop, scene;
comma separated), error second, and optionally "judge" (which checks count for
this clip, e.g. "scene" for stock clips; empty means all three).
The expected character, prop and scene are read from eval/expected.json.
"""
import csv
import json
import sys
from pathlib import Path

CHECK_NAMES = {"wardrobe": "wardrobe", "prop": "prop_state", "prop_state": "prop_state",
               "scene": "scene"}


def _checks(text: str) -> list[str]:
    return [CHECK_NAMES[t.strip().lower()] for t in (text or "").split(",")
            if t.strip().lower() in CHECK_NAMES]


def score_row(row: dict, clip_verdicts: dict) -> dict:
    """Compare one clip's verdicts with its label. Pure function, no model call."""
    judged = _checks(row.get("judge", "")) or ["wardrobe", "prop_state", "scene"]
    labelled = [c for c in _checks(row.get("error type", "")) if c in judged]
    has_error = row.get("has error", "").strip().lower() == "yes"
    flagged = [c for c in judged if clip_verdicts.get(c) == "mismatch"]
    unclear = [c for c in judged if clip_verdicts.get(c) == "unclear"]
    if has_error:
        outcome = "caught" if any(c in flagged + unclear for c in labelled) else "missed"
    else:
        outcome = "false_alarm" if flagged else "correct_accept"
    return {
        "clip": row.get("clip", ""), "outcome": outcome,
        "found_all": has_error and all(c in flagged for c in labelled),
        "extra_flags": [c for c in flagged if c not in labelled],
        "sent_to_review": bool(unclear) and not flagged,
    }


def summarise(scored: list[dict]) -> dict:
    count = lambda name: sum(1 for s in scored if s["outcome"] == name)
    errors = count("caught") + count("missed")
    clean = count("false_alarm") + count("correct_accept")
    return {
        "clips": len(scored),
        "errors_caught": f"{count('caught')} of {errors}",
        "false_alarms": f"{count('false_alarm')} of {clean}",
        "sent_to_review": sum(1 for s in scored if s["sent_to_review"]),
        "missed": [s["clip"] for s in scored if s["outcome"] == "missed"],
        "false_alarm_clips": [s["clip"] for s in scored if s["outcome"] == "false_alarm"],
    }


def find_clip(clips_dir: Path, name: str):
    wanted = Path(name).stem.lower()
    for p in clips_dir.iterdir():
        if p.stem.lower() == wanted:
            return p
    return None


def main(labels_path: str, clips_dir: str, repeats: int = 1) -> dict:
    from dramagraph.check_only import make_canon
    from dramagraph.critic import check_clip, verdicts
    from dramagraph.models import ShotSpec

    e = json.loads((Path(__file__).parent / "expected.json").read_text())
    with open(labels_path, newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip().lower(): (v or "") for k, v in r.items() if k} for r in csv.DictReader(f)]

    scored, raw, unstable = [], {}, []
    for row in rows:
        path = find_clip(Path(clips_dir), row["clip"])
        if path is None:
            print("SKIPPED (file not found):", row["clip"])
            continue
        canon = make_canon(e["name"], e["appearance"], e["wardrobe"], e["prop"],
                           [e["prop_state"]], e["location"], e["time_of_day"])
        spec = ShotSpec(shot_id="S1", beat_id="B1", shot_type="", camera_motion="", action="")
        runs = [check_clip(str(path), spec, canon) for _ in range(repeats)]
        if len({r.decision for r in runs}) > 1:
            unstable.append(row["clip"])
        raw[row["clip"]] = [r.model_dump() for r in runs]
        result = score_row(row, verdicts(runs[0]))
        scored.append(result)
        print(row["clip"], verdicts(runs[0]), runs[0].decision, "->", result["outcome"])

    summary = summarise(scored)
    summary["decision_changed_between_runs"] = unstable
    out = Path(labels_path).parent
    (out / "eval_results.json").write_text(json.dumps({"summary": summary, "raw": raw}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
