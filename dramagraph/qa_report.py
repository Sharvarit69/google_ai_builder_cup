"""Builds the QA report from what is stored in the episode. Nothing is invented here."""
from .critic import lines as critic_lines
from .critic import verdicts
from .models import Episode


def build_report(episode: Episode) -> dict:
    shots = []
    for shot in episode.shots:
        attempt = None
        if shot.chosen_attempt is not None:
            attempt = next(
                (a for a in shot.attempts if a.attempt_no == shot.chosen_attempt), None
            )
        elif shot.attempts:
            attempt = shot.attempts[-1]
        critic = attempt.critic if attempt else None
        shots.append(
            {
                "shot_id": shot.spec.shot_id,
                "status": shot.status,
                "attempts": len(shot.attempts),
                "decision": critic.decision if critic else None,
                "verdicts": verdicts(critic) if critic else {},
                "checks": critic_lines(critic) if critic else [],
                "observations": [o.model_dump() for o in critic.observations] if critic else [],
                "violations": [v.model_dump() for v in critic.violations] if critic else [],
                "error": attempt.error if attempt else None,
                "seeded": any(a.seeded for a in shot.attempts),
                "creator_catches": shot.creator_catches,
                "had_error": any(a.critic and a.critic.decision == "REGENERATE"
                                 for a in shot.attempts),
                "repairs": sum(1 for a in shot.attempts
                               if a.kind in ("AUTO_REPAIR", "CUSTOM_REPAIR") and a.clip_uri),
            }
        )
    with_errors = [s for s in shots if s["violations"]]
    return {
        "title": episode.title,
        "created_at": episode.created_at,
        "mode": episode.mode,
        "shots": shots,
        "clips_checked": len(shots),
        "clips_with_errors": len(with_errors),
        "violations_found": sum(len(s["violations"]) for s in shots),
        "needs_review": sum(1 for s in shots if s["decision"] == "REVIEW"),
        "shots_with_errors_found": sum(1 for s in shots if s["had_error"]),
        "shots_fixed_by_repair": sum(
            1 for s in shots if s["had_error"] and s["status"] == "ACCEPTED"),
        "shots_overruled": sum(1 for s in shots if s["status"] == "OVERRULED"),
        "shots_dropped": sum(1 for s in shots if s["status"] == "DROPPED"),
        "creator_catches": sum(s["creator_catches"] for s in shots),
        "rules": [r.model_dump() for r in episode.canon.rules] if episode.canon else [],
        "observations_noted": sum(len(s["observations"]) for s in shots),
        "major_observations": sum(1 for s in shots for o in s["observations"]
                                  if o.get("severity") == "major"),
        "seeded_test": any(s["seeded"] for s in shots),
        "seconds_generated": episode.seconds_generated,
    }


def report_to_markdown(report: dict) -> str:
    lines = [
        f"# QA report: {report['title']}",
        "",
        f"- Date: {report['created_at']}",
        f"- Clips checked: {report['clips_checked']}",
        f"- Clips with errors: {report['clips_with_errors']}",
        f"- Violations found: {report['violations_found']}",
        f"- Clips needing human review: {report['needs_review']}",
    ]
    if report["mode"] == "GENERATE":
        lines += [
            f"- Shots where an error was caught: {report['shots_with_errors_found']}",
            f"- Shots fixed by automatic repair: {report['shots_fixed_by_repair']}",
            f"- Shots accepted by the creator over the critic: {report['shots_overruled']}",
            f"- Problems the creator caught that the critic missed: {report.get('creator_catches', 0)}",
            f"- Rules learned from the creator: {len(report.get('rules', []))}",
            f"- Other things the critic noticed (advisory): {report.get('observations_noted', 0)}, "
            f"of which major: {report.get('major_observations', 0)}",
            f"- Shots dropped: {report['shots_dropped']}",
            f"- Seconds of video generated: {report['seconds_generated']}",
        ]
        if report["seeded_test"]:
            lines.append("- Note: one error in this episode was planted on purpose as a test.")
    lines.append("")
    for s in report["shots"]:
        lines.append(f"## {s['shot_id']}: {s['status']}"
                     + (f" after {s['repairs']} repair(s)" if s["repairs"] else "")
                     + (" (seeded test error)" if s["seeded"] else ""))
        if s["error"]:
            lines.append(f"- Could not be checked: {s['error']}")
        for c in s.get("checks") or [{"label": k, "verdict": v} for k, v in s["verdicts"].items()]:
            lines.append(f"- {c['label']}: {c['verdict']}")
        for v in s["violations"]:
            lines.append(
                f"- VIOLATION {v['type']} at {v['timestamp_seconds']:.1f}s: "
                f"expected {v['expected']}; observed {v['observed']}"
            )
        for o in s.get("observations", []):
            lines.append(f"- Also noticed ({o.get('severity', 'minor')}, advisory) at "
                         f"{o['timestamp_seconds']:.1f}s: {o['what']}"
                         + (f" Suggested fix: {o['suggested_fix']}" if o.get("suggested_fix") else ""))
        lines.append("")
    if report.get("rules"):
        lines.append("## Rules learned from the creator")
        for r in report["rules"]:
            scope = "every shot" if not r["shot_ids"] else ", ".join(r["shot_ids"])
            lines.append(f"- {r['text']} (applies to {scope})")
        lines.append("")
    return "\n".join(lines)
