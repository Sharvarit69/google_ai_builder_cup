"""Builds the QA report from what is stored in the episode. Nothing is invented here."""
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
                "violations": [v.model_dump() for v in critic.violations] if critic else [],
                "error": attempt.error if attempt else None,
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
        "",
    ]
    for s in report["shots"]:
        lines.append(f"## {s['shot_id']}: {s['decision'] or s['status']}")
        if s["error"]:
            lines.append(f"- Could not be checked: {s['error']}")
        for name, verdict in s["verdicts"].items():
            lines.append(f"- {name}: {verdict}")
        for v in s["violations"]:
            lines.append(
                f"- VIOLATION {v['type']} at {v['timestamp_seconds']:.1f}s: "
                f"expected {v['expected']}; observed {v['observed']}"
            )
        lines.append("")
    return "\n".join(lines)
