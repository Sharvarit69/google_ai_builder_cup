"""What is wrong with a shot, in plain words, for people rather than code.

Every problem has a severity, a time, a short headline, one sentence of detail
and a suggested fix. They come from the critic's failed checks, unclear checks,
broken rules and its own observations, for whatever script the episode uses.
"""
from dataclasses import dataclass
from typing import Optional

from .models import Attempt, CriticResult, Shot

LABELS = {"WARDROBE": "Wardrobe", "PROP_LOOK": "Prop looks different", "PROP_STATE": "Prop state",
          "SCENE": "Scene", "CROSS_SHOT": "Does not match the shot before", "RULE": "Your rule"}
CHECK_NAMES = {"wardrobe": "the person or wardrobe", "prop_look": "what the prop looks like",
               "prop_state": "the prop's state", "scene": "the setting",
               "cross_shot": "the match with the shot before"}


@dataclass
class Problem:
    severity: str        # "major" or "minor"
    seconds: float
    headline: str
    detail: str
    fix: str
    source: str = "observation"   # "check" (built-in), "rule" (yours) or "observation"


def _clean(text: str) -> str:
    return " ".join(text.strip().rstrip(".").split())


def _short(text: str, words: int = 9) -> str:
    parts = _clean(text).split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def for_critic(critic: Optional[CriticResult]) -> list[Problem]:
    if critic is None:
        return []
    found: list[Problem] = []
    for v in critic.violations:
        headline = _clean(v.headline) or f"{LABELS.get(v.type, v.type)}: {_short(v.observed, 6)}"
        found.append(Problem("major", v.timestamp_seconds, headline,
                             f"Expected {_clean(v.expected)}; saw {_clean(v.observed)}.",
                             _clean(v.expected) + ".", "rule" if v.type == "RULE" else "check"))
    reported = {_clean(v.expected).lower() for v in critic.violations}
    for r in critic.rules:
        if r.verdict == "mismatch" and _clean(r.rule).lower() not in reported:
            found.append(Problem("major", 0.0, f"Breaks your rule: {_short(r.rule, 7)}",
                                 _clean(r.observed) + "." if r.observed else "", _clean(r.rule) + ".",
                                 "rule"))
    for key, name in CHECK_NAMES.items():
        check = getattr(critic, key, None)
        if check is not None and check.verdict == "unclear":
            found.append(Problem("minor", 0.0, f"Could not see {name} clearly",
                                 _clean(check.observed) + "." if check.observed else "", "", "check"))
    for r in critic.rules:
        if r.verdict == "unclear":
            found.append(Problem("minor", 0.0, f"Could not check your rule: {_short(r.rule, 6)}",
                                 _clean(r.observed) + "." if r.observed else "", "", "rule"))
    for o in critic.observations:
        found.append(Problem(o.severity, o.timestamp_seconds, _clean(o.headline) or _short(o.what),
                             _clean(o.what) + ".", _clean(o.suggested_fix) + "." if o.suggested_fix.strip() else ""))
    found.sort(key=lambda p: (p.severity != "major", p.seconds))
    return found


def chosen_attempt(shot: Shot) -> Optional[Attempt]:
    found = next((a for a in shot.attempts if a.attempt_no == shot.chosen_attempt), None)
    return found or next((a for a in reversed(shot.attempts) if a.clip_uri), None)


def for_shot(shot: Shot) -> list[Problem]:
    attempt = chosen_attempt(shot)
    return for_critic(attempt.critic if attempt else None)


def one_line(shot: Shot) -> str:
    """The single line shown under a shot in the storyboard."""
    if shot.status == "DROPPED":
        return "Left out of the video"
    attempt = chosen_attempt(shot)
    if attempt is None:
        return "Not made yet" if not shot.attempts else "No clip could be made"
    if attempt.critic is None:
        return "Not checked"
    found = for_shot(shot)
    if not found:
        return "All checks passed"
    names = ", ".join(_short(p.headline, 5) for p in found[:2])
    more = f" and {len(found) - 2} more" if len(found) > 2 else ""
    return f"{len(found)} problem{'s' if len(found) > 1 else ''}: {names}{more}"
