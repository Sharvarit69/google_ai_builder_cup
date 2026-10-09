"""Turn the critic's findings into a corrected prompt, and pick the best attempt."""
from .critic import verdicts
from .models import CriticResult, Shot


def auto_fix_notes(result: CriticResult) -> str:
    """A template, not a model call: one FIX line per violation.

    Only the correct state is named. Describing the mistake ("it showed an open
    box") tends to make a video model draw the mistake again.
    """
    lines = []
    for v in result.violations:
        lines.append(f"FIX, most important: {v.expected.rstrip('.')}. "
                     "The previous attempt got this wrong.")
    if not lines:
        for name, verdict in verdicts(result).items():
            if verdict == "mismatch":
                lines.append(f"FIX, most important: the {name.replace('_', ' ')} must match "
                             "the description above exactly. The previous attempt got this wrong.")
    lines.append("Keep everything else the same.")
    return "\n".join(lines)


def repairs_done(shot: Shot) -> int:
    return sum(1 for a in shot.attempts
               if a.kind in ("AUTO_REPAIR", "CUSTOM_REPAIR") and a.clip_uri)


def pick_best_attempt(shot: Shot):
    """Fewest mismatches, then fewest unclear verdicts, then the latest attempt.

    Clips the creator rejected are never chosen, whatever the critic said.
    """
    scored = []
    usable = [a for a in shot.attempts if not a.rejected]
    for a in usable:
        if a.clip_uri and a.critic:
            values = list(verdicts(a.critic).values())
            scored.append((values.count("mismatch"), values.count("unclear"), -a.attempt_no, a))
    if scored:
        return min(scored, key=lambda t: t[:3])[3].attempt_no
    with_clip = [a.attempt_no for a in usable if a.clip_uri]
    return with_clip[-1] if with_clip else None
