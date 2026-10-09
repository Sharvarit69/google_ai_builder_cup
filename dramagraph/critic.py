"""The continuity critic: Gemini says what it sees, code makes the decision."""
from pathlib import Path
from typing import Optional

from . import llm
from .config import get_settings
from .models import Canon, CriticResult, RuleCheck, ShotSpec

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "critic.txt"

SECOND_PERSON_NOTE = (
    "A second person's hands may appear. Hands alone are not an extra person.\n"
)
CROSS_SHOT_NOTE = (
    "\nTwo videos are attached. The FIRST is the previous shot and is only for\n"
    "comparison. The SECOND is the shot to check; all verdicts above are about\n"
    "the second video. Also give a cross_shot verdict: is it the same person,\n"
    "in the same clothes and the same place as the previous shot, and are the\n"
    "props the same objects (same shape, parts and colour) as in the previous shot?\n"
)
NO_CROSS_SHOT_NOTE = "\nSet cross_shot to null.\n"
LABELS = {"wardrobe": "Wardrobe", "prop_look": "Prop appearance", "prop_state": "Prop state",
          "scene": "Scene", "cross_shot": "Matches previous clip"}


def rules_block(spec: ShotSpec, canon: Canon) -> str:
    rules = canon.rules_for(spec.shot_id)
    if not rules:
        return 'There are no extra rules, so "rules" is an empty list.\n'
    lines = ["RULES THE CREATOR ADDED",
             "These came from problems a person spotted. Give each one a verdict under",
             '"rules", using its number. If a rule is about something that does not occur',
             'in this shot, its verdict is "match".']
    lines += [f"{i + 1}. {r.text}" for i, r in enumerate(rules)]
    return "\n".join(lines) + "\n"


def build_prompt(spec: ShotSpec, canon: Canon, has_previous: bool = False) -> str:
    c = canon.character
    character = f"{c.name}, adult. {c.appearance}. Wearing {c.wardrobe}."
    if c.accessories:
        character += " Accessories: " + ", ".join(c.accessories) + "."

    prop_lines = []
    for p in canon.props:
        start, end = p.states_for(spec.shot_id)
        line = f"{p.name}. It looks like this: {p.description}" if p.description else p.name
        if end and start.strip().lower() == end.strip().lower():
            line += f". For the whole shot it must be: {end.upper()}"
        elif end:
            line += (f". By the end of the shot it must be: {end.upper()}. It may begin as "
                     f"{start} and change during the shot")
        prop_lines.append(line)
    prop = "; ".join(prop_lines) if prop_lines else "no specific prop required"

    scene = f"{canon.location}, {canon.time_of_day}. One person only."
    return PROMPT_FILE.read_text(encoding="utf-8").format(
        character=character,
        prop=prop,
        scene=scene,
        second_person_note=SECOND_PERSON_NOTE if spec.second_person else "",
        cross_shot_note=CROSS_SHOT_NOTE if has_previous else NO_CROSS_SHOT_NOTE,
        prop_look_note="" if canon.props else "No prop is required, so set prop_look to null.",
        rules_block=rules_block(spec, canon),
        action_note=(f"This shot is meant to show: {spec.action}\n"
                     if spec.action and spec.action != "uploaded clip" else ""),
    )


def verdicts(result: CriticResult) -> dict[str, str]:
    out = {"wardrobe": result.wardrobe.verdict}
    if result.prop_look is not None:
        out["prop_look"] = result.prop_look.verdict
    out["prop_state"] = result.prop_state.verdict
    out["scene"] = result.scene.verdict
    if result.cross_shot is not None:
        out["cross_shot"] = result.cross_shot.verdict
    for r in result.rules:
        out[f"rule_{r.number}"] = r.verdict
    return out


def lines(result: CriticResult) -> list[dict]:
    """Every check as a row to display: label, verdict and what was observed."""
    rows = []
    for key in ("wardrobe", "prop_look", "prop_state", "scene", "cross_shot"):
        check = getattr(result, key)
        if check is not None:
            rows.append({"key": key, "label": LABELS[key], "verdict": check.verdict,
                         "observed": check.observed})
    for r in result.rules:
        rows.append({"key": f"rule_{r.number}", "label": f"Your rule: {r.rule}",
                     "verdict": r.verdict, "observed": r.observed})
    return rows


def decide(result: CriticResult) -> str:
    """Any mismatch -> REGENERATE. Else any unclear -> REVIEW. Else ACCEPT."""
    values = verdicts(result).values()
    if "mismatch" in values:
        return "REGENERATE"
    if "unclear" in values:
        return "REVIEW"
    return "ACCEPT"


def check_clip(
    clip_path: str,
    spec: ShotSpec,
    canon: Canon,
    previous_clip_path: Optional[str] = None,
    client=None,
) -> CriticResult:
    """Watch one clip (and optionally the shot before it) and return the result."""
    client = client or llm.get_client()
    media = []
    if previous_clip_path:
        media.append(llm.upload_video(previous_clip_path, client=client))
    media.append(llm.upload_video(clip_path, client=client))

    result = llm.ask_json(
        purpose=f"critic:{spec.shot_id}",
        prompt=build_prompt(spec, canon, has_previous=bool(previous_clip_path)),
        schema=CriticResult,
        media=media,
        model=get_settings().video_understanding_model or None,
        client=client,
    )
    if not previous_clip_path:
        result.cross_shot = None
    if not canon.props:
        result.prop_look = None
    # Attach the rule text, drop numbers that do not exist, and make sure no rule is skipped.
    expected = canon.rules_for(spec.shot_id)
    by_number = {r.number: r for r in result.rules if 1 <= r.number <= len(expected)}
    result.rules = []
    for i, rule in enumerate(expected, start=1):
        got = by_number.get(i) or RuleCheck(number=i, verdict="unclear",
                                            observed="The critic did not report on this rule.")
        got.rule = rule.text
        result.rules.append(got)
    result.observations = sorted(result.observations, key=lambda o: o.severity != "major")[:3]
    for o in result.observations:
        o.timestamp_seconds = max(0.0, o.timestamp_seconds)
    for v in result.violations:
        v.timestamp_seconds = max(0.0, v.timestamp_seconds)
    result.decision = decide(result)  # the model's own decision is ignored
    return result
