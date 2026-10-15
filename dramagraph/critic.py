"""The continuity critic: Gemini says what it sees, code makes the decision."""
from pathlib import Path
from typing import Optional

from functools import partial

from . import close_look, llm, looks
from .config import get_settings
from .models import Canon, CriticResult, RuleCheck, ShotSpec, Violation

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "critic.txt"

SECOND_PERSON_NOTE = (
    "A second person's hands may appear. Hands alone are not an extra person.\n"
)
CROSS_SHOT_NOTE = (
    "\nTwo videos are attached after any pictures. The FIRST video is the previous shot and is only for\n"
    "comparison. The SECOND is the shot to check; all verdicts above are about\n"
    "the second video. Also give a cross_shot verdict: is it the same person,\n"
    "in the same clothes and the same place as the previous shot, and are the\n"
    "props the same objects (same shape, parts and colour) as in the previous shot?\n"
)
NO_CROSS_SHOT_NOTE = "\nSet cross_shot to null.\n"
LABELS = {"wardrobe": "Wardrobe", "prop_look": "Prop appearance", "prop_state": "Prop state",
          "scene": "Scene", "cross_shot": "Matches previous clip"}


def watched_rules(spec: ShotSpec, canon: Canon) -> list:
    """Rules the critic checks while watching the clip; the rest get a close look."""
    return [r for r in canon.rules_for(spec.shot_id) if not r.close_look]


def rules_block(spec: ShotSpec, canon: Canon) -> str:
    rules = watched_rules(spec, canon)
    if not rules:
        return 'There are no extra rules, so "rules" is an empty list.\n'
    lines = ["RULES THE CREATOR ADDED",
             "These came from problems a person spotted. Give each one a verdict under",
             '"rules", using its number. If a rule is about something that does not occur',
             'in this shot, its verdict is "match".']
    lines += [f"{i + 1}. {r.text}" for i, r in enumerate(rules)]
    return "\n".join(lines) + "\n"


def pictures_note(labels: list[str]) -> str:
    if not labels:
        return ""
    lines = ["REFERENCE PICTURES",
             "The pictures attached before the video(s) show how things must look:"]
    lines += [f"Picture {i + 1}: {label}" for i, label in enumerate(labels)]
    lines.append("Compare the person and props in the video with these pictures.")
    return "\n".join(lines) + "\n\n"


def build_prompt(spec: ShotSpec, canon: Canon, has_previous: bool = False,
                 picture_labels: Optional[list[str]] = None) -> str:
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
        pictures_note=pictures_note(picture_labels or []),
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
        if key == "prop_look":
            for p in result.look_points:
                rows.append({"key": f"look_{p.item}_{p.point}",
                             "label": f"{p.item} · {p.point}" + (" (close look)" if p.close_look else ""),
                             "verdict": p.verdict, "observed": p.observed})
    for r in result.rules:
        rows.append({"key": f"rule_{r.number}", "label": f"Your rule: {r.rule}",
                     "verdict": r.verdict, "observed": r.observed})
    return rows


def decide(result: CriticResult) -> str:
    """Any mismatch -> REGENERATE. Else any unclear -> REVIEW. Else ACCEPT.

    A mismatch found on a close look goes to the creator for review instead of a
    remake, because video models cannot reliably draw details like an exact time."""
    close = {f"rule_{r.number}" for r in result.rules if r.close_look}
    values = [v for k, v in verdicts(result).items() if k not in close]
    close_values = [v for k, v in verdicts(result).items() if k in close]
    if "mismatch" in values:
        return "REGENERATE"
    if "unclear" in values or close_values and set(close_values) != {"match"}:
        return "REVIEW"
    return "ACCEPT"


def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def apply_looks(result: CriticResult, canon: Canon) -> None:
    """A clear difference in any point of an item's look fails that item's check.

    The person's look counts towards wardrobe, a prop's towards prop_look. Points
    found on a close look add a problem with the close-ups attached."""
    owners = [(canon.character.name, "wardrobe", "WARDROBE",
               f"{canon.character.appearance}; wearing {canon.character.wardrobe}")]
    owners += [(p.name, "prop_look", "PROP_LOOK", p.description or p.name) for p in canon.props]
    for name, key, kind, expected in owners:
        bad = [p for p in result.look_points if _same(p.item, name) and p.verdict == "mismatch"]
        check = getattr(result, key)
        if not bad or check is None:
            continue
        summary = "; ".join(f"{p.point}: {p.observed}" for p in bad)
        if check.verdict != "mismatch":
            check.verdict, check.observed = "mismatch", summary
        close = [p for p in bad if p.close_look]
        if not close:
            continue
        existing = next((v for v in result.violations if v.type == kind), None)
        if existing is not None:
            existing.crop_uri = existing.crop_uri or close[0].crop_uri
            continue
        points = ", ".join(dict.fromkeys(p.point for p in close))
        result.violations.append(Violation(
            type=kind, headline=f"{name[0].upper()}{name[1:]} looks different: {points}"[:80],
            timestamp_seconds=close[0].timestamp_seconds, expected=expected,
            observed="; ".join(f"{p.point}: {p.observed}" for p in close),
            crop_uri=close[0].crop_uri))


def check_clip(
    clip_path: str,
    spec: ShotSpec,
    canon: Canon,
    previous_clip_path: Optional[str] = None,
    client=None,
    pictures: Optional[list] = None,
    references: Optional[dict] = None,
) -> CriticResult:
    """Watch one clip (and optionally the shot before it) and return the result."""
    client = client or llm.get_client()
    close_look.classify(canon.rules_for(spec.shot_id), ask=partial(llm.ask_json, client=client))
    media, labels = [], []
    for label, data, mime in pictures or []:
        media.append(llm.image_part(data, mime))
        labels.append(label)
    fps = get_settings().critic_fps
    if previous_clip_path:
        media.append(llm.video_part(llm.upload_video(previous_clip_path, client=client), fps))
    media.append(llm.video_part(llm.upload_video(clip_path, client=client), fps))

    result = llm.ask_json(
        purpose=f"critic:{spec.shot_id}",
        prompt=build_prompt(spec, canon, has_previous=bool(previous_clip_path),
                            picture_labels=labels),
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
    expected = watched_rules(spec, canon)
    by_number = {r.number: r for r in result.rules if 1 <= r.number <= len(expected)}
    result.rules = []
    for i, rule in enumerate(expected, start=1):
        got = by_number.get(i) or RuleCheck(number=i, verdict="unclear",
                                            observed="The critic did not report on this rule.")
        got.rule = rule.text
        got.close_look, got.crop_uri = False, None
        result.rules.append(got)
    # Rules about small details: sharp still frames, find the object, enlarge, read.
    close_ask = partial(llm.ask_json, client=client,
                        model=get_settings().video_understanding_model or None)
    for rule in (r for r in canon.rules_for(spec.shot_id) if r.close_look):
        number = len(result.rules) + 1
        try:
            result.rules.append(close_look.check_rule(clip_path, rule, number, ask=close_ask))
        except Exception as error:
            result.rules.append(RuleCheck(number=number, rule=rule.text, verdict="unclear",
                                          close_look=True,
                                          observed=f"The close look failed: {error}"[:200]))
    # How the person and props look, point by point, on sharp close-ups.
    known = {canon.character.name.lower()} | {p.name.lower() for p in canon.props}
    result.look_points = [p for p in result.look_points if p.item.strip().lower() in known]
    for p in result.look_points:
        p.close_look, p.crop_uri = False, None
    try:
        result.look_points += looks.check(clip_path, canon, ask=close_ask, references=references)
    except Exception:
        pass   # no close-ups (for example the clip could not be read); the video checks stand
    apply_looks(result, canon)
    result.observations = sorted(result.observations, key=lambda o: o.severity != "major")[:3]
    for o in result.observations:
        o.timestamp_seconds = max(0.0, o.timestamp_seconds)
    for v in result.violations:
        v.timestamp_seconds = max(0.0, v.timestamp_seconds)
    result.decision = decide(result)  # the model's own decision is ignored
    return result
