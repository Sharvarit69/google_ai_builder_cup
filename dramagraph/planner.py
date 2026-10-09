"""Story beats -> one shot specification per beat."""
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from . import llm
from .config import get_settings
from .models import Beat, Canon, ShotSpec

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
MAX_SHOTS = 4


class ShotDraft(BaseModel):
    shot_type: str
    camera_motion: str
    action: str
    second_person_hands: bool = False


class PlanDraft(BaseModel):
    shots: list[ShotDraft]


def describe_canon(canon: Canon) -> str:
    c = canon.character
    lines = [f"Character: {c.name}, adult. {c.appearance}. Wearing {c.wardrobe}."]
    lines += [f"Prop: {p.name}. {p.description}".strip() for p in canon.props]
    lines.append(f"Location: {canon.location}, {canon.time_of_day}.")
    return "\n".join(lines)


def apply_rules(specs: list[ShotSpec], canon: Canon) -> list[ShotSpec]:
    """Continuity constraints are set by code, so no instruction can remove them."""
    seconds = get_settings().shot_seconds
    for spec in specs:
        must = [f"{canon.character.name} wearing {canon.character.wardrobe}"]
        for p in canon.props:
            state = p.state_by_shot.get(spec.shot_id)
            if state:
                must.append(f"{p.name}: {state}")
        must.append(f"{canon.location}, {canon.time_of_day}")
        if spec.second_person:
            must.append("a second person appears as hands only")
        spec.must_show = must
        spec.must_not_show = ["a second person's face or body", "text on screen"]
        spec.duration_seconds = seconds
    return specs


def _plan(picked: list[Beat], canon: Canon, instruction: str, ask) -> list[ShotSpec]:
    if not picked:
        raise ValueError("Pick at least one beat.")
    if len(picked) > MAX_SHOTS:
        raise ValueError(f"Pick at most {MAX_SHOTS} beats.")
    if any(b.needs_second_person for b in picked):
        raise ValueError("Beats that need a second person fully on screen are not supported.")
    beats = "\n".join(f"Beat {i + 1}: {b.action}" for i, b in enumerate(picked))
    prompt = (PROMPTS / "plan_shots.txt").read_text(encoding="utf-8").format(
        seconds=get_settings().shot_seconds, canon=describe_canon(canon), beats=beats,
        instruction=instruction,
    )
    draft = ask(purpose="plan_shots", prompt=prompt, schema=PlanDraft)
    if len(draft.shots) != len(picked):
        raise ValueError(
            f"The planner returned {len(draft.shots)} shots for {len(picked)} beats. Try again."
        )
    specs = [
        ShotSpec(
            shot_id=f"S{i + 1}", beat_id=beat.beat_id, shot_type=d.shot_type,
            camera_motion=d.camera_motion, action=d.action,
            second_person="hands only" if d.second_person_hands else None,
        )
        for i, (beat, d) in enumerate(zip(picked, draft.shots))
    ]
    return apply_rules(specs, canon)


def plan_shots(picked: list[Beat], canon: Canon, ask=llm.ask_json) -> list[ShotSpec]:
    return _plan(picked, canon, "", ask)


def replan(picked: list[Beat], canon: Canon, current: list[ShotSpec], instruction: str,
           ask=llm.ask_json) -> list[ShotSpec]:
    if not instruction.strip():
        raise ValueError("Type what you want changed.")
    now = "\n".join(
        f"Shot {i + 1}: {s.shot_type}, {s.camera_motion}. {s.action}"
        for i, s in enumerate(current)
    )
    extra = (
        f"\nCURRENT PLAN\n{now}\n\nThe creator wants this change: {instruction.strip()}\n"
        "Return the full plan again with that change applied. Keep everything else.\n"
    )
    return _plan(picked, canon, extra, ask)
