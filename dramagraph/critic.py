"""The continuity critic: Gemini says what it sees, code makes the decision."""
from pathlib import Path
from typing import Optional

from . import llm
from .config import get_settings
from .models import Canon, CriticResult, ShotSpec

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "critic.txt"

SECOND_PERSON_NOTE = (
    "A second person's hands may appear. Hands alone are not an extra person.\n"
)
CROSS_SHOT_NOTE = (
    "\nTwo videos are attached. The FIRST is the previous shot and is only for\n"
    "comparison. The SECOND is the shot to check; all verdicts above are about\n"
    "the second video. Also give a cross_shot verdict: is it the same person,\n"
    "in the same clothes and the same place as the previous shot?\n"
)
NO_CROSS_SHOT_NOTE = "\nSet cross_shot to null.\n"


def build_prompt(spec: ShotSpec, canon: Canon, has_previous: bool = False) -> str:
    c = canon.character
    character = f"{c.name}, adult. {c.appearance}. Wearing {c.wardrobe}."
    if c.accessories:
        character += " Accessories: " + ", ".join(c.accessories) + "."

    prop_lines = []
    for p in canon.props:
        start, end = p.states_for(spec.shot_id)
        line = f"{p.name} ({p.description})" if p.description else p.name
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
    )


def verdicts(result: CriticResult) -> dict[str, str]:
    out = {
        "wardrobe": result.wardrobe.verdict,
        "prop_state": result.prop_state.verdict,
        "scene": result.scene.verdict,
    }
    if result.cross_shot is not None:
        out["cross_shot"] = result.cross_shot.verdict
    return out


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
    for v in result.violations:
        v.timestamp_seconds = max(0.0, v.timestamp_seconds)
    result.decision = decide(result)  # the model's own decision is ignored
    return result
