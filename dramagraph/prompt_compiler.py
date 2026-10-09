"""Shot spec + canon -> the exact text sent to Veo. A template, never a model call."""
from typing import Optional

from .models import Canon, SeededError, ShotSpec


def compile_prompt(spec: ShotSpec, canon: Canon, fix_notes: str = "",
                   seed: Optional[SeededError] = None) -> str:
    c = canon.character
    wardrobe = c.wardrobe
    if seed and seed.kind == "wardrobe":
        wardrobe = seed.value
    lines = [
        f"SHOT: {spec.action}",
        f"CHARACTER: {c.name}, adult. {c.appearance}. Wearing {wardrobe}."
        + (f" {', '.join(c.accessories)}." if c.accessories else ""),
    ]
    for p in canon.props:
        state = p.state_by_shot.get(spec.shot_id, "")
        if seed and seed.kind == "prop":
            state = seed.value
        text = f"PROP: {p.name}. {p.description}".strip()
        if state:
            text += f" In this shot it is: {state}."
        lines.append(text)
    lines.append(f"LOCATION: {canon.location}, {canon.time_of_day}.")
    lines.append(f"CAMERA: vertical 9:16, {spec.shot_type}, {spec.camera_motion}. One continuous take.")
    if spec.second_person:
        lines.append("A second person appears as hands only, never a face or body.")
    lines.append("DO NOT SHOW: other people's faces or bodies, text on screen, camera cuts.")
    if fix_notes.strip():
        lines.append(fix_notes.strip())
    return "\n".join(lines)
