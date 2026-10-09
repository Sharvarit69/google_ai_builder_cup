"""Character, prop and setting details that must stay the same across shots."""
from pathlib import Path

from pydantic import BaseModel, Field

from . import llm
from .models import Beat, Canon, Character, Prop

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PROPS = 2  # more than this and no video model can satisfy every check


class PropDraft(BaseModel):
    name: str
    description: str = ""
    states: list[str] = Field(default_factory=list)  # one per shot, in order


class CanonDraft(BaseModel):
    name: str
    appearance: str
    wardrobe: str
    accessories: list[str] = Field(default_factory=list)
    props: list[PropDraft] = Field(default_factory=list)
    location: str
    time_of_day: str


def build_canon(script_lines: list[str], picked: list[Beat], ask=llm.ask_json) -> Canon:
    beats = "\n".join(f"Shot {i + 1}: {b.action}" for i, b in enumerate(picked))
    prompt = (PROMPTS / "build_canon.txt").read_text(encoding="utf-8").format(
        script="\n".join(script_lines), beats=beats
    )
    d = ask(purpose="build_canon", prompt=prompt, schema=CanonDraft)
    props = []
    for p in d.props[:MAX_PROPS]:
        states = {}
        for i in range(len(picked)):
            # if the model gave too few states, carry the last one forward
            state = p.states[i] if i < len(p.states) else (p.states[-1] if p.states else "")
            if state.strip():
                states[f"S{i + 1}"] = state.strip()
        props.append(Prop(name=p.name, description=p.description, state_by_shot=states))
    return Canon(
        character=Character(name=d.name, appearance=d.appearance, wardrobe=d.wardrobe,
                            accessories=d.accessories),
        props=props, location=d.location, time_of_day=d.time_of_day,
    )


def validate_reference(image_bytes: bytes, file_name: str, consent_ticked: bool) -> None:
    if not consent_ticked:
        raise ValueError(
            "Tick the box confirming the person is an adult who has given permission."
        )
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        raise ValueError("The reference picture must be a PNG or JPG file.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("The reference picture must be under 10 MB.")
