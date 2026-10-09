"""Script text -> numbered lines and story beats."""
from pathlib import Path

from pydantic import BaseModel

from . import llm
from .models import Beat

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
MAX_LINES = 60


class BeatList(BaseModel):
    beats: list[Beat]


def split_lines(script_text: str) -> list[str]:
    return [line.strip() for line in script_text.splitlines() if line.strip()]


def check_script(lines: list[str]) -> None:
    if not lines:
        raise ValueError("The script is empty.")
    if len(lines) > MAX_LINES:
        raise ValueError(f"The script has {len(lines)} lines; the limit is {MAX_LINES}.")
    letters = [c for c in " ".join(lines) if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < 0.8:
        raise ValueError("Only English scripts are supported for now.")


def parse_script(script_text: str, ask=llm.ask_json) -> tuple[list[str], list[Beat]]:
    lines = split_lines(script_text)
    check_script(lines)
    numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    prompt = (PROMPTS / "parse_script.txt").read_text(encoding="utf-8").format(script=numbered)
    reply = ask(purpose="parse_script", prompt=prompt, schema=BeatList)

    beats = []
    for i, beat in enumerate(reply.beats):
        beat.beat_id = f"B{i + 1}"
        beat.script_lines = [n for n in beat.script_lines if 1 <= n <= len(lines)]
        beat.caption = " ".join(beat.caption.split()[:12])
        beats.append(beat)
    if not beats:
        raise ValueError("No story beats were found in the script.")
    return lines, beats
