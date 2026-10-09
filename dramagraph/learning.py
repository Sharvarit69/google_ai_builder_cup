"""Layer 3: checks that come from people.

When a creator spots a problem the critic missed, their note becomes a rule the
critic checks from then on. Rules that apply to any video are also kept in a
shared library, so new episodes start with what earlier ones taught.
"""
import json
from pathlib import Path

from pydantic import BaseModel

from . import llm, storage
from .models import Canon, LearnedRule, ShotSpec

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
LIBRARY = "rules/library.json"
MAX_RULES = 8  # per episode; more than this and the critic's attention is spread thin


class RuleDraft(BaseModel):
    text: str
    applies_to_all_shots: bool = False


def learn_rule(note: str, spec: ShotSpec, source: str, ask=llm.ask_json) -> LearnedRule:
    """Turn a creator's note into a checkable rule. Falls back to their own words."""
    note = note.strip()
    try:
        prompt = (PROMPTS / "learn_rule.txt").read_text(encoding="utf-8").format(
            action=spec.action, note=note)
        draft = ask(purpose="learn_rule", prompt=prompt, schema=RuleDraft)
        text, general = draft.text.strip() or note, draft.applies_to_all_shots
    except Exception:
        text, general = note, False
    return LearnedRule(text=text, shot_ids=[] if general else [spec.shot_id],
                       source=source, from_note=note)


def add_rule(canon: Canon, rule: LearnedRule) -> bool:
    """Add to the episode unless it is already there. Returns True if added."""
    if any(r.text.strip().lower() == rule.text.strip().lower() for r in canon.rules):
        return False
    if len(canon.rules) >= MAX_RULES:
        raise ValueError(f"An episode can hold {MAX_RULES} rules. Remove one in section 3 first.")
    canon.rules.append(rule)
    if not rule.shot_ids:
        remember(rule)
    return True


def library() -> list[LearnedRule]:
    if not storage.exists(LIBRARY):
        return []
    return [LearnedRule.model_validate(r) for r in json.loads(storage.load_bytes(LIBRARY))]


def _save_library(rules: list[LearnedRule]) -> None:
    data = json.dumps([r.model_dump() for r in rules], indent=2).encode("utf-8")
    storage.save_bytes(LIBRARY, data)


def remember(rule: LearnedRule) -> None:
    """Keep a general rule for future episodes."""
    rules = library()
    if any(r.text.strip().lower() == rule.text.strip().lower() for r in rules):
        return
    rules.append(LearnedRule(text=rule.text, source="library", from_note=rule.from_note))
    _save_library(rules[-MAX_RULES:])


def forget(text: str) -> None:
    _save_library([r for r in library() if r.text.strip().lower() != text.strip().lower()])


def starting_rules() -> list[LearnedRule]:
    """Rules a new episode begins with: everything general learned so far."""
    return [r.model_copy() for r in library()]
