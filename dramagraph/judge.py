"""Judge mode: the public demo cannot run up the owner's API bill.

The app looks and works exactly as normal. Behind it, with JUDGE_MODE on:
- Every Gemini request uses one unit of a daily allowance shared by all visitors
  (JUDGE_GEMINI_CALLS_PER_DAY). The unit is taken before the request is sent, and the
  count is updated atomically, so it is a hard cap on the number of requests.
- Making video with Veo is switched off.
- The episodes listed in JUDGE_EPISODES are shown but cannot be changed, so every
  visitor sees the demo exactly as it was made.
"""
from datetime import datetime, timezone

from .config import get_settings

LEDGER = "judge/gemini_calls.json"
OUT_OF_ALLOWANCE = ("Today's free AI checks for this public demo are used up. Please try "
                    "again tomorrow; the demo video shows everything working.")
NO_VIDEO = ("Making new video is switched off in this public demo to protect API costs. "
            "The demo video shows it working.")
PROTECTED = ("This is the demo episode, kept exactly as it was made, so it can't be changed "
             "here. Start a new episode or use Check my own clips to try things yourself.")


def on() -> bool:
    return get_settings().judge_mode


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def used_today() -> int:
    import json

    from . import storage

    if not storage.exists(LEDGER):
        return 0
    return int(json.loads(storage.load_bytes(LEDGER)).get(_today(), 0))


def remaining() -> int:
    return max(0, get_settings().judge_gemini_calls_per_day - used_today())


def spend(units: int = 1) -> bool:
    """Take units from today's allowance, all or nothing. False if not enough left."""
    from . import storage

    cap, today = get_settings().judge_gemini_calls_per_day, _today()

    def change(ledger):
        used = int((ledger or {}).get(today, 0))
        if units <= 0 or used + units > cap:
            return None, False                     # nothing written
        return {today: used + units}, True         # only today is kept

    return storage.update_json(LEDGER, change)


def before_gemini_request() -> None:
    """Called right before every request sent to Gemini."""
    if on() and not spend(1):
        raise RuntimeError(OUT_OF_ALLOWANCE)


def before_video() -> None:
    """Called before any Veo request."""
    if on():
        raise RuntimeError(NO_VIDEO)


def protected(path: str) -> bool:
    """True for files of a demo episode while judge mode is on."""
    if not on():
        return False
    return any(path.startswith(f"episodes/{i}/") for i in get_settings().judge_episodes)
