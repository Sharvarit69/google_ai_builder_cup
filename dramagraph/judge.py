"""Judge mode: a public demo that cannot run up the owner's API bill.

With JUDGE_MODE on, visitors can browse saved episodes, and can try "Check my own
clips" a limited number of times a day for all visitors together
(JUDGE_CHECKS_PER_DAY). Everything else that would call Gemini or Veo, or save
anything, is refused at the source: llm.get_client and storage.save_bytes only work
inside an allowance that a reserved check has opened.

The daily count is kept in Cloud Storage and updated atomically, so two visitors at
the same moment cannot both take the last check. A check is counted before it
starts and is not given back if it fails.
"""
import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import get_settings

LEDGER = "judge/checks.json"
_open = contextvars.ContextVar("judge_allowance", default=False)


def active() -> bool:
    """True while a reserved check is running in this session."""
    return _open.get()


@contextmanager
def opened():
    token = _open.set(True)
    try:
        yield
    finally:
        _open.reset(token)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def used_today() -> int:
    from . import storage
    import json

    if not storage.exists(LEDGER):
        return 0
    return int(json.loads(storage.load_bytes(LEDGER)).get(_today(), 0))


def remaining() -> int:
    return max(0, get_settings().judge_checks_per_day - used_today())


def reserve(clips: int) -> bool:
    """Take `clips` checks from today's allowance, all or nothing. False if not enough left."""
    from . import storage

    cap, today = get_settings().judge_checks_per_day, _today()

    def change(ledger):
        ledger = dict(ledger or {})
        used = int(ledger.get(today, 0))
        if clips <= 0 or used + clips > cap:
            return None, False                     # nothing written
        ledger = {today: used + clips}             # keep only today
        return ledger, True

    return storage.update_json(LEDGER, change)
