"""The Veo spending guard: a ledger of generated seconds with a hard cap."""
import json
from datetime import datetime, timezone

from . import storage
from .config import get_settings

LEDGER = "ration/ledger.json"


def _load() -> dict:
    if storage.exists(LEDGER):
        return json.loads(storage.load_bytes(LEDGER))
    return {"used": 0, "entries": []}


def _save(ledger: dict) -> None:
    storage.save_bytes(LEDGER, json.dumps(ledger, indent=2).encode("utf-8"))


def used() -> int:
    return _load()["used"]


def remaining() -> int:
    return max(0, get_settings().veo_budget_seconds - used())


def _entry(ledger, seconds, note):
    ledger["used"] += seconds
    ledger["entries"].append(
        {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "seconds": seconds, "note": note}
    )
    _save(ledger)


def reserve(seconds: int, note: str = "") -> bool:
    """Take seconds from the budget before a Veo call. False means stop."""
    ledger = _load()
    if ledger["used"] + seconds > get_settings().veo_budget_seconds:
        return False
    _entry(ledger, seconds, note)
    return True


def refund(seconds: int, note: str = "") -> None:
    """Give seconds back. Only for requests Veo rejected before starting any work."""
    ledger = _load()
    _entry(ledger, -min(seconds, ledger["used"]), "refund: " + note)
