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
    return ledger


def reserve(seconds: int, note: str = "") -> bool:
    """Take seconds from the budget before a Veo call. False means stop."""
    cap = get_settings().veo_budget_seconds

    def change(ledger):
        ledger = ledger or {"used": 0, "entries": []}
        if ledger["used"] + seconds > cap:
            return None, False                     # nothing written
        return _entry(ledger, seconds, note), True

    # atomic, so two people making video at the same moment cannot both take the last seconds
    return storage.update_json(LEDGER, change)


def refund(seconds: int, note: str = "") -> None:
    """Give seconds back. Only for requests Veo rejected before starting any work."""
    def change(ledger):
        ledger = ledger or {"used": 0, "entries": []}
        return _entry(ledger, -min(seconds, ledger["used"]), "refund: " + note), None

    storage.update_json(LEDGER, change)
