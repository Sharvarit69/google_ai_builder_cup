"""One wrapper for every Gemini call: JSON out, checked, retried and logged."""
import json
import time
from datetime import datetime, timezone
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import storage
from .config import get_settings

T = TypeVar("T", bound=BaseModel)

RETRY_WAITS = [2, 4, 8]  # seconds, for rate limits and server errors


def get_client():
    from google import genai  # imported only when a real call is made

    key = get_settings().gemini_api_key
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=key)


def _log(purpose: str, model: str, seconds: float, ok: bool, note: str = "") -> None:
    line = json.dumps(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "model": model,
            "seconds": round(seconds, 2),
            "ok": ok,
            "note": note[:300],
        }
    )
    path = f"logs/{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    try:
        old = storage.load_bytes(path) if storage.exists(path) else b""
        storage.save_bytes(path, old + line.encode("utf-8") + b"\n")
    except Exception:
        pass  # logging must never break a real call


def _is_temporary(error: Exception) -> bool:
    text = str(error)
    code = getattr(error, "code", None)
    return code in (429, 500, 502, 503, 504) or any(
        s in text for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def ask_json(
    purpose: str,
    prompt: str,
    schema: Type[T],
    media: Optional[list] = None,
    model: Optional[str] = None,
    client=None,
    sleep=time.sleep,
) -> T:
    """Send a prompt (plus optional uploaded files) and return a validated object.

    The required JSON shape is added to the prompt and the reply is checked
    against `schema`. A reply that fails the check is retried once with the
    error shown to the model.
    """
    settings = get_settings()
    model = model or settings.text_model
    if not model:
        raise RuntimeError("No model name given. Set TEXT_MODEL.")
    client = client or get_client()

    shape = json.dumps(schema.model_json_schema())
    full_prompt = (
        f"{prompt}\n\nReturn only JSON that fits this JSON Schema, "
        f"with no extra text:\n{shape}"
    )
    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0
        )
    except ImportError:  # only happens in tests that pass a fake client
        config = None

    correction = ""
    for validation_try in range(2):
        started = time.time()
        reply_text = None
        for wait in RETRY_WAITS + [None]:
            try:
                reply = client.models.generate_content(
                    model=model,
                    contents=list(media or []) + [full_prompt + correction],
                    config=config,
                )
                reply_text = reply.text or ""
                break
            except Exception as error:
                if wait is None or not _is_temporary(error):
                    _log(purpose, model, time.time() - started, False, str(error))
                    raise
                sleep(wait)
        try:
            result = schema.model_validate_json(_strip_fences(reply_text))
            _log(purpose, model, time.time() - started, True)
            return result
        except ValidationError as error:
            _log(purpose, model, time.time() - started, False, str(error))
            if validation_try == 1:
                raise
            correction = (
                "\n\nYour previous reply was rejected for this reason. "
                f"Fix it and reply again:\n{error}"
            )
    raise RuntimeError("unreachable")


def upload_video(path: str, client=None, sleep=time.sleep, timeout: int = 300):
    """Upload a clip and wait until Gemini has finished processing it."""
    client = client or get_client()
    handle = client.files.upload(file=path)
    waited = 0
    while handle.state.name == "PROCESSING":
        if waited >= timeout:
            raise TimeoutError(f"Video still processing after {timeout}s: {path}")
        sleep(5)
        waited += 5
        handle = client.files.get(name=handle.name)
    if handle.state.name != "ACTIVE":
        raise RuntimeError(f"Video upload failed ({handle.state.name}): {path}")
    return handle
