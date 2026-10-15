"""A close look for rules about small details.

The critic watches a whole clip at low detail, about one frame a second, so a
clock face or a line of writing is only a few blurry pixels. For rules that
depend on such details, this takes sharp still frames from the clip, asks
Gemini where the object is, enlarges that part, and asks Gemini to read it step
by step. If the reading is not certain the answer is "unclear", never a guess.
"""
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from . import llm, storage
from .models import LearnedRule, RuleCheck, Verdict

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
FRACTIONS = (0.15, 0.5, 0.9)   # where in the clip the still frames are taken
CROP_SIDE = 640                # enlarged close-ups are this many pixels on the long side
PADDING = 0.25                 # extra margin around the box, as a share of its size


class RuleKind(BaseModel):
    close_look: bool = False


class Found(BaseModel):
    boxes: list[Optional[list[float]]] = []


class Reading(BaseModel):
    steps: str = ""
    reading: str = ""
    confident: bool = False
    verdict: Verdict = "unclear"
    picture: int = 1


def classify(rules: list[LearnedRule], ask=llm.ask_json) -> None:
    """Decide, once per rule, whether it needs a close look. A failure leaves it for next time."""
    for rule in rules:
        if rule.close_look is not None:
            continue
        try:
            prompt = (PROMPTS / "rule_kind.txt").read_text(encoding="utf-8").format(rule=rule.text)
            rule.close_look = ask(purpose="rule_kind", prompt=prompt, schema=RuleKind).close_look
        except Exception:
            pass


def _seconds(text: float) -> str:
    return f"0:{int(text):02d}"


def frames(clip_path: str) -> list[tuple[float, bytes]]:
    """Sharp still frames (seconds, PNG bytes) from the start, middle and end of the clip."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "json", clip_path], capture_output=True, text=True, check=True)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    out = []
    for share in FRACTIONS:
        at = round(duration * share, 2)
        grab = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(at), "-i", clip_path,
                               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                              capture_output=True, check=True)
        if grab.stdout:
            out.append((at, grab.stdout))
    if not out:
        raise RuntimeError("no frames")
    return out


def crop(png: bytes, box: list[float]):
    """Cut the box (0-1000 scale, ymin xmin ymax xmax) out of a frame, with a margin, enlarged."""
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = image.size
    ymin, xmin, ymax, xmax = (max(0.0, min(1000.0, float(v))) / 1000 for v in box)
    if ymax <= ymin or xmax <= xmin:
        return None
    pad_y, pad_x = (ymax - ymin) * PADDING, (xmax - xmin) * PADDING
    left, top = int(max(0, xmin - pad_x) * w), int(max(0, ymin - pad_y) * h)
    right, bottom = int(min(1, xmax + pad_x) * w), int(min(1, ymax + pad_y) * h)
    if right - left < 4 or bottom - top < 4:
        return None
    part = image.crop((left, top, right, bottom))
    scale = CROP_SIDE / max(part.size)
    return part.resize((max(1, int(part.width * scale)), max(1, int(part.height * scale))),
                       Image.LANCZOS)


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _strip(crops: list[tuple[float, object]], best: int) -> bytes:
    """The close-ups side by side, each labelled with its time, the best one outlined."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=22)
    height = max(img.height for _, img in crops)
    width = sum(img.width for _, img in crops) + 12 * (len(crops) - 1)
    sheet = Image.new("RGB", (width, height + 40), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for i, (at, img) in enumerate(crops):
        sheet.paste(img, (x, 40))
        draw.text((x + 4, 8), f"at {_seconds(at)}" + ("  · read from this one" if i == best else ""),
                  fill="black", font=font)
        if i == best:
            draw.rectangle((x, 40, x + img.width - 1, 39 + img.height), outline=(194, 65, 12), width=5)
        x += img.width + 12
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


def check_rule(clip_path: str, rule: LearnedRule, number: int, ask=llm.ask_json) -> RuleCheck:
    result = RuleCheck(number=number, rule=rule.text, verdict="unclear", close_look=True)
    try:
        stills = frames(clip_path)
    except Exception:
        result.observed = "Could not take still frames from the clip for a close look"
        return result
    times = ", ".join(_seconds(at) for at, _ in stills)
    prompt = (PROMPTS / "close_look_find.txt").read_text(encoding="utf-8").format(
        count=len(stills), times=times, rule=rule.text)
    found = ask(purpose="close_look_find", prompt=prompt, schema=Found,
                media=[llm.image_part(png) for _, png in stills])
    crops = []
    for (at, png), box in zip(stills, found.boxes):
        if box and len(box) == 4:
            image = crop(png, box)
            if image is not None:
                crops.append((at, image))
    if not crops:
        result.observed = "Could not find what this rule is about in the clip"
        return result

    pictures = " ".join(f"Close-up {i + 1} is from {_seconds(at)}." for i, (at, _) in enumerate(crops))
    prompt = (PROMPTS / "close_look_read.txt").read_text(encoding="utf-8").format(
        pictures=pictures, rule=rule.text)
    reading = ask(purpose="close_look_read", prompt=prompt, schema=Reading,
                  media=[llm.image_part(_png(img)) for _, img in crops])
    best = min(max(reading.picture, 1), len(crops)) - 1
    result.verdict = reading.verdict if reading.confident else "unclear"
    result.timestamp_seconds = crops[best][0]
    seen = reading.reading.strip().rstrip(".")
    if result.verdict == "unclear":
        result.observed = "Could not read it clearly" + (f"; it may show {seen}" if seen else "")
    else:
        result.observed = f"Close look shows {seen}" if seen else reading.steps.strip()[:200]

    data = _strip(crops, best)
    with open(clip_path, "rb") as f:
        digest = hashlib.sha1(f.read() + rule.text.encode("utf-8")).hexdigest()[:16]
    result.crop_uri = storage.save_bytes(f"closeups/{digest}.png", data)
    return result
