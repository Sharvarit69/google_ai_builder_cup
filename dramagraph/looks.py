"""Keep the person and props looking the same in every shot.

Two jobs, both on sharp still frames rather than the low-detail video:

1. Lock the look. When a shot first passes, a sharp close-up of the person and
   of each prop is cut from it and kept as that item's approved look. Later
   shots and repairs send it to Veo, and the critic compares against it. A
   picture the creator added always wins over an approved look.
2. Close look. For every clip, close-ups of the person and props are compared
   point by point (shape, parts, size, finish, colour; face, clothes) with the
   approved look or the creator's picture, and with the written description.

Nothing here knows about any particular story: the items come from the canon.
"""
import hashlib
import io
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from . import close_look, llm, pictures, storage
from .models import Canon, Episode, LookPoint, RefPicture, Verdict

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
PERSON = "person"


class ItemBoxes(BaseModel):
    key: str
    boxes: list[Optional[list[float]]] = []


class Found(BaseModel):
    items: list[ItemBoxes] = []


class Point(BaseModel):
    key: str
    point: str
    verdict: Verdict = "unclear"
    observed: str = ""
    picture: int = 0


class Compared(BaseModel):
    points: list[Point] = []


def items(canon: Canon) -> list[tuple[str, str, str]]:
    """(key, owner, what it looks like) for the person and every prop."""
    c = canon.character
    found = [(PERSON, "character", f"{c.name}: {c.appearance}; wearing {c.wardrobe}")]
    for i, prop in enumerate(canon.props, start=1):
        found.append((f"object {i}", prop.name, f"{prop.name}: {prop.description or prop.name}"))
    return found


def find(stills, wanted, ask) -> dict[str, list]:
    """For each item key, one box (or None) per still frame."""
    times = ", ".join(close_look._seconds(at) for at, _ in stills)
    listing = "\n".join(f'- "{key}": {what}' for key, _, what in wanted)
    prompt = (PROMPTS / "looks_find.txt").read_text(encoding="utf-8").format(
        count=len(stills), times=times, items=listing)
    found = ask(purpose="looks_find", prompt=prompt, schema=Found,
                media=[llm.image_part(png) for _, png in stills])
    return {f.key: f.boxes for f in found.items}


def _crops(stills, boxes) -> list[tuple[float, object]]:
    out = []
    for (at, png), box in zip(stills, boxes or []):
        if box and len(box) == 4:
            image = close_look.crop(png, box)
            if image is not None:
                out.append((at, image))
    return out


def _sheet(panels: list[tuple[str, object]], highlight: Optional[int] = None) -> bytes:
    """Pictures side by side, each with a caption above it."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=22)
    height = max(img.height for _, img in panels)
    width = sum(img.width for _, img in panels) + 12 * (len(panels) - 1)
    sheet = Image.new("RGB", (width, height + 40), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for i, (caption, img) in enumerate(panels):
        sheet.paste(img, (x, 40))
        draw.text((x + 4, 8), caption, fill="black", font=font)
        if i == highlight:
            draw.rectangle((x, 40, x + img.width - 1, 39 + img.height), outline=(194, 65, 12), width=5)
        x += img.width + 12
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


def _fit(png: bytes):
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    scale = close_look.CROP_SIDE / max(image.size)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))


# ---------- 1. lock the look ----------

def _missing(episode: Episode) -> list[tuple[str, str, str]]:
    """Items with no picture and no approved look yet."""
    canon = episode.canon
    found = []
    for key, owner, what in items(canon):
        item = canon.character if owner == "character" else pictures._owner(canon, owner)
        if item.auto_look and pictures.reference(episode, owner) is None:
            found.append((key, owner, what))
    return found


def lock(episode: Episode, ask=llm.ask_json) -> list[str]:
    """Take approved looks from the first shot that passed. Returns the owners locked."""
    if episode.canon is None:
        return []
    wanted = _missing(episode)
    if not wanted:
        return []
    shot = next((s for s in episode.shots if s.status == "ACCEPTED"), None)
    attempt = next((a for a in shot.attempts if a.attempt_no == shot.chosen_attempt), None) if shot else None
    if attempt is None or not attempt.clip_uri or not attempt.critic or attempt.critic.decision != "ACCEPT":
        return []
    stills = close_look.frames(storage.local_copy(attempt.clip_uri))
    boxes = find(stills, wanted, ask)
    locked = []
    for key, owner, _ in wanted:
        crops = _crops(stills, boxes.get(key))
        if not crops:
            continue
        _, image = crops[len(crops) // 2]          # the middle of the shot, if it was found there
        uri = f"episodes/{episode.episode_id}/pictures/approved-{key.replace(' ', '-')}.png"
        storage.save_bytes(uri, close_look._png(image))
        picture = RefPicture(uri=uri, label=f"approved look · shot {shot.spec.shot_id[1:]}")
        item = episode.canon.character if owner == "character" else pictures._owner(episode.canon, owner)
        item.approved = picture
        locked.append(owner)
    if locked:
        storage.save_episode(episode)
    return locked


# ---------- 2. close look at every clip ----------

def check(clip_path: str, episode_or_canon, ask=llm.ask_json,
          references: Optional[dict] = None) -> list[LookPoint]:
    """Point-by-point look of the person and props on sharp close-ups.

    references maps an owner ("character" or a prop name) to (caption, png bytes).
    Returns [] if no close-ups could be made; the video-level checks still stand."""
    canon = episode_or_canon.canon if isinstance(episode_or_canon, Episode) else episode_or_canon
    references = references or {}
    wanted = items(canon)
    stills = close_look.frames(clip_path)
    boxes = find(stills, wanted, ask)

    media, legend, panels_for = [], [], {}
    for key, owner, _ in wanted:
        panels = []
        label = f"{key} ({canon.character.name if owner == 'character' else owner})"
        if owner in references:
            caption, png = references[owner]
            media.append(llm.image_part(png))
            legend.append(f"Picture {len(media)}: {label}, {caption} (the reference)")
            panels.append((caption, _fit(png)))
        numbers = {}
        for at, image in _crops(stills, boxes.get(key)):
            media.append(llm.image_part(close_look._png(image)))
            numbers[len(media)] = at
            legend.append(f"Picture {len(media)}: {label}, close-up from the clip at {close_look._seconds(at)}")
            panels.append((f"clip at {close_look._seconds(at)}", image))
        if numbers:
            panels_for[key] = (panels, numbers)
    if not panels_for:
        return []

    listing = "\n".join(f'- "{key}": {what}' for key, _, what in wanted if key in panels_for)
    prompt = (PROMPTS / "looks_compare.txt").read_text(encoding="utf-8").format(
        legend="\n".join(legend), items=listing)
    compared = ask(purpose="looks_compare", prompt=prompt, schema=Compared, media=media)

    owners = {key: owner for key, owner, _ in wanted}
    with open(clip_path, "rb") as f:
        clip_digest = hashlib.sha1(f.read()).hexdigest()[:12]
    sheets: dict[str, str] = {}
    out = []
    for p in compared.points:
        if p.key not in panels_for:
            continue
        panels, numbers = panels_for[p.key]
        at = numbers.get(p.picture, next(iter(numbers.values())))
        if p.key not in sheets:
            uri = f"closeups/look-{clip_digest}-{p.key.replace(' ', '-')}.png"
            sheets[p.key] = storage.save_bytes(uri, _sheet(panels))
        name = canon.character.name if owners[p.key] == "character" else owners[p.key]
        out.append(LookPoint(item=name, point=p.point.strip().lower(), verdict=p.verdict,
                             observed=p.observed.strip(), close_look=True,
                             crop_uri=sheets[p.key], timestamp_seconds=at))
    return out
