"""Reference pictures of the character and props.

Each character or prop can hold up to three labelled pictures. Veo accepts at
most three pictures per shot in total, so for each shot this module picks the
most useful three: one of the character, then for each prop the picture whose
label matches the prop's state in that shot, then spare character angles. The
critic is shown every picture.

When there is no picture of an item, its approved look is used instead: a sharp
still taken from the first shot that passed (see looks.py). A picture the
creator added always comes first.
"""
import uuid
from pathlib import Path
from typing import Optional

from . import storage
from .models import Canon, Episode, RefPicture, ShotSpec

MAX_PER_ITEM = 3
MAX_FOR_VEO = 3
MAX_FOR_CRITIC = 6
MAX_BYTES = 10 * 1024 * 1024
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
LEGACY = "episodes/{episode_id}/reference.png"


def _owner(canon: Canon, owner: str):
    """owner is "character" or a prop name."""
    if owner == "character":
        return canon.character
    for prop in canon.props:
        if prop.name == owner:
            return prop
    raise ValueError(f"There is no prop called {owner!r}.")


def add_picture(episode: Episode, owner: str, data: bytes, file_name: str, label: str,
                consent: bool = True) -> RefPicture:
    if episode.canon is None:
        raise ValueError("Plan the episode first.")
    suffix = Path(file_name).suffix.lower()
    if suffix not in MIME:
        raise ValueError("Pictures must be PNG or JPG files.")
    if len(data) > MAX_BYTES:
        raise ValueError("Pictures must be under 10 MB.")
    if owner == "character" and not consent:
        raise ValueError("Tick the box confirming the picture is AI-generated, or shows an "
                         "adult who gave permission.")
    item = _owner(episode.canon, owner)
    if len(item.pictures) >= MAX_PER_ITEM:
        raise ValueError(f"Each character or prop can have {MAX_PER_ITEM} pictures. Remove one first.")
    uri = f"episodes/{episode.episode_id}/pictures/{uuid.uuid4().hex[:8]}{suffix}"
    storage.save_bytes(uri, data)
    picture = RefPicture(uri=uri, label=label.strip().lower(), mime=MIME[suffix])
    item.pictures.append(picture)
    storage.save_episode(episode)
    return picture


def remove_picture(episode: Episode, owner: str, uri: str) -> None:
    item = _owner(episode.canon, owner)
    item.pictures = [p for p in item.pictures if p.uri != uri]
    if item.approved and item.approved.uri == uri:
        item.approved, item.auto_look = None, False   # do not take it again by itself
    storage.save_episode(episode)


def character_pictures(episode: Episode) -> list[RefPicture]:
    """The character's pictures, including one saved by an older version of the app,
    or else the approved look."""
    found = list(episode.canon.character.pictures) if episode.canon else []
    legacy = LEGACY.format(episode_id=episode.episode_id)
    if not found and storage.exists(legacy):
        found = [RefPicture(uri=legacy, label="reference")]
    if not found and episode.canon and episode.canon.character.approved:
        found = [episode.canon.character.approved]
    return found


def prop_pictures(prop) -> list[RefPicture]:
    """The prop's pictures, or else its approved look."""
    return list(prop.pictures) or ([prop.approved] if prop.approved else [])


def reference(episode: Episode, owner: str) -> Optional[RefPicture]:
    """The one picture an item's look is compared against, if there is one."""
    if owner == "character":
        found = character_pictures(episode)
    else:
        found = prop_pictures(_owner(episode.canon, owner))
    return found[0] if found else None


def _matches(label: str, state: str) -> bool:
    label, state = label.strip().lower(), state.strip().lower()
    return bool(label) and bool(state) and (label in state or state in label)


def pick_for_shot(episode: Episode, spec: ShotSpec) -> list[tuple[str, RefPicture]]:
    """Up to three (description, picture) pairs for Veo, most useful first."""
    canon = episode.canon
    chosen: list[tuple[str, RefPicture]] = []
    people = character_pictures(episode)
    name = canon.character.name if canon else "Character"
    if people:
        chosen.append((name, people[0]))
    for prop in canon.props if canon else []:
        own = prop_pictures(prop)
        if not own:
            continue
        _, end = prop.states_for(spec.shot_id)
        best = next((p for p in own if _matches(p.label, end)), own[0])
        chosen.append((prop.name, best))
    for extra in people[1:]:
        chosen.append((name, extra))
    return chosen[:MAX_FOR_VEO]


def all_pictures(episode: Episode) -> list[tuple[str, RefPicture]]:
    """Every picture, for the critic to compare against."""
    canon = episode.canon
    if canon is None:
        return []
    found = [(canon.character.name, p) for p in character_pictures(episode)]
    for prop in canon.props:
        found += [(prop.name, p) for p in prop_pictures(prop)]
    return found[:MAX_FOR_CRITIC]


def describe(owner: str, picture: RefPicture) -> str:
    return f"{owner} ({picture.label})" if picture.label else owner


def load(picture: RefPicture) -> Optional[bytes]:
    return storage.load_bytes(picture.uri) if storage.exists(picture.uri) else None
