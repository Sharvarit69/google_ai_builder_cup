"""Runs the shots in order: generate, check, repair. Saves after every step."""
from typing import Callable, Optional

from . import generator, storage
from .config import get_settings
from .critic import check_clip
from .models import Attempt, Episode, Shot
from .prompt_compiler import compile_prompt
from .repair import auto_fix_notes, pick_best_attempt, repairs_done

DONE = ("ACCEPTED", "OVERRULED", "DROPPED")
Progress = Optional[Callable[[str, float], None]]


def _attempt(shot: Shot, attempt_no) -> Optional[Attempt]:
    return next((a for a in shot.attempts if a.attempt_no == attempt_no), None)


def chosen_clip(shot: Shot) -> Optional[str]:
    a = _attempt(shot, shot.chosen_attempt)
    return a.clip_uri if a else None


def _previous_clip(episode: Episode, shot: Shot) -> Optional[str]:
    """The chosen clip of the nearest earlier shot that is still in the episode."""
    before = episode.shots[: episode.shots.index(shot)]
    for earlier in reversed(before):
        if earlier.status != "DROPPED" and chosen_clip(earlier):
            return storage.local_copy(chosen_clip(earlier))
    return None


def _check(episode, shot, attempt, check) -> None:
    try:
        attempt.critic = check(storage.local_copy(attempt.clip_uri), shot.spec, episode.canon,
                               _previous_clip(episode, shot))
    except Exception as error:
        attempt.error = f"Critic failed: {error}"[:500]


def _settle(shot: Shot) -> None:
    """Decide the shot's status from its attempts."""
    last = next((a for a in reversed(shot.attempts) if a.clip_uri), None)
    if last and last.critic and last.critic.decision == "ACCEPT":
        shot.chosen_attempt, shot.status = last.attempt_no, "ACCEPTED"
    else:
        shot.chosen_attempt, shot.status = pick_best_attempt(shot), "FLAGGED"


def _generate_and_check(episode, shot, prompt, kind, seeded, generate, check) -> Attempt:
    attempt = generate(episode, shot, prompt, kind, seeded)
    if attempt.clip_uri:
        _check(episode, shot, attempt, check)
    storage.save_episode(episode)
    return attempt


def process_shot(episode: Episode, shot: Shot, generate, check) -> str:
    """Returns "ok", "skipped" (no clip yet) or "ration" (budget used up)."""
    seed = episode.seeded_error
    use_seed = bool(seed and seed.shot_id == shot.spec.shot_id
                    and not any(a.seeded for a in shot.attempts))
    shot.status = "GENERATING"
    prompt = compile_prompt(shot.spec, episode.canon, seed=seed if use_seed else None)
    attempt = _generate_and_check(episode, shot, prompt, "INITIAL", use_seed, generate, check)
    if attempt.error == generator.RATION_EMPTY:
        return "ration"
    if not attempt.clip_uri:
        shot.status = "SKIPPED"
        return "skipped"

    while (attempt.critic and attempt.critic.decision == "REGENERATE"
           and repairs_done(shot) < get_settings().max_repairs):
        fixed = compile_prompt(shot.spec, episode.canon, auto_fix_notes(attempt.critic))
        retry = _generate_and_check(episode, shot, fixed, "AUTO_REPAIR", False, generate, check)
        if retry.error == generator.RATION_EMPTY:
            _settle(shot)
            return "ration"
        if not retry.clip_uri:
            break
        attempt = retry
    _settle(shot)
    return "ok"


def finish(episode: Episode) -> Episode:
    from .qa_report import build_report

    episode.status = "READY" if all(s.status in DONE for s in episode.shots) else "NEEDS_REVIEW"
    episode.report = build_report(episode)
    storage.save_episode(episode)
    return episode


def run_generation(episode: Episode, on_progress: Progress = None,
                   generate=generator.generate_with_retries, check=check_clip) -> Episode:
    """Generate every unfinished shot. Safe to call again after a crash."""
    if episode.canon is None or not episode.shots:
        raise ValueError("The episode needs details and a shot plan first.")
    episode.status = "GENERATING"
    storage.save_episode(episode)
    todo = [s for s in episode.shots if s.status in ("PLANNED", "GENERATING", "SKIPPED")]
    out_of_ration = False

    for second_pass in (False, True):
        for i, shot in enumerate(todo):
            if on_progress:
                label = "Retrying" if second_pass else "Generating and checking"
                on_progress(f"{label} shot {shot.spec.shot_id[1:]}", i / max(1, len(todo)))
            result = process_shot(episode, shot, generate, check)
            storage.save_episode(episode)
            if result == "ration":
                out_of_ration = True
                break
        todo = [s for s in episode.shots if s.status == "SKIPPED"]
        if out_of_ration or not todo:
            break

    for shot in episode.shots:  # anything still unfinished needs a person
        if shot.status in ("PLANNED", "GENERATING", "SKIPPED"):
            shot.chosen_attempt, shot.status = pick_best_attempt(shot), "FLAGGED"
    if on_progress:
        on_progress("Done", 1.0)
    return finish(episode)


def _shot(episode: Episode, shot_id: str) -> Shot:
    return next(s for s in episode.shots if s.spec.shot_id == shot_id)


def repair_with_instruction(episode: Episode, shot_id: str, instruction: str,
                            generate=generator.generate_with_retries, check=check_clip) -> Episode:
    """The creator's own fix. It counts as one of the allowed repair attempts."""
    shot = _shot(episode, shot_id)
    if not instruction.strip():
        raise ValueError("Type what should be fixed.")
    if repairs_done(shot) >= get_settings().max_repairs:
        raise ValueError("This shot has used all its repair attempts. Accept it or drop it.")
    notes = f"FIX: {instruction.strip()}\nKeep everything else the same."
    prompt = compile_prompt(shot.spec, episode.canon, notes)
    attempt = _generate_and_check(episode, shot, prompt, "CUSTOM_REPAIR", False, generate, check)
    if attempt.error == generator.RATION_EMPTY:
        raise ValueError("The Veo ration is used up.")
    _settle(shot)
    return finish(episode)


def overrule(episode: Episode, shot_id: str) -> Episode:
    """Accept a clip the critic did not pass. Kept on record for the false-alarm count."""
    shot = _shot(episode, shot_id)
    if shot.chosen_attempt is None:
        raise ValueError("There is no clip to accept for this shot.")
    shot.status = "OVERRULED"
    return finish(episode)


def drop(episode: Episode, shot_id: str) -> Episode:
    _shot(episode, shot_id).status = "DROPPED"
    return finish(episode)


def can_export(episode: Episode) -> tuple[bool, str]:
    waiting = [s.spec.shot_id for s in episode.shots if s.status not in DONE]
    if waiting:
        return False, "These shots still need a decision: " + ", ".join(waiting)
    if not any(s.status != "DROPPED" for s in episode.shots):
        return False, "Every shot was dropped."
    return True, ""
