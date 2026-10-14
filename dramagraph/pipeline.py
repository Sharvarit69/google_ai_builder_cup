"""Runs the shots in order: generate, check, repair. Saves after every step."""
from typing import Callable, Optional

from . import generator, learning, ration, storage
from .config import get_settings
from .critic import check_clip
from .models import Attempt, Episode, Shot
from .prompt_compiler import compile_prompt
from .repair import auto_fix_notes, pick_best_attempt, repairs_done

DONE = ("ACCEPTED", "OVERRULED", "DROPPED")


def ration_left() -> int:
    return ration.remaining()
Progress = Optional[Callable[[str, float], None]]


def _attempt(shot: Shot, attempt_no) -> Optional[Attempt]:
    return next((a for a in shot.attempts if a.attempt_no == attempt_no), None)


def latest_clip(shot: Shot) -> Optional[Attempt]:
    return next((a for a in reversed(shot.attempts) if a.clip_uri), None)


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


def _pictures_for_critic(episode: Episode) -> list:
    from . import pictures as pics

    found = []
    for owner, picture in pics.all_pictures(episode):
        data = pics.load(picture)
        if data:
            found.append((pics.describe(owner, picture), data, picture.mime))
    return found


def _check(episode, shot, attempt, check) -> None:
    try:
        args = (storage.local_copy(attempt.clip_uri), shot.spec, episode.canon,
                _previous_clip(episode, shot))
        pictures = _pictures_for_critic(episode) if check is check_clip else []
        attempt.critic = check(*args, pictures=pictures) if pictures else check(*args)
    except Exception as error:
        attempt.error = f"Critic failed: {error}"[:500]


def _settle(shot: Shot) -> None:
    """Choose the clip to use. A clip the creator picked by hand always wins; otherwise
    the best one so far, and the shot is accepted only if that clip passed."""
    pinned = _attempt(shot, shot.pinned_attempt)
    if pinned and pinned.clip_uri and not pinned.rejected:
        passed = bool(pinned.critic and pinned.critic.decision == "ACCEPT")
        shot.chosen_attempt = pinned.attempt_no
        shot.status = "ACCEPTED" if passed else "OVERRULED"
        return
    shot.pinned_attempt = None
    best = _attempt(shot, pick_best_attempt(shot))
    shot.chosen_attempt = best.attempt_no if best else None
    passed = bool(best and best.critic and best.critic.decision == "ACCEPT")
    shot.status = "ACCEPTED" if passed else "FLAGGED"


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
    episode.final_video_uri = None      # any change makes an earlier export out of date
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
    """The creator's own fix. The automatic repair limit does not apply, because the
    creator is choosing to spend the seconds; the Veo ration is still the hard cap."""
    shot = _shot(episode, shot_id)
    if not instruction.strip():
        raise ValueError("Type what should be fixed.")
    notes = f"FIX: {instruction.strip()}\nKeep everything else the same."
    prompt = compile_prompt(shot.spec, episode.canon, notes)
    attempt = _generate_and_check(episode, shot, prompt, "CUSTOM_REPAIR", False, generate, check)
    if attempt.error == generator.RATION_EMPTY:
        raise ValueError("The Veo ration is used up.")
    _settle(shot)
    return finish(episode)


def auto_repair(episode: Episode, shot_id: str,
                generate=generator.generate_with_retries, check=check_clip) -> Episode:
    """One more automatic repair of a flagged shot, from the critic's own findings."""
    shot = _shot(episode, shot_id)
    current = _attempt(shot, shot.chosen_attempt)
    if current is None or current.critic is None:
        raise ValueError("There is no checked clip to repair from. Generate the shot first.")
    if repairs_done(shot) >= get_settings().max_repairs:
        raise ValueError("This shot has used all its repair attempts. Accept it or drop it.")
    prompt = compile_prompt(shot.spec, episode.canon, auto_fix_notes(current.critic))
    attempt = _generate_and_check(episode, shot, prompt, "AUTO_REPAIR", False, generate, check)
    if attempt.error == generator.RATION_EMPTY:
        raise ValueError("The Veo ration is used up.")
    _settle(shot)
    return finish(episode)


def recheck(episode: Episode, on_progress: Progress = None, check=check_clip) -> Episode:
    """Run the critic again on every clip already generated. Makes no new video.

    Use after changing the character, prop or setting details.
    """
    todo = [(shot, a) for shot in episode.shots if shot.status != "DROPPED"
            for a in shot.attempts if a.clip_uri]
    failures: list[str] = []
    for i, (shot, attempt) in enumerate(todo):
        if on_progress:
            on_progress(f"Re-checking shot {shot.spec.shot_id[1:]}", i / max(1, len(todo)))
        previous = attempt.critic
        attempt.critic, attempt.error = None, None
        _check(episode, shot, attempt, check)
        if attempt.critic is None and previous is not None:
            # The check failed (for example a network problem). Keep the earlier result
            # rather than losing it; the error is recorded on the attempt.
            attempt.critic = previous
            failures.append(shot.spec.shot_id)
        is_last = all(s is not shot for s, _ in todo[i + 1:])
        if is_last:
            _settle(shot)   # settle before later shots compare against this one
        storage.save_episode(episode)
    if on_progress:
        on_progress("Done", 1.0)
    finish(episode)
    if failures:
        names = ", ".join(sorted({f[1:] for f in failures}))
        raise RuntimeError(f"Could not re-check shot {names}; their earlier results were kept. "
                           "Try again in a minute.")
    return episode


def reject_and_fix(episode: Episode, shot_id: str, instruction: str,
                   generate=generator.generate_with_retries, check=check_clip,
                   ask=None) -> Episode:
    """Kept for older callers: the creator rejects a passed clip. Same as fix_shot."""
    shot = _shot(episode, shot_id)
    if not instruction.strip():
        raise ValueError("Type what is wrong and what should be shown instead.")
    if _attempt(shot, shot.chosen_attempt) is None:
        raise ValueError("There is no clip to reject for this shot.")
    return fix_shot(episode, shot_id, instruction, generate, check, ask)


def save_rule_and_recheck(episode: Episode, shot_id: str, note: str, ask=None,
                          check=check_clip, on_progress: Progress = None) -> bool:
    """'Always check this': add the rule, then check every existing clip against it.
    Makes no new video. Returns False if the rule was already there."""
    added = learn(episode, shot_id, note, "observation", ask)
    if added:
        try:
            recheck(episode, on_progress=on_progress, check=check)
        except RuntimeError as error:
            raise RuntimeError(f"The rule was saved, but: {error}") from error
    return added


def learn(episode: Episode, shot_id: str, note: str, source: str, ask=None) -> bool:
    """Turn a note into a rule the critic checks from now on. Makes no video."""
    if not note.strip():
        raise ValueError("Type the rule or the problem first.")
    shot = _shot(episode, shot_id)
    kwargs = {"ask": ask} if ask else {}
    rule = learning.learn_rule(note, shot.spec, source, **kwargs)
    added = learning.add_rule(episode.canon, rule)
    storage.save_episode(episode)
    return added


def suggested_fix(shot: Shot) -> tuple[str, bool]:
    """What to pre-fill in the fix box, and whether it only restates a built-in check.

    A failed check comes first. Otherwise the most serious thing the critic noticed.
    """
    current = _attempt(shot, shot.chosen_attempt)
    if current is None or current.critic is None:
        return "", False
    critic = current.critic
    if critic.decision == "REGENERATE":
        wanted = [v.expected.strip().rstrip(".") for v in critic.violations if v.expected.strip()]
        wanted += [r.rule.strip().rstrip(".") for r in critic.rules if r.verdict == "mismatch"]
        unique = list(dict.fromkeys(wanted))
        if unique:
            return ". ".join(unique) + ".", True
    major = [o for o in critic.observations if o.severity == "major" and o.suggested_fix.strip()]
    if major:
        return major[0].suggested_fix.strip(), False
    return "", False


def fix_only_repeats_a_check(shot: Shot, instruction: str) -> bool:
    """True if the instruction is the critic's own wording of a built-in check, which
    is checked every time anyway, so saving it as a rule would add nothing."""
    prefill, restates = suggested_fix(shot)
    return restates and instruction.strip() == prefill


def fix_shot(episode: Episode, shot_id: str, instruction: str,
             generate=generator.generate_with_retries, check=check_clip, ask=None,
             save_rule: bool = True) -> Episode:
    """The one fix action. Regenerates the shot with the instruction.

    If the creator is fixing a clip the critic had passed, that clip is set aside and
    counted as a catch. Any instruction that goes beyond the built-in checks becomes a
    rule the critic enforces from now on.
    """
    shot = _shot(episode, shot_id)
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Say what should be shown, then press Fix.")
    if ration_left() < shot.spec.duration_seconds:
        raise ValueError("The Veo ration is used up, so the shot cannot be regenerated.")
    repeats_check = fix_only_repeats_a_check(shot, instruction)
    current = _attempt(shot, shot.chosen_attempt)
    if current is not None and shot.status in ("ACCEPTED", "OVERRULED"):
        current.rejected = True
        shot.creator_catches += 1
    set_aside = current if current is not None and current.rejected else None
    was_pinned, shot.pinned_attempt = shot.pinned_attempt, None
    if save_rule and not repeats_check:
        try:
            learn(episode, shot_id, instruction, "creator_catch", ask)
        except ValueError:
            pass    # the episode already holds the maximum number of rules; still fix the shot
    made_before = sum(1 for a in shot.attempts if a.clip_uri)
    try:
        repair_with_instruction(episode, shot_id, instruction, generate, check)
    finally:
        if sum(1 for a in shot.attempts if a.clip_uri) == made_before and set_aside is not None:
            # Veo made nothing new, so the creator keeps the clip they had.
            set_aside.rejected = False
            shot.creator_catches -= 1
            shot.pinned_attempt = was_pinned
            _settle(shot)
            finish(episode)
    return episode


def edit_shots(episode: Episode, changes: dict) -> Episode:
    """Direct edits to shot descriptions: {shot_id: (what happens, camera)}.

    The continuity requirements are re-applied, so an edit cannot remove them.
    A changed shot keeps the clips it has; the change applies to the next version made.
    """
    from .planner import apply_rules

    for shot in episode.shots:
        if shot.spec.shot_id not in changes:
            continue
        action, camera = changes[shot.spec.shot_id]
        if action.strip():
            shot.spec.action = action.strip()
        if camera.strip():
            shot.spec.shot_type, shot.spec.camera_motion = camera.strip(), ""
    apply_rules([s.spec for s in episode.shots], episode.canon)
    storage.save_episode(episode)
    return episode


def use_attempt(episode: Episode, shot_id: str, attempt_no: int) -> Episode:
    """The creator picks a clip by hand. It is used even if the critic prefers another."""
    shot = _shot(episode, shot_id)
    picked = _attempt(shot, attempt_no)
    if picked is None or not picked.clip_uri:
        raise ValueError("That attempt has no clip.")
    picked.rejected = False
    shot.pinned_attempt = attempt_no
    _settle(shot)
    return finish(episode)


def overrule(episode: Episode, shot_id: str) -> Episode:
    """Accept a clip the critic did not pass. Kept on record for the false-alarm count."""
    shot = _shot(episode, shot_id)
    target = _attempt(shot, shot.chosen_attempt) or latest_clip(shot)
    if target is None:
        raise ValueError("There is no clip to accept for this shot.")
    target.rejected = False
    shot.pinned_attempt = shot.chosen_attempt = target.attempt_no
    shot.status = "OVERRULED"
    return finish(episode)


def restore(episode: Episode, shot_id: str) -> Episode:
    """Bring back a dropped shot. Nothing is regenerated."""
    shot = _shot(episode, shot_id)
    if shot.status != "DROPPED":
        return episode
    _settle(shot)
    if shot.chosen_attempt is None and latest_clip(shot) is not None:
        shot.chosen_attempt = latest_clip(shot).attempt_no    # shown, awaiting a decision
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
