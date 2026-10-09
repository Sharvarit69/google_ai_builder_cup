"""All data shapes used by DramaGraph. One Episode holds everything."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["match", "mismatch", "unclear"]
Decision = Literal["ACCEPT", "REGENERATE", "REVIEW"]
ViolationType = Literal["WARDROBE", "PROP_LOOK", "PROP_STATE", "SCENE", "CROSS_SHOT", "RULE"]


class Beat(BaseModel):
    beat_id: str
    script_lines: list[int]
    action: str
    caption: str
    props_needed: list[str] = Field(default_factory=list)
    needs_second_person: bool = False


class Character(BaseModel):
    name: str
    appearance: str
    wardrobe: str
    accessories: list[str] = Field(default_factory=list)
    reference_image_uri: Optional[str] = None


class Prop(BaseModel):
    name: str
    description: str
    # The prop's state at the END of each shot, e.g. {"S1": "closed", "S4": "open"}
    state_by_shot: dict[str, str] = Field(default_factory=dict)

    def states_for(self, shot_id: str) -> tuple[str, str]:
        """(state at the start, state at the end) of a shot.

        A shot starts in the state the previous shot ended in, so a prop may
        change during a shot (a box being opened) without that being an error.
        """
        end = self.state_by_shot.get(shot_id, "")
        try:
            previous = f"S{int(shot_id[1:]) - 1}"
        except ValueError:
            previous = ""
        start = self.state_by_shot.get(previous, "") or end
        return start, end


class LearnedRule(BaseModel):
    """A check that came from a person, not from the built-in list."""

    text: str                                             # one sentence that must be true
    shot_ids: list[str] = Field(default_factory=list)     # empty means every shot
    source: Literal["creator_catch", "observation", "manual", "library"] = "manual"
    from_note: str = ""                                   # what the creator originally wrote


class Canon(BaseModel):
    character: Character
    props: list[Prop] = Field(default_factory=list)
    location: str
    time_of_day: str
    rules: list[LearnedRule] = Field(default_factory=list)

    def rules_for(self, shot_id: str) -> list[LearnedRule]:
        return [r for r in self.rules if not r.shot_ids or shot_id in r.shot_ids]


class ShotSpec(BaseModel):
    shot_id: str
    beat_id: str
    duration_seconds: int = 8
    shot_type: str
    camera_motion: str
    action: str
    must_show: list[str] = Field(default_factory=list)
    must_not_show: list[str] = Field(default_factory=list)
    second_person: Optional[str] = None  # "hands only" or None


class CheckResult(BaseModel):
    verdict: Verdict
    observed: str


class Violation(BaseModel):
    type: ViolationType
    timestamp_seconds: float = 0.0
    expected: str
    observed: str


class RuleCheck(BaseModel):
    number: int                  # position in the list of rules given to the critic
    rule: str = ""               # filled in by code, not by the model
    verdict: Verdict
    observed: str = ""


class Observation(BaseModel):
    """Something odd the critic noticed that no check covers. Advisory only."""

    what: str
    timestamp_seconds: float = 0.0


class CriticResult(BaseModel):
    seen: str = ""
    wardrobe: CheckResult
    prop_look: Optional[CheckResult] = None  # is it the same object? None if no prop
    prop_state: CheckResult
    scene: CheckResult
    cross_shot: Optional[CheckResult] = None  # None for the first shot
    rules: list[RuleCheck] = Field(default_factory=list)
    violations: list[Violation] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    decision: Optional[Decision] = None  # set by code, never by the model


class Attempt(BaseModel):
    attempt_no: int
    kind: Literal["INITIAL", "AUTO_REPAIR", "CUSTOM_REPAIR", "UPLOADED"]
    prompt: str = ""
    clip_uri: Optional[str] = None
    critic: Optional[CriticResult] = None
    error: Optional[str] = None
    seeded: bool = False  # True if an error was planted on purpose for a test
    rejected: bool = False  # True if the creator rejected a clip the critic had passed


ShotStatus = Literal[
    "PLANNED", "GENERATING", "SKIPPED", "ACCEPTED", "FLAGGED", "OVERRULED", "DROPPED"
]


class Shot(BaseModel):
    spec: ShotSpec
    attempts: list[Attempt] = Field(default_factory=list)
    chosen_attempt: Optional[int] = None
    status: ShotStatus = "PLANNED"
    creator_catches: int = 0  # problems the creator spotted that the critic missed


class SeededError(BaseModel):
    """A deliberate mistake planted in one shot's first prompt, to test the critic."""

    shot_id: str
    kind: Literal["prop", "wardrobe"]
    value: str  # what the first prompt should wrongly ask for


EpisodeStatus = Literal[
    "DRAFT", "PLANNED", "GENERATING", "NEEDS_REVIEW", "READY", "EXPORTED"
]


class Episode(BaseModel):
    episode_id: str
    title: str
    created_at: str
    mode: Literal["GENERATE", "CHECK_ONLY"] = "GENERATE"
    script: list[str] = Field(default_factory=list)
    beats: list[Beat] = Field(default_factory=list)
    picked_beat_ids: list[str] = Field(default_factory=list)
    canon: Optional[Canon] = None
    shots: list[Shot] = Field(default_factory=list)
    status: EpisodeStatus = "DRAFT"
    final_video_uri: Optional[str] = None
    report: Optional[dict] = None
    seconds_generated: int = 0
    seeded_error: Optional[SeededError] = None
