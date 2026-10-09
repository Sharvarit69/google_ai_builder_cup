"""All data shapes used by DramaGraph. One Episode holds everything."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Verdict = Literal["match", "mismatch", "unclear"]
Decision = Literal["ACCEPT", "REGENERATE", "REVIEW"]
ViolationType = Literal["WARDROBE", "PROP_STATE", "SCENE", "CROSS_SHOT"]


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
    state_by_shot: dict[str, str] = Field(default_factory=dict)  # {"S1": "closed"}


class Canon(BaseModel):
    character: Character
    props: list[Prop] = Field(default_factory=list)
    location: str
    time_of_day: str


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


class CriticResult(BaseModel):
    seen: str = ""
    wardrobe: CheckResult
    prop_state: CheckResult
    scene: CheckResult
    cross_shot: Optional[CheckResult] = None  # None for the first shot
    violations: list[Violation] = Field(default_factory=list)
    decision: Optional[Decision] = None  # set by code, never by the model


class Attempt(BaseModel):
    attempt_no: int
    kind: Literal["INITIAL", "AUTO_REPAIR", "CUSTOM_REPAIR", "UPLOADED"]
    prompt: str = ""
    clip_uri: Optional[str] = None
    critic: Optional[CriticResult] = None
    error: Optional[str] = None


ShotStatus = Literal[
    "PLANNED", "GENERATING", "SKIPPED", "ACCEPTED", "FLAGGED", "OVERRULED", "DROPPED"
]


class Shot(BaseModel):
    spec: ShotSpec
    attempts: list[Attempt] = Field(default_factory=list)
    chosen_attempt: Optional[int] = None
    status: ShotStatus = "PLANNED"


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
