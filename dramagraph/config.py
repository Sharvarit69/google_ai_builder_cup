"""Settings, read from environment variables. Nothing secret is written in code."""
import os
from dataclasses import dataclass

try:  # a local .env file is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gen_passcode: str
    gcs_bucket: str  # empty means "use the local_data/ folder"
    text_model: str
    video_understanding_model: str
    veo_model: str
    veo_budget_seconds: int
    shot_seconds: int
    max_repairs: int
    max_gen_retries: int
    critic_fps: float  # frames per second the critic samples; 0 means Gemini's default (about 1)
    judge_mode: bool  # browse-only: no Gemini or Veo calls and nothing saved, so visitors cost nothing
    judge_episodes: tuple  # in judge mode, the episode ids visitors may open; empty means all
    judge_checks_per_day: int  # in judge mode, clip checks allowed per day for all visitors together


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        gemini_api_key=env("GEMINI_API_KEY", "").strip(),
        gen_passcode=env("GEN_PASSCODE", "").strip(),
        gcs_bucket=env("GCS_BUCKET", ""),
        text_model=env("TEXT_MODEL", "") or env("VIDEO_UNDERSTANDING_MODEL", ""),
        video_understanding_model=env("VIDEO_UNDERSTANDING_MODEL", ""),
        veo_model=env("VEO_MODEL", ""),
        veo_budget_seconds=int(env("VEO_BUDGET_SECONDS", "200")),
        shot_seconds=int(env("SHOT_SECONDS", "8")),
        max_repairs=int(env("MAX_REPAIRS", "2")),
        max_gen_retries=int(env("MAX_GEN_RETRIES", "2")),
        critic_fps=float(env("CRITIC_FPS", "0") or 0),
        judge_mode=env("JUDGE_MODE", "").strip().lower() in ("1", "true", "yes", "on"),
        judge_episodes=tuple(i.strip() for i in env("JUDGE_EPISODES", "").split(",") if i.strip()),
        judge_checks_per_day=int(env("JUDGE_CHECKS_PER_DAY", "10") or 0),
    )
