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
    judge_mode: bool  # public demo: Gemini requests capped per day, no Veo, demo episodes locked
    judge_episodes: tuple  # in judge mode, demo episode ids that cannot be changed (and are listed alone)
    judge_gemini_calls_per_day: int  # in judge mode, Gemini requests allowed per day for all visitors


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
        judge_gemini_calls_per_day=int(env("JUDGE_GEMINI_CALLS_PER_DAY", "60") or 0),
    )
