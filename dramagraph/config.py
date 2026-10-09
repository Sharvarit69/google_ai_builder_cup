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


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        gemini_api_key=env("GEMINI_API_KEY", ""),
        gen_passcode=env("GEN_PASSCODE", ""),
        gcs_bucket=env("GCS_BUCKET", ""),
        text_model=env("TEXT_MODEL", ""),
        video_understanding_model=env("VIDEO_UNDERSTANDING_MODEL", ""),
        veo_model=env("VEO_MODEL", ""),
        veo_budget_seconds=int(env("VEO_BUDGET_SECONDS", "200")),
        shot_seconds=int(env("SHOT_SECONDS", "8")),
        max_repairs=int(env("MAX_REPAIRS", "2")),
        max_gen_retries=int(env("MAX_GEN_RETRIES", "2")),
    )
