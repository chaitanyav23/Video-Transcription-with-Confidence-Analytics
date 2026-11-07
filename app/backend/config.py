"""Application Configuration and Environment Settings."""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    """Central configuration parameters for the application."""
    app_name: str = "CS698B Video Transcription with Confidence Analytics"
    app_version: str = "1.0.0"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))

    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "127.0.0.1"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_url: str = Field(default_factory=lambda: os.getenv("API_URL", "http://127.0.0.1:8000"))

    storage_dir: Path = Field(default_factory=lambda: APP_DIR / os.getenv("STORAGE_DIR", "storage"))
    max_upload_size_mb: int = Field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_MB", "250")))
    max_duration_seconds: int = Field(default_factory=lambda: int(os.getenv("MAX_DURATION_SECONDS", "1800")))

    # ML Inference Settings (Phase 20 Real Pretrained Models)
    whisper_model: str = Field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")))
    calibration_method: str = Field(default_factory=lambda: os.getenv("CALIBRATION_METHOD", "isotonic_regression"))
    device: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))
    compute_type: str = Field(default_factory=lambda: os.getenv("COMPUTE_TYPE", "int8"))
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    mock_inference: bool = Field(default=False)

    @property
    def uploads_dir(self) -> Path:
        p = self.storage_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_dir(self) -> Path:
        p = self.storage_dir / "processed"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results_dir(self) -> Path:
        p = self.storage_dir / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
