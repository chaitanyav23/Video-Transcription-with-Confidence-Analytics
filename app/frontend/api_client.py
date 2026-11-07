"""HTTP Client for communicating with the FastAPI backend."""
from __future__ import annotations

from typing import Any, Optional
import requests


class APIClient:
    """Encapsulates REST communication with the FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def upload_media(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(f"{self.base_url}/api/v1/media/upload", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_recordings(self) -> list[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/media", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def trigger_transcription(
        self, recording_id: str, use_mock: bool = False, initial_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "use_mock": use_mock,
            "initial_prompt": initial_prompt,
        }
        resp = requests.post(f"{self.base_url}/api/v1/transcriptions", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_job_status(self, job_or_rec_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/transcriptions/{job_or_rec_id}/status", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_transcript(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/transcriptions/{recording_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 202:
            return None  # In progress
        resp.raise_for_status()
        return None

    def get_analytics(self, recording_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/analytics/{recording_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def submit_feedback(
        self,
        recording_id: str,
        turn_id: Optional[str],
        original_text: str,
        corrected_text: str,
        original_speaker: Optional[str] = None,
        corrected_speaker: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "turn_id": turn_id,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "original_speaker": original_speaker,
            "corrected_speaker": corrected_speaker,
            "notes": notes,
        }
        resp = requests.post(f"{self.base_url}/api/v1/feedback", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_vocabulary(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/v1/feedback/vocabulary", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_export_url(self, recording_id: str, format_str: str) -> str:
        return f"{self.base_url}/api/v1/transcriptions/{recording_id}/export?format={format_str}"

    def get_stream_url(self, recording_id: str) -> str:
        return f"{self.base_url}/api/v1/media/{recording_id}/stream"
