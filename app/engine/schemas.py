"""Canonical Data Schemas for Speech Processing, Transcript Attribution, and Confidence Analytics."""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ConfidenceFeatures(BaseModel):
    """Raw acoustic and model features utilized for confidence calibration."""
    acoustic_score: float = Field(..., description="Acoustic or token log-probability / likelihood")
    duration: float = Field(..., description="Word acoustic duration in seconds")
    speech_ratio: float = Field(1.0, description="Ratio of word duration to parent VAD chunk duration")
    overlap_detected: bool = Field(False, description="Flag indicating presence of overlapping speaker energy")
    energy_level: float = Field(1.0, description="Normalized RMS energy level of the word frame")


class AlignedWord(BaseModel):
    """A single recognized word with temporal bounds, speaker attribution, and confidence."""
    word: str = Field(..., description="Recognized word token")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    speaker: str = Field(default="SPEAKER_00", description="Diarized speaker label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score in [0.0, 1.0]")
    raw_confidence: float = Field(..., ge=0.0, le=1.0, description="Raw uncalibrated confidence score")
    is_calibrated: bool = Field(default=True, description="Whether calibration mapping was applied")
    features: Optional[ConfidenceFeatures] = Field(default=None, description="Acoustic and model features")


class ConversationTurn(BaseModel):
    """Contiguous speech segment attributed to a single speaker."""
    turn_id: str = Field(..., description="Unique turn identifier")
    speaker: str = Field(..., description="Speaker identifier")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Combined text of all words in the turn")
    words: list[AlignedWord] = Field(default_factory=list, description="Contained word tokens")
    avg_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Mean confidence across words")


class CalibrationMetadata(BaseModel):
    """Provenance and metrics for the confidence calibration subsystem."""
    method: str = Field(default="isotonic_regression", description="Calibration technique used")
    version: str = Field(default="1.0.0", description="Calibration model version")
    is_calibrated: bool = Field(default=True, description="Whether scores are calibrated probabilities")
    brier_score: Optional[float] = Field(default=None, description="Empirical Brier Score")
    ece: Optional[float] = Field(default=None, description="Expected Calibration Error")


class TranscriptResult(BaseModel):
    """Complete canonical transcript, attribution, and confidence analysis result."""
    recording_id: str = Field(..., description="Unique recording identifier")
    filename: str = Field(..., description="Original filename")
    duration: float = Field(..., ge=0.0, description="Media duration in seconds")
    sample_rate: int = Field(default=16000, description="Sampling rate in Hz")
    channels: int = Field(default=1, description="Audio channel count")
    method: str = Field(default="Phase20-OracleMSDD-MidpointFusion", description="Pipeline method tag")
    calibration: CalibrationMetadata = Field(default_factory=CalibrationMetadata)
    speakers: list[str] = Field(default_factory=list, description="Unique detected speakers")
    segments: list[ConversationTurn] = Field(default_factory=list, description="Sequential speaker turns")
    words: list[AlignedWord] = Field(default_factory=list, description="All word tokens")
    summary: dict[str, Any] = Field(default_factory=dict, description="Summary statistics")


# API Data Transfer Objects (DTOs)

class MediaUploadResponse(BaseModel):
    recording_id: str
    filename: str
    size_bytes: int
    duration_seconds: float
    sha256: str
    status: str
    message: str


class TranscriptionJobRequest(BaseModel):
    recording_id: str
    use_mock: bool = Field(default=False, description="Deprecated; real Phase 20 pipeline is always executed")
    enable_denoising: bool = Field(default=False, description="Enable speech enhancement")
    language: str = Field(default="en", description="Spoken language code")
    initial_prompt: Optional[str] = Field(default=None, description="Domain vocabulary prompt")


class JobStatusResponse(BaseModel):
    job_id: str
    recording_id: str
    status: str = Field(..., description="queued | processing | completed | failed")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress ratio (0.0 to 1.0)")
    stage: str = Field(default="idle", description="Current pipeline stage")
    error: Optional[str] = None
    created_at: str
    updated_at: str


class UserFeedbackItem(BaseModel):
    feedback_id: Optional[str] = None
    recording_id: str
    turn_id: Optional[str] = None
    word_index: Optional[int] = None
    original_text: str
    corrected_text: str
    original_speaker: Optional[str] = None
    corrected_speaker: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
