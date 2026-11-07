# 09 — Application Architecture & Component Design Report: Video Transcription with Confidence Analytics

**Course Project:** CS698B — Conversational Intelligence Platform  
**Specification:** Video Transcription with Confidence Analytics  
**Date:** November 2026  
**Target:** Standalone Application in `<PROJECT_ROOT>/app/`

---

## 1. Final Module Structure

The application is structured into decoupled, single-responsibility layers:

```text
app/
├── README.md                           # Setup and developer documentation
├── pyproject.toml                      # Clean dependencies specification
├── .env.example                        # Environment variables template
├── .gitignore                          # Ignored runtime artifacts
│
├── backend/                            # FastAPI Application Server
│   ├── __init__.py
│   ├── config.py                       # App settings (Pydantic BaseSettings)
│   ├── main.py                         # FastAPI application factory and router registration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                     # DB & service dependencies
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py               # Aggregated v1 router
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── media.py            # POST /api/v1/media/upload
│   │           ├── transcriptions.py   # POST /api/v1/transcriptions, GET /{id}, GET /{id}/status
│   │           ├── analytics.py        # GET /api/v1/analytics/{id}
│   │           ├── feedback.py         # POST /api/v1/feedback, GET /api/v1/feedback
│   │           └── health.py           # GET /api/v1/health
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audio_service.py            # Upload validation, FFmpeg extraction/conversion, SHA-256
│   │   ├── job_service.py              # Background job state machine, recovery, single-job concurrency
│   │   ├── transcription_service.py    # Pipeline orchestration and persistence
│   │   ├── feedback_service.py         # User corrections & vocabulary extraction
│   │   └── export_service.py           # Formatters: JSON, TXT, SRT, VTT
│   └── database/
│       ├── __init__.py
│       ├── db.py                       # SQLite connection & schema initializer
│       └── models.py                   # Storage records: recordings, jobs, transcripts, feedback
│
├── engine/                             # Independent Speech Intelligence Engine
│   ├── __init__.py
│   ├── schemas.py                      # Canonical data models (Word, Turn, TranscriptResult)
│   ├── base.py                         # BaseSpeechEngine abstract interface
│   ├── mock_engine.py                  # Deterministic mock producing 100% canonical results
│   ├── preprocessor.py                 # 16kHz mono conversion & loudness normalization
│   ├── vad.py                          # Pyannote VAD interface & recursive sub-30s slicer
│   ├── asr.py                          # Faster-Whisper Large-v3 engine (condition=False)
│   ├── alignment.py                    # Wav2Vec2 forced alignment
│   ├── diarization.py                  # NeMo Oracle MSDD + TitaNet diarization
│   └── fusion.py                       # Word-level midpoint speaker attribution
│
├── adaptive/                           # Confidence Analytics & Adaptive Feedback Loop
│   ├── __init__.py
│   ├── confidence.py                   # Feature extraction, temperature scaling, isotonic calibration, metrics (ECE, Brier)
│   ├── calibration.py                  # Calibration model trainer & persistence
│   └── feedback_loop.py                # Correction error tracking & Whisper dynamic prompt boosting
│
├── frontend/                           # Streamlit Web User Interface
│   ├── __init__.py
│   ├── streamlit_app.py                # Main Streamlit dashboard
│   ├── api_client.py                   # HTTP client interacting with FastAPI backend
│   └── components/
│       ├── __init__.py
│       ├── player.py                   # Audio playback & segment preview
│       ├── transcript_view.py          # Accessible styled transcript (Okabe-Ito, shapes, symbols)
│       ├── charts.py                   # Altair scatter, histogram, segment bars, reliability curves
│       └── tables.py                   # Accessible tabular data & CSV download
│
├── storage/                            # Isolated Runtime Storage
│   ├── uploads/                        # Raw user uploads
│   ├── processed/                      # 16kHz normalized WAV files
│   ├── results/                        # Exported JSON/TXT/SRT files
│   └── app.db                          # SQLite persistent database
│
├── configs/
│   └── config.yaml                     # Pipeline parameters & confidence thresholds
│
├── scripts/
│   ├── init_db.py                      # DB initialization script
│   └── run_mock_demo.py                # CLI runner for quick end-to-end demonstration
│
└── tests/                              # Comprehensive Test Battery
    ├── __init__.py
    ├── conftest.py                     # Pytest fixtures & mock client
    ├── test_independence.py            # Static AST verification (0 outside imports)
    ├── test_schemas.py                 # Canonical schemas serialization test
    ├── test_mock_engine.py             # Speech engine test
    ├── test_confidence.py              # Calibration & metric calculation test
    ├── test_api.py                     # FastAPI REST API endpoints test
    ├── test_services.py                # Job scheduler & recovery test
    └── test_e2e_workflow.py            # Full workflow test (Upload -> Process -> Review -> Export)
```

---

## 2. Canonical Transcript and Confidence Data Schema

To reconcile the research outputs with the visualization frontend and API, a single unified schema is defined in `app/engine/schemas.py`:

```python
class ConfidenceFeatures(BaseModel):
    acoustic_score: float = Field(..., description="Acoustic or token log-probability")
    duration: float = Field(..., description="Word duration in seconds")
    speech_ratio: float = Field(1.0, description="Ratio of word duration to VAD segment")
    overlap_detected: bool = Field(False, description="Whether word falls in overlapping speech")
    energy_level: float = Field(1.0, description="Normalized RMS energy level of the word window")

class AlignedWord(BaseModel):
    word: str
    start: float
    end: float
    speaker: str
    confidence: float = Field(..., description="Calibrated confidence score in [0.0, 1.0]")
    raw_confidence: float = Field(..., description="Raw uncalibrated confidence score")
    is_calibrated: bool = Field(True, description="Whether calibration mapping was applied")
    features: Optional[ConfidenceFeatures] = None

class ConversationTurn(BaseModel):
    turn_id: str
    speaker: str
    start: float
    end: float
    text: str
    words: list[AlignedWord]
    avg_confidence: float

class CalibrationMetadata(BaseModel):
    method: str = "isotonic_regression"
    version: str = "1.0.0"
    is_calibrated: bool = True
    brier_score: Optional[float] = None
    ece: Optional[float] = None

class TranscriptResult(BaseModel):
    recording_id: str
    filename: str
    duration: float
    sample_rate: int = 16000
    channels: int = 1
    method: str = "Phase20-OracleMSDD-MidpointFusion"
    calibration: CalibrationMetadata
    speakers: list[str]
    segments: list[ConversationTurn]
    words: list[AlignedWord]
    summary: dict[str, Any] = Field(default_factory=dict)
```

---

## 3. Interfaces Across Subsystems

1. **Ingestion $\rightarrow$ AudioService:**
   - Input: Media file stream (MP4/MOV/WAV/MP3).
   - Validation: MIME check + FFprobe stream inspection.
   - Output: Normalized 16kHz Mono PCM WAV + SHA-256 digest + `Recording` DB row.
2. **AudioService $\rightarrow$ JobService $\rightarrow$ SpeechEngine:**
   - Trigger: Asynchronous background worker thread (bounded by a concurrency lock to avoid multi-job GPU clash).
   - Input: File path to normalized WAV.
   - Output: `TranscriptResult` object.
3. **SpeechEngine $\rightarrow$ Adaptive Confidence Subsystem:**
   - Raw features extracted during ASR/Alignment/VAD.
   - Calibrated using pre-fitted Isotonic Regression mapping (`app/adaptive/confidence.py`).
   - Produces calibrated confidence scores, confidence tiers, and reliability metrics.
4. **Backend $\rightarrow$ Database Storage:**
   - Stores recording metadata, job state machine (`queued` $\rightarrow$ `processing` $\rightarrow$ `completed`/`failed`), structured JSON, and feedback corrections.
5. **FastAPI $\rightarrow$ Streamlit Frontend:**
   - REST API v1 endpoints with JSON payloads.
   - Streamlit communicates strictly via `requests` / `httpx` to FastAPI.

---

## 4. Selective Reuse of Group Member's Prototype (`app.py`)

The root `app.py` provided accessible, colour-blind-friendly visualization:
- **Preserved & Integrated:**
  - Accessible Okabe-Ito, Blue/Orange/Purple, and Greyscale palettes.
  - Multi-sensory confidence tiers (`low` < 0.30, `mid` 0.30–0.60, `high` > 0.60) using shapes (`▲`, `■`, `●`) and text underline styling.
  - Altair scatter plots (confidence over time), histograms, speech timelines, and segment confidence bars.
  - Accessible alternative data tables with CSV download.
- **Enhanced & Redesigned:**
  - Integrated directly with the FastAPI REST API client instead of requiring manual JSON file uploads.
  - Added media playback with audio player (`st.audio`) linked to the actual recording.
  - Reconciled data structures with the canonical `TranscriptResult` schema (handling empty transcripts and missing fields safely).
  - Added an interactive user correction interface enabling transcript and speaker editing.
  - Added a Confidence Reliability & Calibration Diagram (plotting predicted confidence bins against observed empirical accuracy).

---

## 5. Phase 20 Algorithm Exposure

Phase 20 algorithms are encapsulated into modular classes adhering to standard interfaces:
- `AudioPreprocessor`: FFmpeg 16kHz mono conversion & EBU R128 loudness normalization.
- `PyannoteVADEngine`: High-recall VAD with gap merging and recursive energy slicing (< 30s).
- `WhisperXASREngine`: Faster-Whisper `large-v3` transcription (`condition_on_previous_text=False`) and Wav2Vec2 alignment.
- `NeMoDiarizer`: NeMo Oracle MSDD diarization with TitaNet-Large embeddings.
- `WordMidpointFusion`: Word-to-speaker attribution and turn grouping.
- `MockSpeechEngine`: A fast, deterministic engine for CPU development and testing that implements the exact same interface and returns valid `TranscriptResult` objects without requiring GPU or HuggingFace tokens.

---

## 6. Minimal Dependency Set

- **Core & API:** `fastapi`, `uvicorn`, `pydantic>=2.6`, `requests`, `python-multipart`, `sqlite3`
- **Visualization:** `streamlit>=1.30`, `altair>=5.0`, `pandas`, `numpy`
- **Audio & Math:** `scipy`, `soundfile`, `ffmpeg-python`
- **Development & Tests:** `pytest>=8.0`, `httpx`

---

## 7. Testing Strategy

1. **AST Static Independence Test:** Scans all Python files inside `app/` and asserts 0 imports reference outside directories.
2. **Schema Validation Tests:** Verifies strict Pydantic parsing of words, turns, and metadata.
3. **Mock Engine Test:** Verifies that `MockSpeechEngine` runs in < 1 second and outputs complete canonical results.
4. **Confidence Calibration Test:** Tests temperature scaling, isotonic regression, and ECE calculation.
5. **API & Service Integration Tests:** Tests upload, job queueing, state transitions, job recovery on startup, and feedback endpoints via FastAPI `TestClient`.
6. **Full Workflow Test:** Tests end-to-end path: Ingest $\rightarrow$ Mock Pipeline $\rightarrow$ DB Storage $\rightarrow$ API Status $\rightarrow$ Result Retrieval $\rightarrow$ Export.
