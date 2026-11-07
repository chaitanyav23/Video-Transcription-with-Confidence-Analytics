# 06 — Standalone Application Architecture (`app/`)

## 1. Architectural Philosophy and Independence Guarantee

The new application directory:

`<PROJECT_ROOT>/app/`

is engineered as a **completely standalone, self-contained, and production-ready system**. It must be strictly decoupled from the research repository.

### Absolute Independence Invariants:
1. **Zero External Python Imports:** No module within `app/` may import from `phase20_standalone`, `experiments`, `scripts`, or the repository root.
2. **Zero Escaping File Paths:** No script in `app/` may access relative paths containing `../` that cross the `app/` boundary.
3. **Zero Symbolic Links:** No symlinks pointing outside `app/`.
4. **Zero Working-Directory Assumptions:** The application executes properly whether invoked from `app/`, containerized in Docker, or relocated to an entirely separate system (e.g. `/opt/conversational-ai/`).
5. **Separation of Code and Data:** All runtime audio, transcripts, and database files reside in `app/storage/` (or external object storage), completely separated from source code.

---

## 2. Standalone Application Directory Hierarchy

```text
app/
├── README.md                           # Application setup and operations guide
├── .env.example                        # Safe runtime environment variables template
├── .gitignore                          # Application-specific ignore rules
├── pyproject.toml                      # Clean packaging and dependency specification
├── Dockerfile                          # Multi-stage GPU/CPU Docker container definition
├── docker-compose.yml                  # Full stack orchestration (App + Redis + Postgres)
│
├── backend/                            # FastAPI Application Server
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI entrypoint, middleware, CORS, routing
│   │   ├── config.py                   # Pydantic BaseSettings (env parsing)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py                 # DB session and Auth dependencies
│   │   │   ├── v1/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── router.py           # API v1 consolidated router
│   │   │   │   ├── endpoints/
│   │   │   │   │   ├── audio.py        # Ingestion, upload, metadata, streaming
│   │   │   │   │   ├── jobs.py         # Job dispatch, status polling, cancellation
│   │   │   │   │   ├── transcripts.py  # Transcript retrieval, editing, export
│   │   │   │   │   ├── evaluation.py   # cpWER/SAWER benchmark evaluation
│   │   │   │   │   └── adaptive.py     # Vocabulary boosting & strategy tuning
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── logging.py              # Structured JSON logging
│   │   │   ├── security.py             # API Key verification & JWT auth
│   │   │   └── exceptions.py           # Unified error handling
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── session.py              # SQLAlchemy engine and sessionmaker
│   │   │   ├── base.py                 # Declarative base
│   │   │   └── migrations/             # Alembic migration scripts
│   │   ├── models/                     # SQLAlchemy ORM Models
│   │   │   ├── __init__.py
│   │   │   ├── recording.py            # Audio file metadata, hash, duration, path
│   │   │   ├── job.py                  # Processing task state, progress, errors
│   │   │   ├── transcript.py           # Turns, aligned words, speaker attribution
│   │   │   ├── revision.py             # User manual correction history
│   │   │   └── evaluation.py           # Benchmark scores and runs
│   │   ├── schemas/                    # Pydantic v2 Request/Response DTOs
│   │   │   ├── __init__.py
│   │   │   ├── audio.py
│   │   │   ├── job.py
│   │   │   ├── transcript.py
│   │   │   ├── evaluation.py
│   │   │   └── adaptive.py
│   │   ├── services/                   # Business Logic Layer
│   │   │   ├── __init__.py
│   │   │   ├── audio_service.py        # Hashing, validation, storage management
│   │   │   ├── job_service.py          # State machine transitions
│   │   │   ├── transcript_service.py   # Versioning and turn reconstruction
│   │   │   └── export_service.py       # JSON, SRT, VTT, TXT formatters
│   │   └── workers/                    # Asynchronous Task Execution
│   │       ├── __init__.py
│   │       ├── celery_app.py           # Celery / Redis task worker configuration
│   │       └── tasks.py                # Heavy AI inference execution tasks
│   └── tests/                          # Backend Unit and API Integration Tests
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_api/
│       └── test_services/
│
├── engine/                             # Core Conversational Intelligence Engine
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py             # Callable ConversationalIntelligencePipeline
│   │   └── context.py                  # ProcessingContext (holds intermediate tensors)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio.py                    # FFmpeg header repair, mono 16kHz conversion
│   │   └── normalizer.py               # EBU R128 / RMS loudness normalization
│   ├── vad/
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract BaseVADEngine interface
│   │   ├── pyannote_vad.py             # Pyannote 3.1 High-Recall VAD
│   │   ├── recursive_slicer.py         # Sub-30s chunking with probability/RMS valleys
│   │   └── muting_mask.py              # NumPy binary muting mask generator
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract BaseASREngine interface
│   │   └── whisper_engine.py           # Faster-Whisper Large-v3 with condition=False
│   ├── alignment/
│   │   ├── __init__.py
│   │   └── wav2vec2_aligner.py         # WhisperX Wav2Vec2 phoneme/word alignment
│   ├── diarization/
│   │   ├── __init__.py
│   │   ├── base.py                     # Abstract BaseDiarizer interface
│   │   ├── nemo_msdd.py                # NeMo Oracle MSDD + TitaNet-Large
│   │   └── pyannote_diar.py            # Pyannote 3.1 clustering fallback
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── word_midpoint.py            # Midpoint speaker assignment & turn grouping
│   │   └── confidence.py               # Attribution confidence scoring
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Hungarian solver, cpWER, SAWER, Speaker Acc
│   │   └── benchmark.py                # Reference dataset runner
│   └── fallbacks/
│       ├── __init__.py
│       └── mock_engine.py              # Deterministic mock inference for CI/unit tests
│
├── adaptive/                           # Adaptive Intelligence Subsystem
│   ├── __init__.py
│   ├── vocabulary/
│   │   ├── __init__.py
│   │   ├── term_extractor.py           # Law enforcement / domain term extraction
│   │   └── prompt_booster.py           # Dynamic Whisper initial_prompt injection
│   ├── feedback/
│   │   ├── __init__.py
│   │   ├── diff_tracker.py             # Word error tracking from user corrections
│   │   └── error_categorizer.py        # Acoustic vs Lexical error classification
│   └── configuration/
│       ├── __init__.py
│       └── auto_tuner.py               # Reliability-aware VAD & collar parameter tuning
│
├── frontend/                           # React / Vite Single Page Application
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── api/                        # Typed OpenAPI client hooks
│       ├── components/                 # Reusable UI primitives
│       │   ├── AudioPlayer/            # Waveform visualizer (wavesurfer.js)
│       │   ├── TranscriptTimeline/     # Word-level synchronized transcript view
│       │   ├── SpeakerLegend/          # Speaker badges, colors, and renamer
│       │   └── MetricCard/             # cpWER, SAWER, Speaker Accuracy stats
│       ├── pages/
│       │   ├── DashboardPage.tsx       # Recording list, job status, upload dropzone
│       │   ├── AnalysisPage.tsx        # Audio player + synchronized transcript editor
│       │   ├── EvaluationPage.tsx      # Benchmark comparison dashboard
│       │   └── SettingsPage.tsx        # System configuration, API keys, models
│       └── types/
│
├── storage/                            # Isolated Persistent File Storage
│   ├── uploads/                        # User raw uploaded audio
│   ├── processed/                      # Normalized 16kHz WAV working files
│   ├── results/                        # Exported JSON, TXT, SRT transcripts
│   └── database/                       # Default SQLite database file (if not Postgres)
│
├── configs/
│   ├── default_config.yaml             # Canonical pipeline thresholds and parameters
│   └── logging_config.yaml             # Structured logging configuration
│
├── scripts/
│   ├── run_standalone_test.sh          # Self-contained zero-repo verification script
│   ├── download_models.py              # Pre-caches Pyannote, Faster-Whisper, NeMo weights
│   └── init_db.py                      # Database schema bootstrapping
│
└── tests/                              # Standalone Verification Test Battery
    ├── test_imports.py                 # Strict AST scan checking 0 outside dependencies
    ├── test_pipeline_mock.py           # End-to-end pipeline test using mock engine
    └── test_backend_api.py             # REST API startup and route validation
```

---

## 3. Detailed Functional Scope

### A. Audio Ingestion and ETLS
- **Upload & Validation:** Handles multipart audio uploads (`.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`). Validates container headers and audio stream presence via `ffprobe`.
- **Forensic Content Hashing:** Automatically computes SHA-256 hash upon ingest. Blocks exact duplicates or links duplicate uploads to existing processing runs.
- **Preprocessing Pipeline:** In-memory or temporary FFmpeg execution to resample audio to 16,000 Hz, 16-bit mono PCM WAV, followed by EBU R128 loudness normalization (-20 dBFS target).
- **Asynchronous Task Architecture:** Upload endpoints return `202 Accepted` with a `job_id`. Background workers (Celery/Redis or FastAPI `BackgroundTasks`) handle model inference asynchronously.

### B. Conversational Intelligence Engine
- **Pyannote 3.1 VAD:** High-recall segmentation with micro-gap bridging (300ms) and short-segment suppression (200ms).
- **Recursive Chunk Slicing:** Dynamically splits long continuous speech bursts (> 30s) at probability and energy minima to guarantee Whisper compliance.
- **NumPy Binary Muting Mask:** Zeroes out sirens and ambient field noise outside detected speech segments to prevent ASR hallucination loops.
- **Faster-Whisper Large-v3:** High-speed CTranslate2 transcription using `condition_on_previous_text=False` to prevent cyclic repetitions.
- **Wav2Vec2 Forced Alignment:** Sub-word acoustic alignment producing millisecond-accurate word start/end bounds and character bounds.
- **NeMo Oracle MSDD:** Diarization conditioned on Pyannote VAD Oracle RTTM, leveraging TitaNet-Large multi-scale embeddings and telephonic MSDD clustering.
- **Word-Level Midpoint Fusion:** Maps each word's midpoint $t_{mid}$ to active diarization segments with nearest-neighbor collar tolerance, grouping contiguous tokens into formatted speaker turns.

### C. Historical Data and Persistence
- **SQLAlchemy Relational Schema:** Persists recordings, processing jobs, diarized turns, aligned words, and system configuration.
- **Immutable Revision History:** User corrections in the UI create non-destructive audit log revisions, preserving the raw model hypothesis while tracking the human-verified truth.

### D. Adaptive Intelligence Subsystem
- **Dynamic Vocabulary Injection:** Extracts police codes (e.g. "PC", "10-4", "Section 1"), street names, and domain terminology from historical corrections and injects them as dynamic `initial_prompt` prefixes to Faster-Whisper.
- **Correction Tracking:** Compares user-edited transcripts with original model hypotheses using Levenshtein distance, categorizing errors into Acoustic (misheard word) vs Diarization (speaker boundary swap).
- **Strategy Selection:** Dynamically adjusts VAD sensitivity thresholds and MSDD collar parameters based on estimated recording SNR.

### E. Visualization and Frontend Experience
- **Waveform Timeline:** Interactive waveform powered by Wavesurfer.js, highlighting speaker turn regions.
- **Synchronized Transcript Scrolling:** Real-time playback highlighting active words as the audio plays.
- **Speaker Reassignment:** Direct UI actions to reassign speaker turns, merge speaker clusters, or rename generic labels (e.g., `SPEAKER_00` -> "Officer Smith").
- **Metric Dashboard:** Displays cpWER, SAWER, DER, and Speaker Accuracy when reference transcripts are provided.

### F. Application and Developer REST API
- **OpenAPI 3.1 Compliant:** Fully interactive `/docs` Swagger UI and `/redoc` documentation.
- **Core Endpoints:**
  - `POST /api/v1/audio/upload`: Upload raw recording.
  - `POST /api/v1/jobs/process`: Dispatch processing pipeline job.
  - `GET /api/v1/jobs/{job_id}`: Poll job status and progress percentage.
  - `GET /api/v1/transcripts/{recording_id}`: Retrieve full forensic transcript JSON/TXT.
  - `PUT /api/v1/transcripts/{recording_id}`: Submit manual correction revisions.
  - `POST /api/v1/evaluation/benchmark`: Run cpWER/SAWER against uploaded reference.
