# 04 — Configuration, Environment, and Model Audit

## 1. Environment Manifests Comparison

The repository contains five different dependency manifests reflecting different experimental stages:

| Manifest Path | Total Packages | PyTorch Version | Python Target | Primary Use Case / Role |
| :--- | :---: | :---: | :---: | :--- |
| `requirements.txt` (Root) | 26 | `>=2.2.0` | `>=3.10` | Early Stage-1 prototype requirements (loose bounds). |
| `requirements-exact.txt` (Root) | 279 | `2.8.0` | `3.10.12` | Frozen Linux pip freeze from experimental server. |
| `phase20_standalone/requirements.txt` | 179 | `2.8.0` | `3.10.12` | **Verified, reproducible Phase 20 frozen environment.** |
| `phase20_standalone/environment.yml` | 179 | `2.8.0` | `3.10.12` | Conda manifest with CUDA 12.8 / cuDNN 9.1 runtime spec. |
| `vad_aug_temp/requirements.txt` | 8 | Any | `>=3.9` | Synthetic noise evaluation dependencies only. |

---

## 2. Core Library Version Matrix

| Library | Root `requirements.txt` | `phase20_standalone/requirements.txt` | Role in Final Pipeline | Conflict / Compatibility Note |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | `>=3.10` | `3.10.12` | Core Runtime | Python 3.10–3.11 is the gold standard for NeMo and WhisperX. |
| **PyTorch (`torch`)** | `>=2.2.0` | `2.8.0+cu128` | Deep Learning Backend | PyTorch 2.4+ required for modern CTranslate2 and NeMo 2.7+. |
| **TorchAudio** | `>=2.2.0` | `2.8.0+cu128` | Audio Tensor Ops | Must match exact PyTorch CUDA build. |
| **Faster-Whisper** | Not specified | `1.2.1` | ASR Decoding Engine | Highly efficient CTranslate2-based Whisper backend. |
| **CTranslate2** | Not specified | `4.7.2` | Fast Inference Engine | Depends on cuDNN 9+ and CUDA 12+. |
| **WhisperX** | Not specified | `3.8.5` | Alignment & ASR Wrapper | Requires Wav2Vec2 alignment models from HuggingFace. |
| **Pyannote Audio** | `pyannote.metrics` only | `4.0.4` | High-Recall VAD Grounding | Requires HuggingFace token and license agreement. |
| **NVIDIA NeMo** | Not present | `2.7.3` | Oracle MSDD Diarization | Large footprint (`nemo-toolkit`). Requires PyTorch Lightning & OmegaConf. |
| **DeepFilterNet** | Not present | `0.5.6` | Speech Denoising (Optional)| Heavy Rust-based library. Ablated in Epoch 4 due to temporal drift. |
| **librosa** | `>=0.10.1` | `0.11.0` | Audio Analysis / Slicing | Fully compatible. |
| **SoundFile** | `>=0.12.1` | `0.13.1` | WAV I/O Backend | Uses system `libsndfile`. |
| **FastAPI** | Not present | `0.136.3` | Application REST API | Included in Phase 20 environment; ideal for `app/backend/`. |
| **Uvicorn** | Not present | `0.48.0` | ASGI Server | Included in Phase 20 environment. |
| **SQLAlchemy** | Not present | `2.0.50` | Database ORM | Included in Phase 20 environment; ideal for `app/backend/`. |

---

## 3. Pretrained Model Manifest and Authentication Requirements

The verified pipeline uses four primary pretrained deep learning models:

| Model Name / Identifier | Source / Hub | Parameter Size | Memory (VRAM) | Authentication Required? | Local Fallback Available? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`pyannote/voice-activity-detection`** | HuggingFace Hub | ~50 MB | ~1.2 GB | **Yes (`HF_TOKEN`)**<br>User must accept Pyannote conditions | **Yes** (`reference_vad_fallback` in `step03_vad.py`) |
| **Faster-Whisper `large-v3`** | HuggingFace / CTranslate2 | ~3.1 GB | ~4.5 GB (FP16)<br>~2.2 GB (INT8) | **No** (Public weights) | **Yes** (`fallback_words_from_reference` in `step05b_whisperx.py`) |
| **Wav2Vec2 Alignment (`en`)** | HuggingFace Hub | ~1.2 GB | ~1.5 GB | **No** (Public weights) | **Yes** (Uniform segment-duration interpolation) |
| **NeMo TitaNet Large (`titanet_large`)** | NVIDIA NGC / HF | ~100 MB | ~1.0 GB | **No** (NGC Public CDN) | **Yes** (Copies reference RTTM) |
| **NeMo MSDD (`diar_msdd_telephonic`)** | NVIDIA NGC / HF | ~80 MB | ~1.5 GB | **No** (NGC Public CDN) | **Yes** (Copies reference RTTM) |

### Model Download Management
`phase20_standalone/download_models.py` provides a centralized pre-caching mechanism. It loads all required weights into `.cache/` (`HF_HOME`, `TORCH_HOME`), ensuring that subsequent inference runs operate strictly offline.

---

## 4. Hardware and Platform Constraints

The verified production run documented in `phase20_standalone/README.md` was executed on:
- **Operating System:** Ubuntu 22.04.5 LTS (Linux kernel 6.8.0-1057-aws)
- **GPU:** NVIDIA Tesla T4 (16 GB GDDR6 VRAM)
- **NVIDIA Driver:** 580.159.04
- **CUDA Runtime:** 12.8 / 13.0
- **cuDNN:** 9.1.3

### Execution on CPU / Windows Development Environments:
1. **NeMo MSDD Limitations:** NeMo's NeuralDiarizer and Flash-Attention primitives are heavily optimized for Linux x86_64 and often fail or hang when compiled on native Windows without WSL2.
2. **DeepFilterNet Compilation:** DeepFilterNet utilizes Rust extensions (`DeepFilterLib`) which require MSVC C++ build tools on Windows.
3. **Recommendation for `app/`:**
   - The standalone `app/` must be containerized using a multi-stage Docker environment (`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`).
   - For local development on Windows/macOS without CUDA, `app/` must provide a `MOCK_INFERENCE=true` mode that leverages the built-in reference fallbacks already present in `phase20_standalone/scripts/`.

---

## 5. Proposed Dependency Architecture for `app/`

Rather than inheriting bloated 279-package freeze files or under-specified prototype requirements, `app/` should declare clean, structured dependencies in `app/pyproject.toml` using optional dependency groups:

```toml
[project]
name = "conversational-intelligence-platform"
version = "1.0.0"
description = "Production Conversational Intelligence & Speech Analytics Platform"
readme = "README.md"
requires-python = ">=3.10,<3.12"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.28.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "sqlalchemy>=2.0.28",
    "alembic>=1.13.1",
    "python-multipart>=0.0.9",
    "celery>=5.3.6",
    "redis>=5.0.3",
    "soundfile>=0.12.1",
    "librosa>=0.10.1",
    "pyloudnorm>=0.1.1",
    "scipy>=1.12.0",
    "numpy>=1.26.0,<2.0.0",
    "python-dotenv>=1.0.1",
    "rich>=13.7.0"
]

[project.optional-dependencies]
# Heavy AI / ML inference engine dependencies
engine = [
    "torch==2.4.0",
    "torchaudio==2.4.0",
    "faster-whisper==1.0.3",
    "whisperx==3.1.1",
    "pyannote-audio>=3.1.1,<4.0.0",
    "nemo-toolkit[asr]>=2.0.0",
    "ctranslate2>=4.4.0",
    "transformers>=4.40.0"
]

# Development and testing
dev = [
    "pytest>=8.1.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "httpx>=0.27.0",
    "ruff>=0.3.4",
    "mypy>=1.9.0"
]
```

### Key Dependency Benefits:
1. **Lightweight Backend Development:** Developers can install just `pip install -e .` without CUDA or 4 GB of PyTorch weights to develop the FastAPI backend, database schemas, frontend, and API endpoints.
2. **Deterministic Engine Deployment:** GPU workers install `pip install -e .[engine]` inside the CUDA Docker container.
