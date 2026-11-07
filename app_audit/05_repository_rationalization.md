# 05 — Repository Classification and Rationalization Report

## 1. Rationalization Framework and Classification Taxonomy

To transform the research codebase into a production-grade, independently runnable application without compromising historical research reproducibility, every major file and directory is classified according to the following 8-tier schema:

| Category | Definition | Action for Standalone `app/` |
| :--- | :--- | :--- |
| **`MIGRATE`** | Required non-code material (configs, fixtures, manifests) needed directly inside `app/`. | Copy and isolate into `app/`. |
| **`REUSE`** | Production-grade code ready for direct adoption without architectural changes. | Copy into `app/` maintaining provenance. |
| **`ADAPT`** | Scientifically sound code requiring refactoring for modularity, async execution, or OOP. | Refactor and rewrite into `app/engine/`. |
| **`KEEP`** | Vital research artifacts, reports, and benchmarks preserved in repo root outside `app/`. | Retain in place outside `app/`. |
| **`ARCHIVE`** | Historical experimental phases, intermediate logs, and legacy frozen runs. | Retain permanently in `experiments/` outside `app/`. |
| **`REDUNDANT`** | Duplicate implementations or superseded prototypes superseded by later phases. | Retain for research reference; exclude from `app/`. |
| **`OBSOLETE`** | Abandoned one-off scripts (e.g. Markdown text-appender scripts). | Retain for Git provenance; do NOT copy to `app/`. |
| **`INVESTIGATE`**| Files requiring user policy clarification (e.g. tracked secret keys). | Document security risk; exclude from `app/`. |

---

## 2. Directory-by-Directory Classification Matrix

| Current Path | Classification | Role & Forensic Purpose | Dependencies | Required by `app/`? | Proposed Destination / Policy |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `phase20_standalone/scripts/` | **`ADAPT`** | Verified Phase 20 pipeline and evaluation logic. | Python 3.10+, Pyannote, Faster-Whisper, NeMo | **Yes** | Refactor into `app/engine/` modules. |
| `phase20_standalone/environment/`| **`MIGRATE`** | Exact CUDA 12.8 / cuDNN runtime package hashes. | Conda / Pip | **Yes** | Reference for `app/Dockerfile`. |
| `phase20_standalone/` (remainder)| **`KEEP`** | Historical reproducible standalone research artifact. | Self-contained | No | Retain in place as research milestone. |
| `raw_audio/` | **`KEEP`** | 8 benchmark MP3 recordings (26.7 MB). | None | No | Retain outside `app/` as benchmark dataset. |
| `ground_truth/` | **`KEEP`** | 8 canonical reference transcripts (.txt). | None | No | Retain outside `app/` as reference benchmarks. |
| `ground_truth_json/` | **`KEEP`** | Segment & word JSON references (`bwc`, `karen`). | None | No | Retain outside `app/`; use 1 as test fixture. |
| `ground_truth_rttm/` | **`KEEP`** | NIST RTTM references (`bwc.rttm`, `karen.rttm`). | None | No | Retain outside `app/`. |
| `experiments/` (22 subdirs) | **`ARCHIVE`** | 2,602 research phase files, plots, and logs. | Historical venvs | No | Retain permanently in `experiments/`. |
| `external/silero-vad/` | **`ARCHIVE`** | Full clone of Silero VAD git repo. | PyTorch / ONNX | No | Exclude from `app/`. Load via Torch Hub if needed. |
| `external/WhisperX.../` | **`ARCHIVE`** | Legacy HuggingFace Space wrapper code. | None | No | Exclude from `app/`. |
| `papers/` | **`KEEP`** | 9 reference academic literature PDFs. | None | No | Retain in repository root. |
| `documentation/` | **`KEEP`** | Engineering specifications & evolution logs. | None | No | Retain in repository root. |
| `config/` (`secret_key`, etc.) | **`INVESTIGATE`**| Active credentials and encryption keys. | None | **NO (Security)** | Do NOT migrate. Require runtime `.env`. |
| `configs/` (`stage1_config.yaml`)| **`ADAPT`** | Stage-1 pipeline YAML configuration. | PyYAML | **Yes** | Refactor into `app/configs/default_config.yaml`. |
| `scripts/utils/logger.py` | **`REUSE`** | Structured console/file logging utility. | Standard Lib | **Yes** | Copy to `app/backend/core/logging.py`. |
| `scripts/utils/device.py` | **`REUSE`** | CUDA / CPU device selection utility. | `torch` | **Yes** | Copy to `app/engine/core/device.py`. |
| `scripts/vad_engine.py` | **`ADAPT`** | Multi-VAD evaluation runner (Silero + Pyannote).| Pyannote, Torch | No | Replaced by `app/engine/vad/`. |
| `preprocessed_B_normalized/` | **`REDUNDANT`** | Intermediate normalized WAV files. | Ephemeral | No | Exclude from `app/`. |
| `preprocessed_C_...` | **`REDUNDANT`** | Bandpassed intermediate WAV files. | Ephemeral | No | Exclude from `app/`. |
| `speech_segments/` | **`REDUNDANT`** | 26 sliced WAV audio chunks. | Ephemeral | No | Exclude from `app/`. |
| `asr_outputs_wx*/` (3 dirs) | **`REDUNDANT`** | Cached historical ASR transcript JSONs. | Ephemeral | No | Exclude from `app/`. |
| `diarization_outputs_*/` (3 dirs)| **`REDUNDANT`** | Cached historical diarization RTTMs. | Ephemeral | No | Exclude from `app/`. |
| `error_analysis_*/` (4 dirs) | **`KEEP`** | Forensic diagnostic logs and error tables. | None | No | Retain in place for research audit. |
| `final_bwc_outputs*/` (5 dirs)| **`KEEP`** | Frozen historical pipeline outputs. | None | No | Retain in place for research audit. |
| `vad_aug_temp/` | **`KEEP`** | Synthetic noise benchmark generator suite. | MS-SNSD | No | Retain outside `app/` as auxiliary research tool. |
| `reports/` | **`KEEP`** | Historical Markdown/CSV evaluation summaries. | None | No | Retain in repository root. |
| `logs/` | **`REDUNDANT`** | Historical execution log files. | None | No | Exclude from `app/`. |

---

## 3. Detailed Root Python Scripts Rationalization

The 44 root Python scripts are categorized below with explicit forensic reasoning:

### 3.1 Candidates for Code Adaptation into `app/`
- None directly from root. **All production components are superior in `phase20_standalone/scripts/`**. The root versions are un-refactored ancestors.

### 3.2 Redundant or Superseded Root Scripts (Keep in Root, Exclude from `app/`)
- `run_production_pipeline.py`: Superseded by `phase20_standalone/run_phase20.py`.
- `run_final_hardened_pipeline.py`: Superseded by `phase20_standalone/run_phase20.py`.
- `run_stage1.py` & `run_stage1_whisperx.py`: Precursor VAD scripts superseded by `step03_vad.py`.
- `run_stage4_parallel.py`: Early parallel prototype superseded by Phase 20 dual-branch execution.
- `run_diarization.py` & `run_diarization_pyannote_31.py`: Standalone diarization scripts superseded by `step05a_msdd.py`.
- `run_asr.py`: Standalone ASR script superseded by `step05b_whisperx.py`.
- `evaluate_asr.py` & `evaluate_asr_improved.py`: Standard WER evaluation superseded by Hungarian cpWER in `step06b_asr_eval.py`.
- `evaluate_diarization.py` & `evaluate_diarization_improved.py`: Superseded by `step06a_diar_eval.py`.
- `evaluate_alignment.py` & `evaluate_conversational_timeline.py`: Superseded by `timeline_metrics.py`.
- `evaluate_karen.py`, `run_karen_inference.py`, `run_karen_audit.py`: Karen-specific scripts superseded by general `--file` handling in Phase 20.
- `preprocess_audio.py`: Standalone preprocessing script superseded by `step01_preprocess.py` and `step02b_loudnorm.py`.

### 3.3 Obsolete or One-Off Helper Scripts (Exclude from `app/`)
- `append_part2.py`, `append_part3.py`, `append_part4.py`: Scripts used only to append markdown text into `COMPLETE_TECHNICAL_RESEARCH_REPORT.md`.
- `fix_karen_gt.py`, `parse_karen_gt.py`, `convert_karen.py`, `truncate_karen.py`: One-off regex fixing scripts for Karen transcript iterations.
- `audio_audit.py`, `check.py`, `find_vad.py`: Diagnostic exploration scripts.
- `test_audio_quality.py`, `test_jiwer.py`, `test_pipeline_load.py`, `test_silero.py`, `test_vad.py`, `test_vad_params.py`, `test_vad_simple.py`: Ad-hoc test scripts superseded by formal pytest suites in `app/backend/tests/`.

---

## 4. Migration Plan for Reusable Components

The following table documents the precise file-to-module migration path from `phase20_standalone/scripts/` to `app/`:

| Source Component (`phase20_standalone/`) | Proposed Destination in `app/` | Refactoring Required |
| :--- | :--- | :--- |
| `scripts/step01_preprocess.py`<br>`scripts/step02b_loudnorm.py` | `app/engine/preprocessing/audio.py` | Encapsulate into an `AudioPreprocessor` class with FFmpeg subprocess calls and EBU R128 normalization. |
| `scripts/step03_vad.py` | `app/engine/vad/pyannote_vad.py` | Encapsulate Pyannote pipeline loading, recursive slicing, and fallback logic into `PyannoteVADEngine`. |
| `scripts/step05b_whisperx.py` | `app/engine/asr/whisper_engine.py` | Encapsulate Faster-Whisper transcription and Wav2Vec2 alignment into `WhisperXASREngine`. |
| `scripts/step05a_msdd.py` | `app/engine/diarization/nemo_msdd.py` | Encapsulate NeMo Oracle MSDD diarization into `NeMoDiarizationEngine`. |
| `scripts/step07_fusion.py` | `app/engine/fusion/word_midpoint.py` | Encapsulate word-midpoint speaker attribution into `WordMidpointFusionEngine`. |
| `scripts/speaker_mapping.py`<br>`scripts/cpwer.py`<br>`scripts/sawer.py`<br>`scripts/timeline_metrics.py` | `app/engine/evaluation/metrics.py` | Consolidate Hungarian matching, cpWER, SAWER, Speaker Accuracy, and Timeline Accuracy into a unified metric suite. |
| `scripts/common.py` (data models) | `app/engine/schemas/results.py` | Convert dictionary payloads into strict Pydantic v2 schemas (`TranscriptSegment`, `AlignedWord`, `AttributedWord`, `ProcessingResult`). |
| `run_phase20.py` (orchestrator) | `app/engine/pipeline/orchestrator.py` | Convert CLI subprocess runner into a callable async Python class `ConversationalIntelligencePipeline`. |
