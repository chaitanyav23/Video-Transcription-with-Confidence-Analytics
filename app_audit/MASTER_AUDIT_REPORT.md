# MASTER AUDIT REPORT: Comprehensive Repository Forensic Audit, Codebase Rationalization, and Standalone Application Design (`app/`)

**Author:** DeepMind Agentic Pair Programmer  
**Repository:** `https://github.com/chaitanyav23/Conversational-Intelligence-Platform`  
**Inspected Commit:** `f0a4ff646f99c713e2c425ad6b74704d93584300` (`main` HEAD)  
**Date:** November 2026  
**Status:** Audit and Architectural Blueprint Complete (Zero existing files modified or deleted)

---

## 1. Executive Summary & Definite Conclusion

### The Central Question:
> **Can we create a completely standalone `app/` directory using the existing repository's verified research components without depending on the rest of the repository?**

### The Definitive Answer:
**YES, UNEQUIVOCALLY.**

Our forensic investigation of the 3,133 files (464.69 MB) across the repository demonstrates that the research work has already culminated in a highly decoupled, mathematically rigorous, and fully verified pipeline encapsulated inside:

`<PROJECT_ROOT>/phase20_standalone/`

This sub-repository was created specifically to isolate Phase 20's hardened dual-branch pipeline (`run_phase20.py` and its 21 modular scripts). It does not import from parent experimental directories, does not depend on legacy root scripts, implements permutation-invariant Hungarian/Levenshtein metric evaluation (cpWER, SAWER, Speaker Accuracy, Timeline Accuracy), and contains built-in reference fallbacks for offline testing.

By selectively adapting the verified algorithms from `phase20_standalone/scripts/` into a production-grade FastAPI + React architecture inside `<PROJECT_ROOT>/app/`, we can construct a **100% self-contained application** that does not depend on any file or script outside `app/`.

---

## 2. Answers to the 11 Core Forensic Questions

### Q1: What do we currently have?
We have an extensive, multi-epoch research repository spanning 20 experimental phases (2,602 files in `experiments/`), 44 root-level prototype scripts, 8 raw benchmark audio recordings in `raw_audio/`, matching reference transcripts in `ground_truth/`, structured JSON and RTTM annotations in `ground_truth_json/` and `ground_truth_rttm/`, a standalone research milestone in `phase20_standalone/`, cloned third-party tools in `external/`, and numerous historical evaluation reports.

### Q2: What actually works?
- **The Phase 20 Hardened Dual-Branch Pipeline WORKS:**
  1. Header repair & resampling to 16 kHz Mono PCM WAV.
  2. EBU R128 loudness normalization (-20 dBFS).
  3. Pyannote 3.1 high-recall VAD with gap-merging and recursive chunk splitting (< 30s).
  4. NumPy binary muting mask (physically zeroing out sirens and non-speech before ASR).
  5. Faster-Whisper `large-v3` transcription (`condition_on_previous_text=False`).
  6. Wav2Vec2 phoneme/word forced alignment.
  7. NeMo Oracle MSDD Diarization conditioned on Pyannote VAD segments, using TitaNet-Large embeddings.
  8. Word-level temporal midpoint speaker attribution with nearest-neighbor collar tolerance.
  9. Permutation-invariant Hungarian evaluation (cpWER, SAWER, Speaker Accuracy, Timeline Accuracy).
- **What FAILED in earlier phases:**
  - Aggressive pre-ASR denoising using DeepFilterNet (Phase 2 & Phase 10) destroyed acoustic formants and caused 10–25ms temporal phase drift, breaking word-to-speaker fusion.
  - Naive WhisperX internal VAD skipped low-volume field speech.
  - Unconditioned NeMo MarbleNet VAD crashed on police sirens.

### Q3: Which implementation should become the application engine?
**The implementation in `phase20_standalone/scripts/` MUST become the application engine.**  
The root-level scripts (`run_production_pipeline.py`, `run_final_hardened_pipeline.py`, `run_stage1.py`) are un-refactored historical prototypes that rely on legacy hardcoded paths and ad-hoc `.env` loading. `phase20_standalone/scripts/` represents the latest verified state of the art.

### Q4: Which code should be reused directly?
- `scripts/utils/logger.py`: Clean, structured Python logging utility.
- `scripts/utils/device.py`: Clean CUDA/MPS/CPU device detection utility.
- Hungarian assignment logic (`linear_sum_assignment`) in `phase20_standalone/scripts/speaker_mapping.py`.

### Q5: Which code should be adapted?
- `phase20_standalone/scripts/step01_preprocess.py` & `step02b_loudnorm.py` -> `app/engine/preprocessing/audio.py`
- `phase20_standalone/scripts/step03_vad.py` -> `app/engine/vad/pyannote_vad.py` & `recursive_slicer.py`
- `phase20_standalone/scripts/step05b_whisperx.py` -> `app/engine/asr/whisper_engine.py` & `app/engine/alignment/`
- `phase20_standalone/scripts/step05a_msdd.py` -> `app/engine/diarization/nemo_msdd.py`
- `phase20_standalone/scripts/step07_fusion.py` -> `app/engine/fusion/word_midpoint.py`
- `phase20_standalone/scripts/cpwer.py`, `sawer.py`, `timeline_metrics.py` -> `app/engine/evaluation/metrics.py`
- `phase20_standalone/run_phase20.py` -> `app/engine/pipeline/orchestrator.py`

### Q6: Which historical artifacts should be retained?
All 22 experimental phases in `experiments/`, frozen outputs in `final_bwc_outputs*/`, error analyses in `error_analysis_*/`, research papers in `papers/`, documentation in `documentation/`, and the frozen milestone in `phase20_standalone/` must remain preserved outside `app/` to guarantee permanent scientific reproducibility.

### Q7: Which files are unnecessary?
- Temporary markdown-appender scripts (`append_part2.py`, `append_part3.py`, `append_part4.py`).
- Iterative regex fixers (`fix_karen_gt.py`, `parse_karen_gt.py`, `truncate_karen.py`, `convert_karen.py`).
- Intermediate WAV chunks (`speech_segments/`, `preprocessed_B_normalized/`, `preprocessed_C_...`).
- Historical intermediate JSON/RTTM dumps in root (`asr_outputs_wx*`, `diarization_outputs_*`, `vad_outputs*`).
*Note: These files remain outside `app/`; none are deleted during this audit.*

### Q8: Which datasets and ground-truth files are important?
The latest commit (`f0a4ff6`) established a clean 1-to-1 mapping across 8 benchmark recordings:
1. `bwc` ("That was a bit of a cheap shot weren’t it mate") - 175.43s (BWC Police stop)
2. `rich_karen` ("Rich Karen Messes Around & Finds Out") - 652.18s (Escalated traffic stop)
3. `sara_sharif` ("Bodycam footage show arrests of Sara Sharif's alleged killers") - 100.96s (Arrest bodycam)
4. `dennis_skinner` ("Dennis Skinner kicked out of Commons") - 176.26s (Parliamentary acoustic chamber)
5. `disabled_parking` ("Karen Has A Problem With Disabled Parking") - 206.22s (Outdoor noisy dispute)
6. `piers_tommy` ("Piers Confronts Tommy Robinson") - 204.54s (High speech overlap interview)
7. `southwest_pilot` ("Police bodycam shows arrest of Southwest Airlines pilot") - 162.45s (Airport police bodycam)
8. `uk_parliament` ("UK Parliament Debate") - 215.34s (Multi-speaker debate)

All 8 have verified timestamp compatibility with their audio files. `bwc` and `karen` additionally possess word-level JSON and RTTM annotations.

### Q9: What should the standalone `app/` directory contain?
- `backend/`: FastAPI application server, SQLAlchemy models, asynchronous job queue, REST API v1.
- `engine/`: Decoupled, object-oriented pipeline engine (preprocessing, VAD, ASR, alignment, NeMo MSDD diarization, midpoint fusion, Hungarian evaluation, and mock fallbacks).
- `adaptive/`: Self-improving intelligence (domain vocabulary extraction, prompt boosting, correction error tracking).
- `frontend/`: Modern React/Vite web application with interactive Wavesurfer waveform audio player and synchronized word-level transcript editor.
- `storage/`: Isolated local storage for uploads, normalized files, SQLite database, and exports.
- `Dockerfile` & `docker-compose.yml`: Containerized production deployment.
- `tests/`: 8-tier verification suite including static AST import scanners.

### Q10: How can we prove that it is independent?
1. **Static AST Verification:** An automated script (`tests/test_imports.py`) scans all AST import nodes in `app/` to prove 0 imports escape to parent directories.
2. **Physical Isolation Test:** We copy `app/` to an isolated system directory (`/tmp/standalone_app_test`), build a clean virtual environment, and run the complete test suite with no access to the parent git repository.

### Q11: What is the safest implementation sequence?
1. Phase 1: App foundation, packaging, Pydantic v2 schemas, and mock engine.
2. Phase 2: Engine extraction & modularization from `phase20_standalone/scripts/`.
3. Phase 3: Backend API, database models, and upload endpoints.
4. Phase 4: Asynchronous worker queue and job state machine.
5. Phase 5: Modern React web frontend.
6. Phase 6: Adaptive intelligence subsystem.
7. Phase 7: Docker packaging and physical isolation verification.

---

## 3. Deliverables Summary in `app_audit/`

| Deliverable File | Focus Area & Contents |
| :--- | :--- |
| `app_audit/01_repository_inventory.md` | Full repository tree, file counts, sizes, latest commit forensics, security findings. |
| `app_audit/02_code_dependency_audit.md` | Entrypoint inventory, script-by-script analysis, import & execution graphs, epoch evolution. |
| `app_audit/03_dataset_ground_truth_audit.md` | Raw audio manifest, SHA-256 hashes, ground-truth mapping matrix, timestamp compatibility. |
| `app_audit/04_environment_model_audit.md` | Version matrices, PyTorch/CUDA dependencies, model manifests, packaging specification. |
| `app_audit/05_repository_rationalization.md` | Complete file-level and directory-level classification table (MIGRATE, ADAPT, KEEP, ARCHIVE). |
| `app_audit/06_standalone_app_architecture.md` | Complete architecture design, directory tree, and functional scope for `app/`. |
| `app_audit/07_independence_verification_plan.md`| Static AST scanner, isolated sandbox test protocol, 8-tier verification battery, provenance matrix. |
| `app_audit/08_migration_roadmap.md` | 7-phase implementation roadmap, risks, acceptance criteria, and key architectural choices. |
| `app_audit/MASTER_AUDIT_REPORT.md` | This consolidated master audit report. |

---

## 4. Immediate Next Steps

With the audit and architectural blueprint fully complete, we are ready to proceed with **Phase 1 of the implementation roadmap**: scaffolding the `<PROJECT_ROOT>/app/` directory, configuring packaging, declaring Pydantic schemas, and setting up the mock verification suite.
