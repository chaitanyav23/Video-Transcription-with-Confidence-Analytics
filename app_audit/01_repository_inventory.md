# 01 — Repository Inventory and Reconnaissance

## 1. Repository Overview

| Property | Value / Status |
| :--- | :--- |
| **Repository Name** | Conversational-Intelligence-Platform |
| **Remote URL** | `https://github.com/chaitanyav23/Conversational-Intelligence-Platform` |
| **Current Branch** | `main` |
| **Tracking Upstream** | `origin/main` (Up to date, clean working tree) |
| **Current HEAD Commit** | `f0a4ff646f99c713e2c425ad6b74704d93584300` |
| **Latest Commit Date** | `Sat Nov 7 17:16:27 2026 +0530` |
| **Latest Commit Author** | `chaitanyav23 <chaitanyav23@iitk.ac.in>` |
| **Latest Commit Message**| `added raw_audio and ground_truth` |
| **Total Git Commits** | 2 (`dc2e329` -> `f0a4ff6`) |
| **Tracked Files in Git**| 3,080 |
| **Total Files on Disk** | 3,133 (excluding `.git/`) |
| **Total Workspace Size** | 464.69 MB (excluding `.git/`) |
| **Operating System** | Windows (Dev workspace) / Target Runtime: Linux (Ubuntu 22.04 LTS / Docker) |

---

## 2. Commit History and Latest Commit Forensics

The repository contains exactly two commits in its Git history:

1. **Commit 1 (`dc2e329a1e7b5c226dc6e082bbeac3ec808f09a4`)**
   - **Message:** `first commit`
   - **Scope:** 3,085 files added, 1,599,018 insertions.
   - **Contents:** Full experimental research tree (Phases 1 through 20), root orchestration scripts, research papers, documentation reports, evaluation tables, intermediate feature dumps (`speech_segments/`), preprocessed audio files, model checkpoints, external submodules, and the standalone `phase20_standalone` directory.

2. **Commit 2 — Latest HEAD (`f0a4ff646f99c713e2c425ad6b74704d93584300`)**
   - **Message:** `added raw_audio and ground_truth`
   - **Scope:** 17 files changed, 813 insertions(+), 3,525 deletions(-).
   - **Detailed Change Breakdown:**
     - **Ground-Truth Renaming and Normalization:**
       - `ground_truth/Dennis_skinner_kicked_out.txt` -> renamed to `ground_truth/Dennis Skinner kicked out of Commons.txt` (R100).
       - `ground_truth/Good Morning Britain vide_2.txt` -> renamed to `ground_truth/Piers Confronts Tommy Robinson.txt` (R100).
       - `ground_truth/Police bodycam shows arrest of Southwest Airlines pilot for alleged alcohol consumption 1.txt` -> renamed to `ground_truth/Police bodycam shows arrest of Southwest Airlines pilot.txt` (R100).
       - `ground_truth/bwc.txt` -> renamed to `ground_truth/That was a bit of a cheap shot weren’t it mate.txt` (R100).
       - `ground_truth/UK_parliment_debate.txt` -> deleted and replaced by `ground_truth/UK Parliament Debate.txt` (+47 lines).
       - `ground_truth/Bodycam footage show arrests of Sara Sharif's alleged killers_1.txt` -> replaced by `ground_truth/Bodycam footage show arrests of Sara Sharif’s alleged killers.txt` (+30 lines).
       - `ground_truth/Karen Has A Problem With Disabled Parking.txt` -> newly added (+176 lines).
       - `ground_truth/Rich Karen Messes Around & Finds Out.txt` -> newly added (+560 lines).
     - **Ground-Truth File Deletions:**
       - `ground_truth/Drunk Karen Gets a Huge Reality Check.txt` (-872 lines).
       - `ground_truth/Entitled_karen_refuses_pay_bill.txt` (-723 lines).
       - `ground_truth/When an Entitled Girl Gets Instantly Humbled_3.txt` (-375 lines).
       - `ground_truth/Woman's Attitude Makes Traffic Stop 10x Worse.txt` (-819 lines).
       - `ground_truth/podcast.txt` (-73 lines).
       - `ground_truth/rich_karen.txt` (-569 lines).
     - **Raw Audio Changes:**
       - `raw_audio/.gitkeep` was deleted from Git tracking.
       - **Crucial Git Status Finding:** In `.gitignore` (lines 124–125), `raw_audio/*` is ignored with an exception `!raw_audio/.gitkeep`. When `raw_audio/.gitkeep` was deleted, Git stopped tracking the directory itself. However, **8 physical MP3 audio files exist on disk in `raw_audio/`**, matching the 8 ground-truth text files in `ground_truth/`.

---

## 3. Complete Directory Hierarchy

The root directory contains 41 major entries (excluding `.git`):

```text
Conversational-Intelligence-Platform/
├── asr_outputs_wx/                       # Historical WhisperX ASR outputs (JSON)
├── asr_outputs_wx_custom/                # Aligned ASR outputs from custom pipeline
├── asr_outputs_wx_custom_conditioned/    # ASR outputs conditioned on VAD
├── config/                               # Project keys and Codex configuration
├── configs/                              # Stage-1 pipeline YAML configs
├── diarization_grounded_test/            # Diagnostic grounded diarization outputs
├── diarization_outputs_pyannote_3.1/     # Pyannote 3.1 baseline diarization outputs
├── diarization_outputs_wx/               # WhisperX diarization outputs
├── documentation/                        # Project architecture and technical docs
├── error_analysis_A/                     # Path A error logs and metric breakdowns
├── error_analysis_B/                     # Path B error logs and metric breakdowns
├── error_analysis_C/                     # Path C error logs and metric breakdowns
├── error_analysis_hardened/              # Hardened pipeline error analysis
├── experiments/                          # 22 experimental phases (Phases 1-20, audits)
├── external/                             # Cloned dependencies: silero-vad, WhisperX platform
├── final_bwc_outputs/                    # Final baseline pipeline output JSON/RTTM
├── final_bwc_outputs_B/                  # Final Path B pipeline output JSON/RTTM
├── final_bwc_outputs_C/                  # Final Path C pipeline output JSON/RTTM
├── final_bwc_outputs_v2/                 # Version 2 outputs
├── final_bwc_outputs_v3/                 # Version 3 outputs
├── ground_truth/                         # 8 canonical reference transcripts (.txt)
├── ground_truth_json/                    # Ground-truth word & segment JSON (bwc, karen)
├── ground_truth_rttm/                    # Ground-truth speaker diarization RTTM (bwc, karen)
├── logs/                                 # Stage execution logs (.log)
├── papers/                               # Published academic reference PDFs
├── phase20_standalone/                   # Self-contained Phase 20 pipeline artifact
├── preprocessed_B_normalized/            # RMS-normalized audio files (bwc.wav, podcast.wav)
├── preprocessed_C_normalized_bandpass/   # Bandpass-filtered audio files
├── raw_audio/                            # 8 raw benchmark MP3 recordings
├── reports/                              # Historical Markdown and CSV evaluation reports
├── scripts/                              # Core pipeline scripts and utilities
├── speech_segments/                      # Pre-sliced speech chunks (WAV)
├── speech_segments_whisperx/             # Speech segment manifests for WhisperX
├── transcripts_custom/                   # Custom pipeline transcripts
├── transcripts_custom_conditioned/       # Conditioned pipeline transcripts
├── vad_aug_temp/                         # Synthetic VAD noise augmentation benchmark suite
├── vad_custom_grounding/                 # Custom VAD grounding timestamps (JSON)
├── vad_custom_grounding_B/               # Path B VAD grounding timestamps
├── vad_custom_grounding_C/               # Path C VAD grounding timestamps
├── vad_outputs/                          # Raw Pyannote VAD segment JSON
├── vad_outputs_whisperx/                 # Raw WhisperX VAD segment JSON
└── [44 root Python scripts & 20 root markdown/text/config files]
```

---

## 4. File Counts and Storage Breakdown

### By Top Directory

| Directory Path | File Count | Size (MB) | Primary Content Types |
| :--- | :---: | :---: | :--- |
| `experiments/` | 2,602 | 310.89 | `.png` (775), `.json` (718), `.csv` (496), `.py` (220), `.md` (215) |
| `preprocessed_B_normalized/` | 3 | 30.32 | `.wav` (2 files, 30.32 MB), `.gitkeep` |
| `preprocessed_C_normalized_bandpass/` | 3 | 30.32 | `.wav` (2 files, 30.32 MB), `.gitkeep` |
| `speech_segments/` | 26 | 27.07 | `.wav` (26 segmented WAV files) |
| `raw_audio/` | 8 | 26.70 | `.mp3` (8 benchmark recordings) |
| `papers/` | 9 | 11.94 | `.pdf` (9 academic research papers) |
| `external/` | 108 | 11.06 | Silero VAD (`.jit`, `.onnx`), WhisperX platform wrapper |
| `phase20_standalone/` | 122 | 1.02 | `.py` (21), `.md` (23), `.json` (35), `.txt` (13) |
| Root level files | 64 | 11.75 | `.pdf` (1, 3.14 MB), `.png` (2, 3.31 MB), `.py` (44), `.md` (12) |
| `ground_truth/` | 8 | 0.04 | `.txt` (8 reference transcripts) |
| `ground_truth_json/` | 2 | 0.04 | `.json` (`bwc.json`, `karen.json`) |
| `ground_truth_rttm/` | 2 | 0.02 | `.rttm` (`bwc.rttm`, `karen.rttm`) |
| Other directories combined | 186 | 3.52 | Cached outputs, logs, intermediate VAD/ASR JSON |
| **Total** | **3,133** | **464.69 MB** | Full repository on disk |

### By File Extension

| Extension | Count | Size (MB) | Description |
| :--- | :---: | :---: | :--- |
| `.png` | 778 | 253.89 | Metric visualizations, error plots, HLA architecture diagrams |
| `.wav` | 30 | 87.71 | Raw preprocessed audio and segmented audio chunks |
| `.json` | 846 | 30.73 | ASR transcripts, VAD boundaries, word alignments, evaluations |
| `.mp3` | 8 | 26.70 | Raw bodycam and speech recordings |
| `.pkl` | 45 | 21.22 | Pickled speaker embeddings and clustering diagnostics |
| `.pdf` | 10 | 15.08 | Academic literature and BWC research report |
| `.html` | 2 | 9.29 | Interactive error analysis reports |
| `.onnx` | 4 | 7.38 | Silero VAD ONNX model checkpoints in `external/` |
| `.csv` | 501 | 3.53 | Benchmark metrics, stage timings, ablation results |
| `.jit` | 1 | 2.17 | Silero VAD PyTorch JIT model checkpoint |
| `.md` | 317 | 2.16 | Documentation, phase reports, architectural logs |
| `.py` | 312 | 1.26 | Pipeline stages, evaluation scripts, orchestration code |
| `.txt` | 97 | 0.93 | Reference transcripts, environment snapshots, logs |
| `.rttm` | 39 | 0.22 | Diarization reference and hypothesis timeline annotations |
| Other | 143 | 2.43 | YAML configs, `.log`, `.label`, `.toml` |

---

## 5. Additions and Modifications in the Latest Commit

Commit `f0a4ff6` focused entirely on cleaning and standardizing the audio and ground-truth pairing:

1. **Audio-to-Transcript Alignment:** 
   The commit establishes a 1-to-1 matching naming convention between the 8 raw audio files in `raw_audio/` and the 8 reference transcripts in `ground_truth/`.
2. **Expansion Beyond BWC:**
   While historical phases 1–20 focused predominantly on `bwc` ("That was a bit of a cheap shot weren’t it mate") and `karen` ("Rich Karen Messes Around & Finds Out"), the latest commit establishes 6 additional evaluation targets:
   - `Bodycam footage show arrests of Sara Sharif’s alleged killers` (100.96s)
   - `Dennis Skinner kicked out of Commons` (176.26s)
   - `Karen Has A Problem With Disabled Parking` (206.22s)
   - `Piers Confronts Tommy Robinson` (204.54s)
   - `Police bodycam shows arrest of Southwest Airlines pilot` (162.45s)
   - `UK Parliament Debate` (215.34s)
3. **Removal of Legacy Inconsistent GT:**
   Old unstructured transcripts (`Drunk Karen Gets a Huge Reality Check.txt`, `Entitled_karen_refuses_pay_bill.txt`, `podcast.txt`, `rich_karen.txt`) were removed or cleanly standardized into the new naming scheme.

---

## 6. Directory Interdependencies and Coupling

Our forensic scan revealed significant structural coupling across historical folders:

1. **Root Scripts depend on `external/WhisperX-Audio-Intelligence-Platform/.env`:**
   Root execution scripts (`run_asr.py`, `run_diarization.py`, `run_final_hardened_pipeline.py`, `run_production_pipeline.py`, `run_stage4_parallel.py`) dynamically read `.env` from `external/WhisperX-Audio-Intelligence-Platform/.env` rather than root.
2. **Root Scripts depend on Hardcoded Directories:**
   The root scripts write and read directly from flat directories: `vad_outputs_whisperx`, `asr_outputs_wx`, `diarization_outputs_wx`, `speech_segments_whisperx`.
3. **Experiments Interdependence:**
   Multiple experimental phases in `experiments/` depend on intermediate outputs created by prior phases (e.g. `phase11_dfn_ablation` consumes audio from `phase10_dual_branch_fusion`; `phase19` references `phase18`).
4. **Self-Containment of `phase20_standalone`:**
   In contrast to the root scripts and experiments, **`phase20_standalone` is almost completely decoupled from the rest of the repository**. It encapsulates its own `scripts/`, `fusion/`, `grounding/`, `diarization/`, `metrics/`, `reports/`, and internal fallbacks. It only references its own directory tree (`PROJECT_ROOT = Path(__file__).resolve().parents[1]`).

---

## 7. Sensitive Files, Credentials, and Security Findings

> [!WARNING] Active Secrets Detected in Git History
> The following sensitive tokens and secrets are currently tracked directly in the Git repository:
> 
> 1. `config/secret_key`: Contains an active Django secret key: `p-5t$+hv1i&se#t9+zd*!_#&y9suyo1il$1k6*cm7z$ct1q9j@`.
> 2. `config/field_encryption_key`: Contains a base64-encoded Fernet symmetric encryption key: `gparkF0Pt7WjwI3fM5HnwnAr67qG8T0r_8Lezwdbryo=`.
> 3. `external/WhisperX-Audio-Intelligence-Platform/.env.example` & `phase20_standalone/.env.example`: Contain environment placeholders for `HF_TOKEN`.

**Remediation Recommendation for `app/`:**
- Do NOT copy `config/secret_key` or `config/field_encryption_key` into `app/`.
- Provide an `.env.example` inside `app/` requiring the user to supply `HF_TOKEN`, `SECRET_KEY`, and `ENCRYPTION_KEY` at runtime via standard environment variables or a `.env` file excluded by `.gitignore`.

---

## 8. Areas Requiring Further Forensic Inspection

1. **Audio Tracking Policy:** `raw_audio/` is currently ignored by Git due to `.gitignore:124`. Because `app/` must be independently runnable, a decision is needed on whether sample audio fixtures (e.g. a 5-second synthetic or public clip) should be bundled or ingested at runtime.
2. **RTTM and JSON Availability for New Datasets:** Only `bwc` and `karen` have verified `ground_truth_json` and `ground_truth_rttm`. The other 6 recordings in the latest commit currently only have `.txt` reference transcripts.
3. **Execution Verification of `phase20_standalone`:** Verifying whether `phase20_standalone/run_phase20.py` can execute end-to-end with deterministic fallbacks when GPU and HF tokens are absent.
