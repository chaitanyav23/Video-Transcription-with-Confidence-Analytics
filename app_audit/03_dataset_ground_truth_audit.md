# 03 — Dataset and Ground-Truth Forensics

## 1. Raw Audio Forensics and Audio Manifest

The latest commit (`f0a4ff6`) standardized the raw audio recordings into `raw_audio/`. We performed direct forensic inspection of all physical audio files using `ffprobe` and SHA-256 cryptographic hashing.

### Canonical Raw Audio Manifest (`raw_audio/`)

| Recording Identifier | Filename | Format / Codec | Channels | Sample Rate (Hz) | Bitrate (kbps) | Duration (s) | File Size (Bytes) | SHA-256 Digest (First 16 chars) | Integrity Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `sara_sharif` | `Bodycam footage show arrests of Sara Sharif’s alleged killers.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~82 | 100.96 | 1,033,921 | `cb7e7f193927accf` | **Valid / Readable** |
| `dennis_skinner` | `Dennis Skinner kicked out of Commons.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~114 | 176.26 | 2,507,151 | `fff89079ddfba198` | **Valid / Readable** |
| `disabled_parking`| `Karen Has A Problem With Disabled Parking.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~185 | 206.22 | 4,778,186 | `9a57543f36b9e57c` | **Valid / Readable** |
| `piers_tommy` | `Piers Confronts Tommy Robinson.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~140 | 204.54 | 3,589,566 | `562e3a5e88ed51a2` | **Valid / Readable** |
| `southwest_pilot` | `Police bodycam shows arrest of Southwest Airlines pilot.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~106 | 162.45 | 2,157,836 | `d83fa8bd44b50edb` | **Valid / Readable** |
| `rich_karen` | `Rich Karen Messes Around & Finds Out.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~106 | 652.18 | 8,617,004 | `05a3461cec74b429` | **Valid / Readable** |
| `bwc` | `That was a bit of a cheap shot weren’t it mate.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~109 | 175.43 | 2,389,658 | `0a742882a00aa980` | **Valid / Readable** |
| `uk_parliament` | `UK Parliament Debate.mp3` | MP3 / mp3 | 2 (Stereo) | 44,100 | ~109 | 215.34 | 2,926,059 | `c1ae36884e098e48` | **Valid / Readable** |

### Key Raw Audio Findings:
1. **Zero Corrupted Files:** All 8 recordings parse cleanly with FFprobe without stream or header errors.
2. **Acoustic Characteristics:** All 8 recordings are stereo MP3s sampled at 44.1 kHz. They represent diverse conversational acoustics: high-stress body-worn camera stops (`bwc`, `southwest_pilot`, `sara_sharif`), argumentative citizen interactions (`rich_karen`, `disabled_parking`), studio debates (`piers_tommy`), and acoustic chamber room debates (`dennis_skinner`, `uk_parliament`).
3. **Identity of `bwc` and `rich_karen`:**
   - `That was a bit of a cheap shot weren’t it mate.mp3` is identical in duration (175.43s) and content to the canonical `bwc.wav` evaluated across all research phases (Phase 1 through Phase 20).
   - `Rich Karen Messes Around & Finds Out.mp3` is identical in duration (652.18s) to the canonical `karen` dataset used in Phase 11A, Phase 12, Phase 13, and Phase 18.

---

## 2. Ground-Truth Inventory and Parsing Forensics

The ground truth exists across three distinct formats in the repository:
1. **Raw Reference Transcripts (`ground_truth/*.txt`, 8 files):** Speaker-labeled, timestamped transcripts corresponding to each raw MP3 recording.
2. **Structured Ground-Truth JSON (`ground_truth_json/*.json`, 2 files):** Word-level and segment-level timestamps (`bwc.json`, `karen.json`).
3. **Diarization Reference RTTMs (`ground_truth_rttm/*.rttm`, 2 files):** Standard NIST RTTM speaker turn annotations (`bwc.rttm`, `karen.rttm`).

### Ground-Truth Text Transcripts (`ground_truth/`)

| Reference Transcript File | Lines | Words | Detected Speakers | Timestamp Format | Max GT Timestamp | Audio Duration | Compatibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Bodycam footage show arrests of Sara Sharif’s alleged killers.txt` | 19 | 342 | 5 (`Speaker_1` to `Speaker_5`) | `[HH:MM:SS - HH:MM:SS]` | 95.00s | 100.96s | **Compatible** |
| `Dennis Skinner kicked out of Commons.txt` | 30 | 424 | 5 (`Speaker_0` to `Speaker_4`) | `[HH:MM:SS - HH:MM:SS]` | 175.00s | 176.26s | **Compatible** |
| `Karen Has A Problem With Disabled Parking.txt` | 118 | 1,106 | 5 (`Speaker_0` to `Speaker_4`) | `[HH:MM:SS - HH:MM:SS]` | 206.00s | 206.22s | **Compatible** |
| `Piers Confronts Tommy Robinson.txt` | 66 | 1,118 | 3 (`Speaker_0` to `Speaker_2`) | `[HH:MM:SS – HH:MM:SS]` (en-dash) | 194.00s | 204.54s | **Compatible** |
| `Police bodycam shows arrest of Southwest Airlines pilot.txt` | 122 | 824 | 4 (`Speaker_1` to `Speaker_4`) | `[HH:MM:SS - HH:MM:SS]` | 160.00s | 162.45s | **Compatible** |
| `Rich Karen Messes Around & Finds Out.txt` | 374 | 3,222 | 9 (`Speaker_0` to `Speaker_8`) | `[HH:MM:SS - HH:MM:SS]` | 628.00s | 652.18s | **Compatible** |
| `That was a bit of a cheap shot weren’t it mate.txt` | 54 | 964 | 4 (`Speaker 1` to `Speaker 4`) | `[HH:MM:SS - HH:MM:SS]: inline` | 164.00s | 175.43s | **Compatible** |
| `UK Parliament Debate.txt` | 32 | 608 | 6 (`Speaker_0` to `Speaker_all`) | `[HH:MM:SS - HH:MM:SS]` | 215.00s | 215.34s | **Compatible** |

### Structured Benchmark References (`ground_truth_json/` & `ground_truth_rttm/`)

| Recording Identifier | JSON Annotation | JSON Segments | RTTM Annotation | RTTM Lines | Speakers in RTTM | Total Speech Duration |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| `bwc` | `ground_truth_json/bwc.json` | 48 segments | `ground_truth_rttm/bwc.rttm` | 48 lines | 4 (`Speaker_1` to `Speaker_4`) | 119.00s |
| `karen` | `ground_truth_json/karen.json` | 239 segments| `ground_truth_rttm/karen.rttm`| 239 lines | 6 (`Speaker_1` to `Speaker_6`) | 355.00s |

---

## 3. Recording-to-Ground-Truth Mapping Matrix

| Recording Identifier | Raw Audio Path | Duration | Matching Ground Truth | Evaluation Suitability | Notes & Forensic Observations |
| :--- | :--- | :---: | :--- | :---: | :--- |
| **`bwc`** | `raw_audio/That was a bit of a cheap shot weren’t it mate.mp3` | 175.43s | `ground_truth/That was a bit of a cheap shot...txt`<br>`ground_truth_json/bwc.json`<br>`ground_truth_rttm/bwc.rttm` | **Production Benchmark (Verified)** | Standard BWC police drug search in Peterborough. 4 speakers. Full word-level & RTTM ground truth verified across Phases 1–20. |
| **`rich_karen`** | `raw_audio/Rich Karen Messes Around & Finds Out.mp3` | 652.18s | `ground_truth/Rich Karen Messes Around & Finds Out.txt`<br>`ground_truth_json/karen.json`<br>`ground_truth_rttm/karen.rttm` | **Production Benchmark (Verified)** | Traffic stop escalation. 6–9 speakers. Full word-level & RTTM ground truth verified in Phase 11A/18. |
| **`sara_sharif`** | `raw_audio/Bodycam footage show arrests of Sara Sharif’s alleged killers.mp3` | 100.96s | `ground_truth/Bodycam footage show arrests...txt` | **Evaluation Ready (ASR + Diar)** | Bodycam arrest footage. 5 speakers. Needs JSON/RTTM parser step (`step00_ground_truth.py`) before metric computation. |
| **`dennis_skinner`** | `raw_audio/Dennis Skinner kicked out of Commons.mp3` | 176.26s | `ground_truth/Dennis Skinner kicked out of Commons.txt` | **Evaluation Ready (ASR + Diar)** | UK House of Commons debate. 5 speakers. Compatible timestamps. |
| **`disabled_parking`**| `raw_audio/Karen Has A Problem With Disabled Parking.mp3` | 206.22s | `ground_truth/Karen Has A Problem With Disabled Parking.txt` | **Evaluation Ready (ASR + Diar)** | High-noise outdoor dispute. 5 speakers. Compatible timestamps. |
| **`piers_tommy`** | `raw_audio/Piers Confronts Tommy Robinson.mp3` | 204.54s | `ground_truth/Piers Confronts Tommy Robinson.txt` | **Evaluation Ready (ASR + Diar)** | Studio interview with heavy speech overlap. 3 speakers. Uses en-dash in timestamps. |
| **`southwest_pilot`** | `raw_audio/Police bodycam shows arrest of Southwest Airlines pilot.mp3` | 162.45s | `ground_truth/Police bodycam shows arrest...txt` | **Evaluation Ready (ASR + Diar)** | Airport police bodycam. 4 speakers. Frequent short utterances. |
| **`uk_parliament`** | `raw_audio/UK Parliament Debate.mp3` | 215.34s | `ground_truth/UK Parliament Debate.txt` | **Evaluation Ready (ASR + Diar)** | Parliamentary debate. Includes collective speaker tag `Speaker_all`. |

---

## 4. Preprocessed and Intermediate Audio Inventory

Outside `raw_audio/`, the repository contains several intermediate audio collections:

1. **`preprocessed_B_normalized/` (30.32 MB):**
   - `bwc.wav` (175.43s, 16kHz Mono, 5.61 MB)
   - `podcast.wav` (818.04s, 16kHz Mono, 26.18 MB)
   - *Status:* Generated artifacts from RMS normalization. Ignored by Git.
2. **`preprocessed_C_normalized_bandpass/` (30.32 MB):**
   - `bwc.wav` (175.43s, 16kHz Mono bandpassed)
   - `podcast.wav` (818.04s, 16kHz Mono bandpassed)
   - *Status:* Generated artifacts from bandpass filter ablation. Ignored by Git.
3. **`speech_segments/` (27.07 MB):**
   - 14 segmented WAV files for `bwc` (`bwc_seg00` to `bwc_seg13`).
   - 12 segmented WAV files for `podcast` (`podcast_seg00` to `podcast_seg11`).
   - *Status:* Sliced audio chunks from VAD boundaries. Ignored by Git.

---

## 5. Dataset Classification and Boundary Policy for `app/`

| Classification Category | Repository Items | Retention Policy | Standalone `app/` Inclusion |
| :--- | :--- | :--- | :--- |
| **1. Essential Runtime Data** | None (The application processes user uploads) | N/A | **Do NOT bundle large datasets in `app/`.** |
| **2. Development / Test Fixtures** | 1 synthetic 5-second WAV fixture (`fixtures/sample.wav`), `ground_truth_json/bwc.json` (mock fallback) | Retain in `app/tests/fixtures/` | **Bundle minimal test fixture (< 1 MB) for unit & CI tests.** |
| **3. Evaluation / Benchmarking Data**| `raw_audio/` (8 MP3s), `ground_truth/` (8 TXTs), `ground_truth_json/`, `ground_truth_rttm/` | Retain in repository root / benchmark suite | Exclude from `app/` production package. Import on demand via evaluation CLI. |
| **4. Historical Research Data** | `experiments/*/audio/`, `speech_segments/`, `papers/` | Retain permanently in `experiments/` | Exclude from `app/`. |
| **5. Generated Intermediate Files**| `preprocessed_B_normalized/`, `preprocessed_C_...`, `vad_outputs/`, `asr_outputs_wx/` | Ephemeral / Reproducible | Exclude from `app/`. The app generates its own working files in `storage/`. |

---

## 6. Recommendations for `app/` Data Architecture

1. **Zero Hardcoded Data Paths:** `app/` must store all ingested user audio in a configurable storage root (`app/storage/uploads/` or S3/MinIO bucket).
2. **Deterministic Preprocessing:** Audio uploaded to `app/` must be converted via in-memory or tempfile FFmpeg pipelines into standard 16 kHz Mono WAV format and validated for duration and channel layout.
3. **Automated Content Hashing:** Every uploaded recording in `app/` must be hashed with SHA-256 upon ingestion to detect duplicates and maintain strict forensic audit trails.
4. **Evaluation CLI Plugin:** Benchmark datasets (`raw_audio/` and `ground_truth/`) should be accessible to developers via a dedicated standalone evaluation script (`app/scripts/run_benchmarks.py --data-dir <path>`) rather than baked into the application's runtime Docker image.
