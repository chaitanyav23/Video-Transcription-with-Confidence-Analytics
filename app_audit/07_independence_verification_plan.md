# 07 — Standalone Independence Verification Strategy

## 1. Objective and Independence Mandate

The standalone application at `<PROJECT_ROOT>/app/` must be capable of being physically isolated from the research repository and executed on any clean Linux or containerized system without access to the parent git repository or historical phase directories.

To prove that `app/` achieves 100% independence, we establish a two-phase verification protocol:
1. **Static AST and Path Forensics:** Automated inspection of all code, configs, and scripts inside `app/` to ensure zero references to parent directories.
2. **Isolated Sandbox Runtime Verification:** Physical extraction of `app/` into an isolated temporary workspace (`/tmp/standalone_app_test` or `C:\Temp\standalone_app_test`) followed by execution of a multi-tiered test suite.

---

## 2. Static Dependency Verification Suite

We define a dedicated verification script `app/tests/test_imports.py` that executes during CI:

```python
"""app/tests/test_imports.py - Static AST Independence Verification."""
import ast
import os
import re
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATTERNS = [
    r"phase\d+",
    r"experiments",
    r"preprocessed_",
    r"vad_aug_temp",
    r"speech_segments",
    r"\.\./\.\.",  # Path escaping app/
    r"Conversational-Intelligence-Platform",
]

def test_no_escaping_imports():
    """Verify that every python file inside app/ only imports internal modules or installed packages."""
    violations = []
    for py_file in APP_ROOT.rglob("*.py"):
        if ".venv" in py_file.parts or "node_modules" in py_file.parts:
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", None) or ""
                for alias in node.names:
                    full_name = f"{mod}.{alias.name}" if mod else alias.name
                    for pat in FORBIDDEN_PATTERNS:
                        if re.search(pat, full_name, re.IGNORECASE):
                            violations.append((py_file.relative_to(APP_ROOT), full_name))
    assert not violations, f"Forbidden external imports detected: {violations}"

def test_no_hardcoded_external_paths():
    """Verify that no string literal points to repository root or parent directories."""
    violations = []
    for p in APP_ROOT.rglob("*"):
        if p.is_dir() or p.suffix not in [".py", ".sh", ".yaml", ".json", ".toml", ".ts", ".tsx"]:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in [r"\.\./[a-zA-Z]", r"experiments/", r"phase20_standalone/"]:
            if re.search(pat, text):
                violations.append((p.relative_to(APP_ROOT), pat))
    assert not violations, f"Escaping file paths detected: {violations}"

def test_no_symlinks():
    """Verify no symlinks exist that escape the app directory."""
    violations = []
    for p in APP_ROOT.rglob("*"):
        if p.is_symlink():
            target = p.resolve()
            if not target.is_relative_to(APP_ROOT):
                violations.append((p.relative_to(APP_ROOT), target))
    assert not violations, f"Escaping symlinks detected: {violations}"
```

---

## 3. Isolated Sandbox Runtime Verification Protocol

The verification protocol copies `app/` to an isolated environment with no access to the parent git repository:

```bash
# Step 1: Copy app to an isolated temporary sandbox
rm -rf /tmp/standalone_app_test
cp -a app /tmp/standalone_app_test
cd /tmp/standalone_app_test

# Step 2: Ensure zero Git parent history
test ! -d ../.git && echo "Verified: Isolated from Git"

# Step 3: Create clean virtual environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Step 4: Install standalone core application
pip install -e .
pip install -e .[dev]

# Step 5: Execute tiered verification battery
pytest tests/test_imports.py -v
pytest tests/test_backend_api.py -v
pytest tests/test_pipeline_mock.py -v
```

---

## 4. Multi-Tiered Verification Test Battery

| Tier | Test Scope | Execution Command | Required Assets / Mocks | Expected Outcome |
| :---: | :--- | :--- | :--- | :--- |
| **Tier 1** | **Static AST & Path Scan** | `pytest tests/test_imports.py` | None | **0 forbidden imports or escaping relative paths.** |
| **Tier 2** | **Backend Module Import** | `python -c "import backend.app.main; print('Backend loaded')"` | Virtualenv only | Clean import without errors. |
| **Tier 3** | **Database Initialization** | `python scripts/init_db.py` | SQLite (in `storage/`) | Tables created: `recordings`, `jobs`, `transcripts`. |
| **Tier 4** | **API Health & Endpoints** | `pytest backend/tests/test_api/` | TestClient (FastAPI) | `GET /api/v1/health` returns `{"status": "ok"}`. |
| **Tier 5** | **Engine Unit Tests** | `pytest engine/tests/` | None (Mock Tensors) | Audio preprocessor, loudness normalizer, and Hungarian solver pass. |
| **Tier 6** | **Synthetic Audio E2E (Mock)**| `pytest tests/test_pipeline_mock.py` | 5-second synthetic WAV fixture | Complete pipeline executes (VAD -> ASR -> MSDD -> Fusion -> Evaluator) producing valid attributed transcript JSON in < 5 seconds on CPU. |
| **Tier 7** | **Frontend Production Build** | `cd frontend && npm install && npm run build` | Node.js 18+ | Clean TypeScript compilation; `frontend/dist/` generated. |
| **Tier 8** | **Full GPU Inference** | `python -m engine.pipeline.orchestrator --file bwc` | GPU, `HF_TOKEN`, downloaded weights | Full pipeline execution matching Phase 20 ground-truth metrics (cpWER < 0.15). |

> [!NOTE] Tier 1 through Tier 7 run fully on CPU in CI/CD without HuggingFace tokens or GPU hardware. Tier 8 runs in a dedicated GPU deployment container.

---

## 5. Component Provenance Tracking Framework

Every research component migrated from `phase20_standalone` to `app/engine/` must maintain complete provenance documentation in `app/engine/PROVENANCE.md`:

### Provenance Audit Matrix

| New Module in `app/` | Source Research Component | Original Commit | Modifications Made | Verification Status |
| :--- | :--- | :---: | :--- | :---: |
| `app/engine/preprocessing/audio.py` | `phase20_standalone/scripts/step01_preprocess.py`<br>`scripts/step02b_loudnorm.py` | `dc2e329` | Replaced CLI argparse with `AudioPreprocessor` class; added tempfile isolation and FFprobe stream validation. | Verified by unit test |
| `app/engine/vad/pyannote_vad.py` | `phase20_standalone/scripts/step03_vad.py` | `dc2e329` | Refactored recursive slicer into standalone sub-module; added structured Pydantic return types; preserved reference fallback. | Verified by mock test |
| `app/engine/asr/whisper_engine.py` | `phase20_standalone/scripts/step05b_whisperx.py` | `dc2e329` | Encapsulated Faster-Whisper and Wav2Vec2 aligner into `WhisperXASREngine`; added fallback word generator. | Verified by mock test |
| `app/engine/diarization/nemo_msdd.py`| `phase20_standalone/scripts/step05a_msdd.py` | `dc2e329` | Refactored NeMo MSDD into `NeMoDiarizer` class; dynamically creates temp manifests; handles clean CUDA memory release. | Verified by unit test |
| `app/engine/fusion/word_midpoint.py` | `phase20_standalone/scripts/step07_fusion.py` | `dc2e329` | Encapsulated word-midpoint assignment and turn grouping; returns typed `ConversationTurn` Pydantic models. | Verified by synthetic test |
| `app/engine/evaluation/metrics.py` | `phase20_standalone/scripts/cpwer.py`<br>`scripts/sawer.py`<br>`scripts/speaker_mapping.py` | `dc2e329` | Unified Hungarian assignment and Levenshtein permutation calculation into a unified `EvaluationEngine`. | Verified vs MeetEval spec |
