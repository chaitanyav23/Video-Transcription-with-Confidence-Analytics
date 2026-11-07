"""Integration tests for Phase20 native Wav2Vec2 forced-alignment pipeline.

Covers:
1. Aligner integration with ASR output.
2. Strict separation of ASR confidence, alignment score, and calibrated confidence.
3. Zero-duration segment edge case (status="fallback").
4. Unaligned text handling (status="unaligned").
5. Empty transcript handling.
6. Diarization boundary enforcement (SPEAKER_UNASSIGNED and UNAVAILABLE_PENDING_MSDD).
7. External RTTM speaker attribution (status="EXTERNAL_RTTM").
8. TranscriptionService persistence and multi-format exports.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf

from backend.config import settings
from backend.database.models import RecordingRecord, RecordingRepository, TranscriptRepository
from backend.services.export_service import ExportService
from backend.services.transcription_service import TranscriptionService
from engine.phase20_engine import Phase20SpeechEngine
from engine.schemas import (
    AlignedWord,
    CalibrationMetadata,
    ConfidenceFeatures,
    ConversationTurn,
    TranscriptResult,
)


@pytest.fixture
def dummy_audio_16k(tmp_path: Path) -> Path:
    """Generate a 2.0-second 16 kHz mono synthetic sine tone for pipeline tests."""
    audio_file = tmp_path / "test_tone_16k.wav"
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False, dtype=np.float32)
    waveform = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(audio_file), waveform, sr)
    return audio_file


def test_aligner_integration_with_synthetic_segment(dummy_audio_16k: Path):
    """Verify that Phase20SpeechEngine successfully aligns words and separates confidences."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8")

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.2
    mock_seg.end = 1.6
    mock_seg.text = "TESTING PIPELINE"
    mock_seg.avg_logprob = -0.22314355  # exp(-0.22314355) ~ 0.80
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [
        {"word": "TESTING", "start": 0.24, "end": 0.88, "confidence": 0.9421},
        {"word": "PIPELINE", "start": 0.92, "end": 1.56, "confidence": 0.8850},
    ]
    engine._aligner = mock_aligner

    result = engine.transcribe(
        dummy_audio_16k,
        recording_id="synth_test_01",
        enable_alignment=True,
    )

    assert isinstance(result, TranscriptResult)
    assert len(result.words) == 2

    w1 = result.words[0]
    assert w1.word == "TESTING"
    assert w1.start == 0.24
    assert w1.end == 0.88
    assert w1.alignment_status == "aligned"
    assert w1.alignment_score == 0.9421
    assert w1.raw_confidence == 0.9421
    assert w1.asr_confidence == pytest.approx(0.80, abs=0.01)
    # Check separate features
    assert w1.features is not None
    assert w1.features.alignment_status == "aligned"
    assert w1.features.alignment_score == 0.9421
    assert w1.features.asr_confidence == pytest.approx(0.80, abs=0.01)


def test_confidence_separation_integrity(dummy_audio_16k: Path):
    """Ensure asr_confidence, alignment_score, and confidence are distinct values."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8")

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.1
    mock_seg.end = 1.0
    mock_seg.text = "OFFICER"
    # avg_logprob of -0.5108 gives asr_confidence ~ 0.60
    mock_seg.avg_logprob = -0.5108
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    # CTC alignment confidence of 0.92
    mock_aligner.align_segment.return_value = [
        {"word": "OFFICER", "start": 0.15, "end": 0.95, "confidence": 0.9200},
    ]
    engine._aligner = mock_aligner

    result = engine.transcribe(dummy_audio_16k, recording_id="conf_sep_01")
    assert len(result.words) == 1
    w = result.words[0]

    # asr_confidence (~0.60) must NOT equal alignment_score (0.92)
    assert w.asr_confidence == pytest.approx(0.60, abs=0.01)
    assert w.alignment_score == 0.9200
    assert w.asr_confidence != w.alignment_score
    # Calibrated confidence comes from calibrator applied to raw_confidence (0.92)
    assert w.confidence > 0.0
    assert w.is_calibrated is True


def test_zero_duration_edge_case(dummy_audio_16k: Path):
    """Verify that zero-duration segments get status='fallback', alignment_score=None, confidence=0.0."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8")

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 1.050
    mock_seg.end = 1.050  # duration = 0.000s
    mock_seg.text = "GHOST TOKEN"
    mock_seg.avg_logprob = -0.6931  # ~ 0.50
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    engine._aligner = mock_aligner

    result = engine.transcribe(dummy_audio_16k, recording_id="zero_dur_test")
    # Aligner must NOT even be called for zero duration segment
    mock_aligner.align_segment.assert_not_called()

    assert len(result.words) == 2
    for w in result.words:
        assert w.alignment_status == "fallback"
        assert w.alignment_score is None
        assert w.confidence == 0.0
        assert w.asr_confidence == pytest.approx(0.50, abs=0.01)

    assert result.summary["fallback_word_count"] == 2
    assert result.summary["aligned_word_count"] == 0


def test_unaligned_text_handling(dummy_audio_16k: Path):
    """Verify that when alignment fails or is disabled, words are marked 'unaligned'."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8")

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.5
    mock_seg.end = 1.5
    mock_seg.text = "UNALIGNED WORDS"
    mock_seg.avg_logprob = -0.3
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    # Force aligner failure / exception
    mock_aligner = MagicMock()
    mock_aligner.align_segment.side_effect = RuntimeError("Acoustic matrix alignment error")
    engine._aligner = mock_aligner

    result = engine.transcribe(dummy_audio_16k, recording_id="unaligned_test")

    assert len(result.words) == 2
    for w in result.words:
        assert w.alignment_status == "unaligned"
        assert w.alignment_score is None
        assert w.confidence == 0.0

    assert result.summary["unaligned_word_count"] == 2


def test_empty_transcript_handling(dummy_audio_16k: Path):
    """Verify graceful handling when ASR produces no speech segments."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8")

    mock_whisper = MagicMock()
    mock_whisper.transcribe.return_value = ([], MagicMock())
    engine._whisper_model = mock_whisper

    result = engine.transcribe(dummy_audio_16k, recording_id="empty_test")
    assert len(result.words) == 0
    assert len(result.segments) == 0
    assert result.summary["word_count"] == 0
    assert result.summary["overall_confidence"] == 0.0
    assert result.summary["speakers_detected"] == 0


def test_diarization_blocker_disclosure(dummy_audio_16k: Path):
    """Verify that absent NeMo MSDD assigns SPEAKER_UNASSIGNED and sets explicit blocker."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8", hf_token=None)

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.2
    mock_seg.end = 1.0
    mock_seg.text = "HELLO THERE"
    mock_seg.avg_logprob = -0.1
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [
        {"word": "HELLO", "start": 0.22, "end": 0.55, "confidence": 0.95},
        {"word": "THERE", "start": 0.60, "end": 0.95, "confidence": 0.91},
    ]
    engine._aligner = mock_aligner

    result = engine.transcribe(dummy_audio_16k, recording_id="diar_blocker_test")

    # All words must be attributed to SPEAKER_UNASSIGNED
    for w in result.words:
        assert w.speaker == "SPEAKER_UNASSIGNED"

    for turn in result.segments:
        assert turn.speaker == "SPEAKER_UNASSIGNED"

    assert result.summary["diarization_status"] == "UNAVAILABLE_PENDING_MSDD"
    assert "NVIDIA NeMo MSDD / TitaNet" in result.summary["diarization_blocker"]


def test_external_rttm_attribution(dummy_audio_16k: Path, tmp_path: Path):
    """Verify that an external RTTM file attributes words to exact speakers and status='EXTERNAL_RTTM'."""
    engine = Phase20SpeechEngine(model_size="tiny", device="cpu", compute_type="int8")

    # Create dummy RTTM file
    rttm_path = tmp_path / "reference.rttm"
    rttm_content = (
        "SPEAKER dummy 1 0.100 0.700 <NA> <NA> DRIVER <NA> <NA>\n"
        "SPEAKER dummy 1 0.900 1.000 <NA> <NA> OFFICER <NA> <NA>\n"
    )
    rttm_path.write_text(rttm_content, encoding="utf-8")

    mock_whisper = MagicMock()
    mock_seg = MagicMock()
    mock_seg.start = 0.1
    mock_seg.end = 1.8
    mock_seg.text = "YES SIR"
    mock_seg.avg_logprob = -0.1
    mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())
    engine._whisper_model = mock_whisper

    mock_aligner = MagicMock()
    mock_aligner.align_segment.return_value = [
        {"word": "YES", "start": 0.20, "end": 0.60, "confidence": 0.95},
        {"word": "SIR", "start": 1.10, "end": 1.50, "confidence": 0.93},
    ]
    engine._aligner = mock_aligner

    result = engine.transcribe(
        dummy_audio_16k,
        recording_id="rttm_test",
        diarization_rttm=rttm_path,
    )

    assert result.summary["diarization_status"] == "EXTERNAL_RTTM"
    assert result.summary["diarization_blocker"] is None
    assert result.words[0].speaker == "DRIVER"
    assert result.words[1].speaker == "OFFICER"
    assert set(result.speakers) == {"DRIVER", "OFFICER"}


def test_transcription_service_persistence_and_export(dummy_audio_16k: Path, tmp_path: Path):
    """Verify TranscriptionService end-to-end DB persistence and multi-format export."""
    rec_id = "test_persistence_rec"
    rec = RecordingRecord(
        recording_id=rec_id,
        filename="dummy.wav",
        filepath=str(dummy_audio_16k),
        duration=2.0,
        size_bytes=dummy_audio_16k.stat().st_size,
        sha256="dummy_sha256",
        created_at="2026-11-07T12:00:00Z",
    )
    RecordingRepository.create(rec)

    feat = ConfidenceFeatures(
        acoustic_score=0.92,
        alignment_score=0.95,
        asr_confidence=0.92,
        alignment_status="aligned",
        duration=0.5,
    )
    w1 = AlignedWord(
        word="STOP",
        start=0.5,
        end=1.0,
        speaker="SPEAKER_UNASSIGNED",
        confidence=0.95,
        raw_confidence=0.95,
        alignment_status="aligned",
        alignment_score=0.95,
        asr_confidence=0.92,
        features=feat,
    )
    turn = ConversationTurn(
        turn_id="turn_001",
        speaker="SPEAKER_UNASSIGNED",
        start=0.5,
        end=1.0,
        text="STOP",
        words=[w1],
        avg_confidence=0.95,
    )
    mock_transcript = TranscriptResult(
        recording_id=rec_id,
        filename="dummy.wav",
        duration=2.0,
        calibration=CalibrationMetadata(method="isotonic_regression"),
        speakers=["SPEAKER_UNASSIGNED"],
        segments=[turn],
        words=[w1],
        summary={
            "overall_confidence": 0.95,
            "mean_confidence": 0.95,
            "word_count": 1,
            "aligned_word_count": 1,
            "fallback_word_count": 0,
            "unaligned_word_count": 0,
            "turn_count": 1,
            "speakers_detected": 1,
            "diarization_status": "UNAVAILABLE_PENDING_MSDD",
            "diarization_blocker": "NVIDIA NeMo MSDD / TitaNet diarization is not installed in the environment.",
        },
    )

    stub_engine = MagicMock()
    stub_engine.process.return_value = mock_transcript
    TranscriptionService.set_engine(stub_engine)

    with patch("backend.services.audio_service.AudioService.prepare_16k_mono_audio", return_value=dummy_audio_16k):
        result = TranscriptionService.execute_transcription(recording_id=rec_id)

    # 1. Verify returned result
    assert result.recording_id == rec_id
    assert len(result.words) == 1
    assert result.words[0].alignment_status == "aligned"

    # 2. Verify DB persistence
    db_record = TranscriptRepository.get(rec_id)
    assert db_record is not None
    assert db_record["recording_id"] == rec_id
    assert db_record["summary"]["diarization_status"] == "UNAVAILABLE_PENDING_MSDD"

    # 3. Verify filesystem JSON persistence
    json_path = settings.results_dir / f"{rec_id}.json"
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["recording_id"] == rec_id

    # 4. Verify ExportService formats (JSON, TXT, SRT, VTT)
    txt_out = ExportService.to_txt(result)
    assert "STOP" in txt_out

    srt_out = ExportService.to_srt(result)
    assert "00:00:00,500 --> 00:00:01,000" in srt_out
    assert "STOP" in srt_out

    vtt_out = ExportService.to_vtt(result)
    assert "WEBVTT" in vtt_out
    assert "00:00:00.500 --> 00:00:01.000" in vtt_out

    json_export = ExportService.to_json(result)
    assert "test_persistence_rec" in json_export
