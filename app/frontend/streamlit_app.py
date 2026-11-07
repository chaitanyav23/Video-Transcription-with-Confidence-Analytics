"""Streamlit Web Application: Video Transcription with Confidence Analytics."""
from __future__ import annotations

import time
from typing import Any
import altair as alt
import pandas as pd
import requests
import streamlit as st

from frontend.api_client import APIClient
from frontend.components.charts import (
    render_confidence_histogram,
    render_confidence_scatter,
    render_reliability_diagram,
    render_segment_confidence_bar,
    render_speech_timeline,
)
from frontend.components.tables import render_accessible_tables
from frontend.components.transcript_view import render_accessible_transcript

st.set_page_config(
    page_title="Video Transcription with Confidence Analytics",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------- Visual Palettes & Tiers
PALETTES = {
    "Okabe-Ito (Colourblind-Safe)": {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2"},
    "Blue / Orange / Purple (Colourblind-Safe)": {"low": "#E66100", "mid": "#5D3A9B", "high": "#005AB5"},
    "Greyscale (High Contrast / Print)": {"low": "#000000", "mid": "#666666", "high": "#AAAAAA"},
}
TIER_LABEL = {"low": "Low confidence", "mid": "Medium confidence", "high": "High confidence"}
TIER_SHAPE = {"low": "triangle-up", "mid": "square", "high": "circle"}
TIER_SYMBOL = {"low": "▲", "mid": "■", "high": "●"}
TIERS = ["low", "mid", "high"]


def assign_tier(conf: float, low_t: float, high_t: float) -> str:
    if conf < low_t:
        return "low"
    elif conf < high_t:
        return "mid"
    return "high"


# ---------------------------------------------------------------- Sidebar Controls
st.sidebar.title("⚙️ Control Panel")
api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")
client = APIClient(api_url)

# Backend Health Status
try:
    health = client.health()
    st.sidebar.success(f"Backend Connected ({health.get('app', 'API')})")
except Exception:
    st.sidebar.error("Backend Disconnected. Start FastAPI backend with: uvicorn backend.main:app")

st.sidebar.markdown("---")
st.sidebar.subheader("Accessibility & View Settings")
palette_name = st.sidebar.selectbox("Colour Scheme", list(PALETTES))
colors = PALETTES[palette_name]
font_px = st.sidebar.slider("Transcript Text Size (px)", 14, 32, 18)
low_t = st.sidebar.slider("'Low Confidence' Below", 0.0, 1.0, 0.30, 0.05)
high_t = st.sidebar.slider("'High Confidence' From", 0.0, 1.0, 0.60, 0.05)
if high_t < low_t:
    st.sidebar.warning("High threshold is lower than low threshold.")
    high_t = low_t

st.sidebar.markdown("---")
st.sidebar.subheader("Phase 20 Speech Engine")
st.sidebar.markdown("• **ASR**: Faster-Whisper `large-v3`\n• **VAD**: Pyannote / Silero 16kHz\n• **Attribution**: Maximum-Overlap Midpoint Fusion\n• **Calibration**: Isotonic Regression (ECE < 0.05)")
domain_prompt = st.sidebar.text_input("Contextual Vocabulary Prompt (Optional)", value="")


# ---------------------------------------------------------------- Main Header
st.title("🎙️ Video Transcription with Confidence Analytics")
st.markdown(
    "**CS698B Conversational Intelligence Platform** | Real Pretrained Speech Intelligence, "
    "Calibrated Confidence Estimation, and Interactive Review."
)

# Workflow Selection
action_tab = st.radio(
    "Choose Mode:",
    ["🎯 Select Benchmark Audio Recording", "📤 Upload New Audio / Video", "📂 Past Ingested Recordings"],
    horizontal=True,
)

current_recording_id: str | None = None
current_filename: str | None = None

if action_tab == "🎯 Select Benchmark Audio Recording":
    from pathlib import Path
    st.info("Benchmark Mode: Select one of the verified project benchmark recordings.")
    
    # Locate candidate benchmark directories
    possible_dirs = [
        Path("../raw_audio"),
        Path("raw_audio"),
        Path("../preprocessed_B_normalized"),
        Path("storage/uploads"),
    ]
    benchmark_files = []
    for d in possible_dirs:
        if d.exists():
            for p in d.glob("*.*"):
                if p.suffix.lower() in (".mp3", ".wav", ".mp4", ".m4a") and not p.name.startswith("."):
                    benchmark_files.append(p)

    if benchmark_files:
        benchmark_map = {f"{p.name} ({p.stat().st_size / (1024*1024):.1f} MB)": p for p in benchmark_files}
        chosen_label = st.selectbox("Choose Benchmark Audio:", list(benchmark_map.keys()))
        chosen_path = benchmark_map[chosen_label]

        if st.button("Run Phase 20 Pipeline on Selected Recording", type="primary"):
            with st.spinner(f"Ingesting {chosen_path.name}..."):
                try:
                    file_bytes = chosen_path.read_bytes()
                    upload_res = client.upload_media(file_bytes, chosen_path.name)
                    rec_id = upload_res["recording_id"]
                    st.session_state["recording_id"] = rec_id
                    st.session_state["filename"] = upload_res["filename"]

                    # Trigger real transcription job
                    client.trigger_transcription(rec_id, use_mock=False, initial_prompt=domain_prompt)
                    st.success(f"Job submitted for {chosen_path.name}! Polling real inference progress...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to submit benchmark: {e}")
    else:
        st.warning("No benchmark audio found in raw_audio/. Please upload a file via the Upload tab.")

elif action_tab == "📤 Upload New Audio / Video":
    uploaded_file = st.file_uploader(
        "Upload Media File (.mp4, .mov, .wav, .mp3, .m4a)",
        type=["mp4", "mov", "wav", "mp3", "m4a", "aac", "flac"],
    )
    if uploaded_file is not None:
        if st.button("Upload and Process Recording", type="primary"):
            with st.spinner("Uploading and validating media..."):
                try:
                    file_bytes = uploaded_file.getvalue()
                    upload_res = client.upload_media(file_bytes, uploaded_file.name)
                    rec_id = upload_res["recording_id"]
                    st.session_state["recording_id"] = rec_id
                    st.session_state["filename"] = upload_res["filename"]
                    
                    client.trigger_transcription(rec_id, use_mock=False, initial_prompt=domain_prompt)
                    st.success("Recording uploaded and Phase 20 pipeline dispatched!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")

elif action_tab == "📂 Past Ingested Recordings":
    try:
        recordings = client.list_recordings()
        if recordings:
            rec_options = {f"{r['filename']} ({r['recording_id'][:8]}...) - {r['duration']:.1f}s": r["recording_id"] for r in recordings}
            selected_label = st.selectbox("Select Recording", list(rec_options.keys()))
            if selected_label:
                st.session_state["recording_id"] = rec_options[selected_label]
                st.session_state["filename"] = selected_label.split(" (")[0]
        else:
            st.info("No past recordings found. Upload a recording or run the sample demo.")
    except Exception as e:
        st.warning(f"Could not load recording list: {e}")

current_recording_id = st.session_state.get("recording_id")
current_filename = st.session_state.get("filename", "recording.wav")

if not current_recording_id:
    st.info("Please load a sample recording or upload media above to view results.")
    st.stop()

# ---------------------------------------------------------------- Polling & Job Status
job_status = client.get_job_status(current_recording_id)
status_val = job_status.get("status", "unknown")
progress_val = float(job_status.get("progress", 0.0))
stage_val = job_status.get("stage", "idle")

if status_val in ("queued", "processing"):
    st.warning(f"⏳ **Processing in progress:** {stage_val}")
    st.progress(progress_val)
    time.sleep(1.0)
    st.rerun()
elif status_val == "failed":
    st.error(f"❌ Processing failed: {job_status.get('error')}")
    st.stop()

# Retrieve Transcript Result
transcript_res = client.get_transcript(current_recording_id)
if not transcript_res:
    st.info("No transcript found yet.")
    st.stop()

# Load DataFrames
raw_words = transcript_res.get("words", [])
raw_segs = transcript_res.get("segments", [])

words_df = pd.DataFrame(raw_words)
if not words_df.empty:
    words_df["duration"] = words_df["end"] - words_df["start"]
    words_df["tier"] = words_df["confidence"].apply(lambda c: assign_tier(c, low_t, high_t))
    words_df["Confidence level"] = words_df["tier"].map(TIER_LABEL)
    words_df["Marker"] = words_df["tier"].map(TIER_SYMBOL)

segs_df = pd.DataFrame([{k: v for k, v in s.items() if k != "words"} for s in raw_segs])
if not segs_df.empty:
    segs_df["segment"] = [f"Turn {i + 1}" for i in range(len(segs_df))]
    segs_df["duration"] = segs_df["end"] - segs_df["start"]
    segs_df["n_words"] = [len(s.get("words", [])) for s in raw_segs]

# Scales
color_scale = alt.Scale(domain=[TIER_LABEL[t] for t in TIERS], range=[colors[t] for t in TIERS])
shape_scale = alt.Scale(domain=[TIER_LABEL[t] for t in TIERS], range=[TIER_SHAPE[t] for t in TIERS])

# ---------------------------------------------------------------- Summary Cards
st.header("1. Executive Summary")
total_dur = float(transcript_res.get("duration", 0.0))
n_words = len(words_df)
mean_conf = float(words_df["confidence"].mean()) if not words_df.empty else 0.0
n_low = int((words_df["tier"] == "low").sum()) if not words_df.empty else 0
speakers_count = len(transcript_res.get("speakers", []))
wpm = (n_words / (total_dur / 60.0)) if total_dur > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Media Duration", f"{total_dur:.1f} s")
c2.metric("Total Words", n_words)
c3.metric("Avg Confidence", f"{mean_conf:.1%}")
c4.metric("Low-Confidence Words", f"{n_low} ({n_low / max(1, n_words):.0%})")
c5.metric("Speakers Detected", speakers_count)

# Audio playback player
stream_url = client.get_stream_url(current_recording_id)
st.subheader("🎧 Media Audio Playback")
st.audio(stream_url)

# ---------------------------------------------------------------- Main Dashboard Tabs
tab_tx, tab_charts, tab_data, tab_adaptive, tab_export = st.tabs([
    "📝 Transcript & Review",
    "📊 Confidence Analytics",
    "📋 Accessible Data Tables",
    "🧠 Adaptive Intelligence",
    "💾 Export Transcripts",
])

# ---------------------------------------------------------------- Tab 1: Transcript & Review
with tab_tx:
    st.subheader("Word-Level Transcript with Multi-Sensory Confidence Markers")
    
    def on_submit_correction(turn_id: str, old_t: str, new_t: str, new_spk: str):
        client.submit_feedback(
            recording_id=current_recording_id,
            turn_id=turn_id,
            original_text=old_t,
            corrected_text=new_t,
            corrected_speaker=new_spk,
        )

    render_accessible_transcript(
        segs_df=segs_df,
        words_df=words_df,
        colors=colors,
        tier_label=TIER_LABEL,
        tier_symbol=TIER_SYMBOL,
        low_t=low_t,
        high_t=high_t,
        font_px=font_px,
        on_submit_correction=on_submit_correction,
    )

# ---------------------------------------------------------------- Tab 2: Confidence Analytics
with tab_charts:
    st.header("Confidence Analytics & Reliability Visualizations")
    
    # Scatter
    st.subheader("1. Word Confidence Over Time")
    st.caption("Each point represents a recognized word. Triangles are low, squares medium, circles high confidence.")
    render_confidence_scatter(words_df, color_scale, shape_scale)
    
    # Histogram
    st.subheader("2. Distribution of Confidence Scores")
    st.caption("Histogram showing frequency of words across confidence intervals.")
    render_confidence_histogram(words_df, color_scale)

    # Speech Timeline
    st.subheader("3. Conversational Speech Turns & Silences")
    st.caption("Bars represent active speech turns. Intervening gaps represent silences.")
    render_speech_timeline(segs_df, colors["high"])

    # Segment Bar
    st.subheader("4. Average Confidence Per Turn")
    render_segment_confidence_bar(segs_df, colors["high"])

    # Reliability Diagram
    st.subheader("5. Calibration Reliability Diagram & Metrics")
    st.caption("Compares predicted confidence levels against observed empirical accuracy (Isotonic Regression).")
    try:
        analytics = client.get_analytics(current_recording_id)
        m1, m2 = st.columns(2)
        m1.metric("Brier Score", f"{analytics.get('brier_score', 0.0):.4f}")
        m2.metric("Expected Calibration Error (ECE)", f"{analytics.get('ece', 0.0):.4f}")
        render_reliability_diagram(analytics.get("reliability_diagram", []))
    except Exception as e:
        st.info(f"Calibration metrics unavailable: {e}")

# ---------------------------------------------------------------- Tab 3: Data Tables
with tab_data:
    render_accessible_tables(segs_df, words_df)

# ---------------------------------------------------------------- Tab 4: Adaptive Intelligence
with tab_adaptive:
    st.header("Adaptive Feedback and Terminology Boosting")
    st.markdown(
        "The system extracts verified domain vocabulary from user manual corrections and injects them "
        "as contextual prompts into subsequent Faster-Whisper recognition runs."
    )
    vocab = client.get_vocabulary()
    st.subheader(f"Accumulated Domain Vocabulary ({len(vocab)} terms)")
    if vocab:
        st.write(", ".join([f"`{v}`" for v in vocab]))
    else:
        st.info("No domain terms accumulated yet. Edit any turn in the Transcript tab to add corrections.")

    st.subheader("Dynamic Whisper Contextual Prompt Preview")
    prompt_preview = f"Domain terms: {', '.join(vocab[:25])}." if vocab else "None"
    st.code(prompt_preview, language="text")

# ---------------------------------------------------------------- Tab 5: Export
with tab_export:
    st.header("Download & Export Formatted Transcripts")
    st.markdown("Select an export format for the finalized speaker-attributed transcript:")
    
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    
    # JSON
    json_bytes = client.get_export_url(current_recording_id, "json")
    try:
        res_json = requests.get(json_bytes).text
        col_e1.download_button("📄 Download JSON", res_json, f"{current_filename}.json", "application/json")
    except Exception:
        col_e1.button("📄 Download JSON", disabled=True)

    # TXT
    txt_url = client.get_export_url(current_recording_id, "txt")
    try:
        res_txt = requests.get(txt_url).text
        col_e2.download_button("📝 Download TXT", res_txt, f"{current_filename}.txt", "text/plain")
    except Exception:
        col_e2.button("📝 Download TXT", disabled=True)

    # SRT
    srt_url = client.get_export_url(current_recording_id, "srt")
    try:
        res_srt = requests.get(srt_url).text
        col_e3.download_button("🎬 Download SubRip (SRT)", res_srt, f"{current_filename}.srt", "text/plain")
    except Exception:
        col_e3.button("🎬 Download SubRip (SRT)", disabled=True)

    # VTT
    vtt_url = client.get_export_url(current_recording_id, "vtt")
    try:
        res_vtt = requests.get(vtt_url).text
        col_e4.download_button("🌐 Download WebVTT", res_vtt, f"{current_filename}.vtt", "text/vtt")
    except Exception:
        col_e4.button("🌐 Download WebVTT", disabled=True)
