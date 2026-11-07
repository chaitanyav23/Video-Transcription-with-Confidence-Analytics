"""
Transcript Dashboard  [VIZ component]
Simple, accessible Streamlit visualisation of a speech-to-text JSON file
(words + segments with timings, confidence and speaker labels).

Run:  streamlit run app.py

Accessibility features
- Colourblind-safe palettes (Okabe-Ito etc.) + greyscale option
- Colour is NEVER the only signal: confidence tiers also use shapes, symbols,
  underline styles and text labels
- Every chart has a written description + a data table alternative
- Logical heading order, plain-text transcript view, adjustable font size
- Works with keyboard only; all content is real text (screen-reader friendly)
"""
import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Transcript Dashboard", page_icon="🎙️", layout="wide")

# ---------------------------------------------------------------- palettes
PALETTES = {
    "Okabe-Ito (colourblind-safe)": {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2"},
    "Blue / Orange / Purple (colourblind-safe)": {"low": "#E66100", "mid": "#5D3A9B", "high": "#005AB5"},
    "Greyscale (print / low vision)": {"low": "#000000", "mid": "#666666", "high": "#AAAAAA"},
}
TIER_LABEL = {"low": "Low confidence", "mid": "Medium confidence", "high": "High confidence"}
TIER_SHAPE = {"low": "triangle-up", "mid": "square", "high": "circle"}
TIER_SYMBOL = {"low": "▲", "mid": "■", "high": "●"}
TIERS = ["low", "mid", "high"]


# ---------------------------------------------------------------- data
@st.cache_data
def load(raw: str):
    data = json.loads(raw)
    words = pd.DataFrame(data["words"])
    words["duration"] = words["end"] - words["start"]
    segs = pd.DataFrame([{k: v for k, v in s.items() if k != "words"} for s in data["segments"]])
    segs["segment"] = [f"Segment {i + 1}" for i in range(len(segs))]
    segs["duration"] = segs["end"] - segs["start"]
    segs["n_words"] = [len(s.get("words", [])) for s in data["segments"]]
    segs["gap_before"] = (segs["start"] - segs["end"].shift(1)).fillna(0)
    return data.get("file", "unknown"), data.get("method", "unknown"), words, segs


def tier(conf, low_t, high_t):
    if conf < low_t:
        return "low"
    if conf < high_t:
        return "mid"
    return "high"


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Transcript JSON file", type="json")
default_path = Path("attributed_words.json")

raw = None
if uploaded is not None:
    raw = uploaded.getvalue().decode("utf-8")
elif default_path.exists():
    raw = default_path.read_text(encoding="utf-8")

palette_name = st.sidebar.selectbox("Colour scheme", list(PALETTES))
colors = PALETTES[palette_name]
font_px = st.sidebar.slider("Transcript text size (px)", 14, 32, 18)
low_t = st.sidebar.slider("'Low confidence' below", 0.0, 1.0, 0.30, 0.05)
high_t = st.sidebar.slider("'High confidence' from", 0.0, 1.0, 0.60, 0.05)
if high_t < low_t:
    st.sidebar.warning("The high threshold is lower than the low threshold. Adjust one of them.")
    high_t = low_t

# ---------------------------------------------------------------- header
st.title("Transcript Dashboard")
st.write(
    "Visualises a speech-to-text result: what was said, when, and how reliable "
    "each word is. All charts have a text description and a data table."
)

if raw is None:
    st.info("Upload a transcript JSON file in the sidebar to begin.")
    st.stop()

file_name, method, words, segs = load(raw)
words["tier"] = words["confidence"].apply(lambda c: tier(c, low_t, high_t))
words["Confidence level"] = words["tier"].map(TIER_LABEL)
words["Marker"] = words["tier"].map(TIER_SYMBOL)

color_scale = alt.Scale(
    domain=[TIER_LABEL[t] for t in TIERS], range=[colors[t] for t in TIERS]
)
shape_scale = alt.Scale(
    domain=[TIER_LABEL[t] for t in TIERS], range=[TIER_SHAPE[t] for t in TIERS]
)

# ---------------------------------------------------------------- summary
total_dur = float(words["end"].max())
n_words = len(words)
mean_conf = float(words["confidence"].mean())
n_low = int((words["tier"] == "low").sum())
speakers = words["speaker"].nunique()
wpm = n_words / (total_dur / 60) if total_dur else 0

st.header("1. Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Audio length", f"{total_dur:.1f} s")
c2.metric("Words", n_words)
c3.metric("Average confidence", f"{mean_conf:.0%}")
c4.metric("Low-confidence words", f"{n_low} ({n_low / n_words:.0%})")
c5.metric("Speakers detected", speakers)

st.write(
    f"**Text summary:** file `{file_name}` (method: {method}) is {total_dur:.1f} seconds long and "
    f"contains {n_words} words in {len(segs)} segments, about {wpm:.0f} words per minute. "
    f"Average confidence is {mean_conf:.0%}. {n_low} words ({n_low / n_words:.0%}) fall below the "
    f"low-confidence threshold of {low_t:.2f}. "
    + (
        "Only one speaker label was found, so the speakers were not separated."
        if speakers == 1
        else f"{speakers} speaker labels were found."
    )
)

tab_tx, tab_charts, tab_data = st.tabs(["Transcript", "Charts", "Data tables"])

# ---------------------------------------------------------------- transcript
with tab_tx:
    st.header("2. Transcript with confidence markers")
    st.write(
        "Each word is marked by confidence in more than one way, so it does not rely on colour alone:"
    )
    st.markdown(
        f"- {TIER_SYMBOL['low']} **Low** (below {low_t:.2f}): bold, double underline, marked with ▲\n"
        f"- {TIER_SYMBOL['mid']} **Medium**: dotted underline, marked with ■ (hidden in text view)\n"
        f"- {TIER_SYMBOL['high']} **High** (from {high_t:.2f}): plain text"
    )

    view = st.radio(
        "View", ["Styled (colour + underline)", "Plain text (best for screen readers)"], horizontal=True
    )

    for _, seg in segs.iterrows():
        sw = words[(words["start"] >= seg["start"] - 1e-6) & (words["end"] <= seg["end"] + 1e-6)]
        st.subheader(f"{seg['segment']}  ({seg['start']:.1f}s to {seg['end']:.1f}s)")
        if view.startswith("Styled"):
            parts = []
            for _, w in sw.iterrows():
                txt = html.escape(w["word"])
                t = w["tier"]
                title = f"{TIER_LABEL[t]}: {w['confidence']:.2f}"
                if t == "low":
                    style = f"font-weight:700;color:{colors['low']};text-decoration:underline double"
                    txt += "▲"
                elif t == "mid":
                    style = f"color:{colors['mid']};text-decoration:underline dotted"
                    txt += "■"
                else:
                    style = ""
                parts.append(f'<span title="{title}" style="{style}">{txt}</span>')
            st.markdown(
                f'<p style="font-size:{font_px}px;line-height:1.7">{" ".join(parts)}</p>',
                unsafe_allow_html=True,
            )
        else:
            plain = " ".join(
                f"[{w['word']}?]" if w["tier"] == "low" else w["word"] for _, w in sw.iterrows()
            )
            st.markdown(f'<p style="font-size:{font_px}px;line-height:1.7">{html.escape(plain)}</p>',
                        unsafe_allow_html=True)
            st.caption("Words in [square brackets with ?] are low confidence.")

# ---------------------------------------------------------------- charts
with tab_charts:
    st.header("3. Charts")

    # Chart A: confidence over time
    st.subheader("3.1 Confidence of each word over time")
    st.caption(
        "Each point is a word. Horizontal axis: time in the audio (seconds). Vertical axis: model "
        "confidence from 0 to 1. Triangles are low, squares medium, circles high confidence."
    )
    scatter = (
        alt.Chart(words)
        .mark_point(filled=True, size=70, opacity=0.85)
        .encode(
            x=alt.X("start:Q", title="Time (seconds)"),
            y=alt.Y("confidence:Q", title="Confidence (0 to 1)", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Confidence level:N", scale=color_scale, legend=alt.Legend(title="Level")),
            shape=alt.Shape("Confidence level:N", scale=shape_scale),
            tooltip=["word", "start", "end", "confidence", "Confidence level"],
        )
        .properties(height=320)
        .interactive()
    )
    st.altair_chart(scatter, width="stretch")
    worst = words.nsmallest(1, "confidence").iloc[0]
    st.write(
        f"**Text alternative:** the {n_low} low-confidence words are spread through the recording; "
        f"the lowest is “{worst['word']}” at {worst['start']:.1f}s (confidence {worst['confidence']:.2f})."
    )

    # Chart B: histogram
    st.subheader("3.2 Distribution of confidence scores")
    st.caption("How many words fall into each confidence range.")
    hist = (
        alt.Chart(words)
        .mark_bar()
        .encode(
            x=alt.X("confidence:Q", bin=alt.Bin(step=0.1), title="Confidence range"),
            y=alt.Y("count():Q", title="Number of words"),
            color=alt.Color("Confidence level:N", scale=color_scale, legend=alt.Legend(title="Level")),
            tooltip=["count()"],
        )
        .properties(height=280)
    )
    st.altair_chart(hist, width="stretch")
    counts = words["Confidence level"].value_counts()
    st.write(
        "**Text alternative:** "
        + "; ".join(f"{TIER_LABEL[t]}: {int(counts.get(TIER_LABEL[t], 0))} words" for t in TIERS)
        + "."
    )

    # Chart C: segment timeline
    st.subheader("3.3 Speech timeline (segments and silences)")
    st.caption("Each bar is a continuous stretch of speech. Empty space between bars is silence.")
    timeline = (
        alt.Chart(segs)
        .mark_bar(color=colors["high"], stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("start:Q", title="Time (seconds)"),
            x2="end:Q",
            y=alt.Y("segment:N", sort=None, title=None),
            tooltip=["segment", "start", "end", "duration", "n_words"],
        )
        .properties(height=40 * len(segs) + 40)
    )
    st.altair_chart(timeline, width="stretch")
    longest_gap = segs.loc[segs["gap_before"].idxmax()]
    st.write(
        f"**Text alternative:** speech is split into {len(segs)} segments. The longest silence "
        f"is {longest_gap['gap_before']:.1f}s, before {longest_gap['segment']}."
    )

    # Chart D: segment confidence
    st.subheader("3.4 Average confidence per segment")
    seg_conf = []
    for _, s in segs.iterrows():
        sw = words[(words["start"] >= s["start"] - 1e-6) & (words["end"] <= s["end"] + 1e-6)]
        seg_conf.append(sw["confidence"].mean())
    segs["avg_confidence"] = seg_conf
    bar = (
        alt.Chart(segs)
        .mark_bar(color=colors["high"], stroke="black", strokeWidth=1)
        .encode(
            x=alt.X("segment:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("avg_confidence:Q", title="Average confidence", scale=alt.Scale(domain=[0, 1])),
            tooltip=["segment", alt.Tooltip("avg_confidence:Q", format=".2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(bar, width="stretch")
    best = segs.loc[segs["avg_confidence"].idxmax()]
    low = segs.loc[segs["avg_confidence"].idxmin()]
    st.write(
        f"**Text alternative:** the most reliable segment is {best['segment']} "
        f"({best['avg_confidence']:.2f}); the least reliable is {low['segment']} ({low['avg_confidence']:.2f})."
    )

# ---------------------------------------------------------------- data
with tab_data:
    st.header("4. Data tables (accessible alternative to charts)")
    st.subheader("Segments")
    seg_table = segs[["segment", "start", "end", "duration", "n_words", "gap_before", "avg_confidence", "text"]]
    st.dataframe(seg_table, width="stretch", hide_index=True)

    st.subheader("Words")
    only_low = st.checkbox("Show only low-confidence words")
    wt = words[["word", "start", "end", "confidence", "Confidence level", "speaker"]]
    if only_low:
        wt = wt[words["tier"] == "low"]
    st.dataframe(wt, width="stretch", hide_index=True)
    st.download_button(
        "Download words as CSV", wt.to_csv(index=False).encode("utf-8"), "words.csv", "text/csv"
    )

st.caption("Colour schemes: Okabe-Ito and similar palettes are designed to be distinguishable with common colour-vision deficiencies.")