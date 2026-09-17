"""
app.py: GridSense dashboard (Streamlit).

Design direction: a precision instrument in a dim plant room. Charcoal is
tinted warm (OKLCH hue 18) rather than navy, and colour appears ONLY as
machine state. The crimson that carries the brand IS the alarm colour, so
identity and semantics never compete.

Fonts are system stacks only (no webfonts), because the project must run
fully offline, and a Google Fonts <link> would break that promise.

Run:
    python run.py
"""

import io
import os
import glob
import random

import numpy as np
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from features import load_audio, extract_features, SAMPLE_RATE

MODEL_PATH = "gridsense_model.pkl"
DATA_DIR = "sample_data"

# --- Palette (OKLCH-derived, contrast-verified against the background) ------ #
BG       = "#0D0909"   # oklch(0.145 0.006 18)
SURFACE  = "#181414"   # oklch(0.196 0.007 18)
RAISED   = "#252020"   # oklch(0.248 0.008 18)
BORDER   = "#383131"   # oklch(0.320 0.010 18)
INK      = "#F5F3F3"   # 17.9:1 on bg
MUTED    = "#B1A8A8"   #  8.6:1 on bg
FAINT    = "#867E7E"   #  5.0:1 on bg
CRITICAL = "#E93955"   #  4.9:1 on bg, in-gamut crimson
WARNING  = "#F2B036"   # 10.4:1 on bg
HEALTHY  = "#4FCC8D"   #  9.8:1 on bg

STATUS = {
    "Healthy":          {"color": HEALTHY,  "dot": "●", "label": "Healthy"},
    "Early Warning":    {"color": WARNING,  "dot": "●", "label": "Early Warning"},
    "Critical Failure": {"color": CRITICAL, "dot": "●", "label": "Critical Failure"},
}

st.set_page_config(page_title="GridSense", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(f"""<style>
:root {{
  --bg:{BG}; --surface:{SURFACE}; --raised:{RAISED}; --border:{BORDER};
  --ink:{INK}; --muted:{MUTED}; --faint:{FAINT};
  --critical:{CRITICAL}; --warning:{WARNING}; --healthy:{HEALTHY};
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
  --mono:"Cascadia Code","Cascadia Mono",ui-monospace,Consolas,"SF Mono",Menlo,monospace;
  --ease:cubic-bezier(.22,1,.36,1);
}}
html, body, [class*="css"] {{ font-family: var(--sans); }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}

/* ---- Masthead ---- */
.gs-head {{
  display:flex; align-items:baseline; justify-content:space-between;
  gap:1rem; flex-wrap:wrap;
  border-bottom:1px solid var(--border); padding-bottom:1rem; margin-bottom:2rem;
}}
.gs-word {{ font-size:1.5rem; font-weight:650; letter-spacing:-0.02em; color:var(--ink); }}
.gs-word span {{ color:var(--critical); }}
.gs-tag {{ font-size:.875rem; color:var(--muted); margin-top:.25rem; }}
.gs-offline {{
  font-family:var(--mono); font-size:.75rem; color:var(--muted);
  border:1px solid var(--border); border-radius:999px; padding:.3rem .7rem;
  background:var(--surface); white-space:nowrap;
}}

/* ---- Verdict banner ---- */
.gs-verdict {{
  border:1px solid var(--border); border-radius:14px; background:var(--surface);
  padding:1.6rem 1.75rem; margin-bottom:1.25rem;
}}
.gs-verdict-top {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }}
.gs-status {{ font-size:1.75rem; font-weight:650; letter-spacing:-0.02em; line-height:1.1; }}
.gs-read {{ color:var(--muted); font-size:1rem; margin-top:.6rem; max-width:68ch; line-height:1.55; }}

/* ---- Health meter ---- */
.gs-meter-head {{ display:flex; justify-content:space-between; align-items:baseline; margin-top:1.4rem; }}
.gs-meter-label {{ font-size:.8rem; color:var(--faint); text-transform:none; }}
.gs-meter-val {{ font-family:var(--mono); font-size:1.6rem; font-weight:600; letter-spacing:-0.02em; }}
.gs-track {{
  position:relative; height:10px; border-radius:999px; background:var(--raised);
  margin-top:.5rem; overflow:hidden;
}}
.gs-fill {{ height:100%; border-radius:999px; transition:width .5s var(--ease); }}
.gs-ticks {{ position:relative; height:1.1rem; margin-top:.35rem; }}
.gs-tick {{
  position:absolute; transform:translateX(-50%);
  font-family:var(--mono); font-size:.7rem; color:var(--faint);
}}

/* ---- Measurement strip ---- */
.gs-strip {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px; background:var(--border); border:1px solid var(--border);
  border-radius:12px; overflow:hidden; margin-bottom:1.75rem;
}}
.gs-cell {{ background:var(--surface); padding:1rem 1.1rem; }}
.gs-cell-k {{ font-size:.78rem; color:var(--faint); margin-bottom:.35rem; }}
.gs-cell-v {{ font-family:var(--mono); font-size:1.15rem; font-weight:600; color:var(--ink); }}
.gs-cell-v small {{ font-size:.8rem; color:var(--muted); font-weight:400; }}

/* ---- Section headings ---- */
.gs-h {{
  font-size:1.05rem; font-weight:620; color:var(--ink); letter-spacing:-0.01em;
  margin:2.25rem 0 .4rem;
}}
.gs-sub {{ font-size:.875rem; color:var(--muted); margin-bottom:1rem; max-width:70ch; line-height:1.55; }}

/* ---- Confidence rows ---- */
.gs-conf {{ display:flex; flex-direction:column; gap:.7rem; }}
.gs-row {{ display:grid; grid-template-columns:110px 1fr 52px; align-items:center; gap:.85rem; }}
.gs-row-k {{ font-size:.875rem; color:var(--muted); }}
.gs-row-t {{ height:8px; border-radius:999px; background:var(--raised); overflow:hidden; }}
.gs-row-f {{ height:100%; border-radius:999px; transition:width .5s var(--ease); }}
.gs-row-v {{ font-family:var(--mono); font-size:.85rem; color:var(--muted); text-align:right; }}

/* ---- Steps (a genuine ordered sequence) ---- */
.gs-steps {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:1.5rem; margin:1.5rem 0 2.25rem; }}
.gs-step-n {{ font-family:var(--mono); font-size:.8rem; color:var(--critical); margin-bottom:.4rem; }}
.gs-step-t {{ font-weight:600; color:var(--ink); margin-bottom:.25rem; }}
.gs-step-d {{ font-size:.875rem; color:var(--muted); line-height:1.55; }}

/* ---- Recommendations ---- */
.gs-rec {{
  border:1px solid var(--border); border-radius:12px; background:var(--surface);
  padding:1.1rem 1.25rem; margin-bottom:.7rem;
}}
.gs-rec-t {{ font-weight:600; color:var(--ink); margin-bottom:.3rem; }}
.gs-rec-d {{ font-size:.9rem; color:var(--muted); line-height:1.6; }}

/* ---- Buttons ---- */
.stButton > button {{
  width:100%; background:var(--surface); color:var(--ink);
  border:1px solid var(--border); border-radius:10px;
  padding:.7rem 1rem; font-weight:550; font-size:.9rem;
  transition:background .18s var(--ease), border-color .18s var(--ease), transform .18s var(--ease);
}}
.stButton > button:hover {{ background:var(--raised); border-color:var(--faint); transform:translateY(-1px); }}
.stButton > button:focus-visible {{ outline:2px solid var(--critical); outline-offset:2px; }}
.stButton > button:active {{ transform:translateY(0); }}

[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--border); }}
[data-testid="stSidebar"] .stButton > button {{ background:var(--raised); }}

@media (prefers-reduced-motion: reduce) {{
  *, .gs-fill, .gs-row-f, .stButton > button {{ transition:none !important; animation:none !important; }}
  .stButton > button:hover {{ transform:none; }}
}}
</style>""", unsafe_allow_html=True)


def style_plots():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": MUTED, "axes.labelcolor": MUTED,
        "axes.edgecolor": BORDER, "axes.titlecolor": INK,
        "xtick.color": FAINT, "ytick.color": FAINT,
        "grid.color": BORDER, "axes.grid": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
    })


# --------------------------------------------------------------------------- #
# Model + audio
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="First launch: generating sample sounds and training the model...")
def load_model(path=MODEL_PATH):
    # A fresh copy of the repo (for example on Streamlit Community Cloud) has
    # no sample data or trained model, since both are in .gitignore. Build them
    # once, exactly as the README's setup steps do.
    if not os.path.exists(path):
        import generate_synthetic_data
        import train_model
        if not sample_files():
            generate_synthetic_data.generate()
        train_model.main()
    return joblib.load(path) if os.path.exists(path) else None


def decode_audio(raw_bytes) -> np.ndarray:
    y, _ = librosa.load(io.BytesIO(raw_bytes), sr=SAMPLE_RATE, mono=True)
    return load_audio(y)


def dominant_frequency(y, sr=SAMPLE_RATE):
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    return 0.0 if spec.size <= 1 else float(freqs[np.argmax(spec[1:]) + 1])


def high_freq_energy_ratio(y, sr=SAMPLE_RATE, cutoff=3000.0):
    spec = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    total = spec.sum()
    return 0.0 if total <= 0 else float(spec[freqs >= cutoff].sum() / total)


def diagnose(y, bundle):
    feats = extract_features(y).reshape(1, -1)
    classes = list(bundle["classes"])
    proba = bundle["model"].predict_proba(feats)[0]
    prob_map = dict(zip(classes, proba))
    health = 100.0 * prob_map.get("healthy", 0.0)

    if health >= 75:
        status = "Healthy"
    elif health >= 40:
        status = "Early Warning"
    else:
        status = "Critical Failure"

    return {
        "predicted": classes[int(np.argmax(proba))],
        "prob_map": prob_map,
        "health": health,
        "status": status,
        "color": STATUS[status]["color"],
        "dominant_hz": dominant_frequency(y),
        "hf_ratio": high_freq_energy_ratio(y),
        "confidence": float(max(proba)) * 100,
    }


def plain_reading(d):
    """One sentence a non-expert can act on."""
    if d["status"] == "Healthy":
        return ("This machine sounds normal. Its acoustic signature matches a healthy "
                "motor, with energy concentrated in the expected low-frequency hum.")
    if d["predicted"] == "friction":
        return ("GridSense hears high-frequency grinding, the signature of metal-on-metal "
                "bearing wear. This usually appears well before the bearing seizes.")
    if d["predicted"] == "imbalance":
        return ("GridSense hears a slow low-frequency wobble, the signature of a rotor "
                "spinning off-centre, or loose mounting hardware.")
    return ("The signal is mostly nominal, but not cleanly healthy. Capture another "
            "sample to confirm whether this is a trend or a one-off.")


def recommendations(d):
    """(title, detail) pairs, most important first."""
    hz, hf = d["dominant_hz"], d["hf_ratio"]
    if d["status"] == "Healthy":
        return [("No action required",
                 f"Continue routine monitoring. Baseline dominant frequency is {hz:.0f} Hz, "
                 "which is nominal for a healthy motor."),
                ("Re-check on your normal schedule",
                 "Recording the same machine periodically lets you spot drift early.")]

    recs = []
    if d["predicted"] == "friction":
        recs.append(("Inspect and re-lubricate the bearings",
                     f"{hf*100:.0f}% of the acoustic energy sits above 3 kHz, a grinding "
                     "signature consistent with dry or worn bearings."))
        recs.append(("Check for metal-on-metal contact",
                     "Look for scoring on the shaft and races. Replace the bearing if the "
                     "grinding persists after lubrication."))
    elif d["predicted"] == "imbalance":
        recs.append(("Check rotor balance and alignment",
                     f"A dominant low-frequency component near {hz:.0f} Hz indicates the "
                     "rotating mass is off-centre."))
        recs.append(("Re-torque mounting bolts and couplings",
                     "Loose mounts and misaligned couplings produce the same wobble and are "
                     "the cheapest causes to rule out first."))
    else:
        recs.append(("Capture a second sample",
                     "Confidence is reduced. A follow-up recording will confirm whether this "
                     "is a developing fault or background noise."))

    if d["status"] == "Critical Failure":
        recs.append(("Schedule downtime for inspection",
                     "The fault signature is strong. Taking the unit offline now is far "
                     "cheaper than an unplanned failure."))
    return recs


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_waveform(y, color, sr=SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(7, 2.4))
    t = np.arange(len(y)) / sr
    ax.plot(t, y, linewidth=0.5, color=color, alpha=0.9)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, t[-1] if len(t) else 1)
    fig.tight_layout()
    return fig


def plot_spectrogram(y, sr=SAMPLE_RATE):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz",
                                   ax=ax, cmap="magma")
    ax.set_ylim(0, 8000)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.015)
    cb.outline.set_edgecolor(BORDER)
    cb.ax.yaxis.set_tick_params(color=FAINT)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def masthead():
    st.markdown(f"""<div class="gs-head">
  <div>
    <div class="gs-word">Grid<span>Sense</span></div>
    <div class="gs-tag">Hear failure before it happens: acoustic predictive maintenance.</div>
  </div>
  <div class="gs-offline">◆ Offline · nothing leaves this device</div>
</div>""", unsafe_allow_html=True)


def sample_files():
    out = {}
    for cls in ("healthy", "friction", "imbalance"):
        found = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.wav")))
        if found:
            out[cls] = found
    return out


def empty_state():
    st.markdown(
        '<div class="gs-h" style="font-size:1.35rem;margin-top:.5rem">'
        'Point a microphone at a machine. Find out if it is failing.</div>'
        '<div class="gs-sub">GridSense listens to a few seconds of sound, breaks it into its '
        'frequency components, and recognises the acoustic fingerprints of bearing friction '
        'and mechanical imbalance, faults that are audible long before they are visible.</div>',
        unsafe_allow_html=True)

    st.markdown("""<div class="gs-steps">
  <div><div class="gs-step-n">Step 1</div><div class="gs-step-t">Capture</div>
       <div class="gs-step-d">Record the machine live, or load a sound file.</div></div>
  <div><div class="gs-step-n">Step 2</div><div class="gs-step-t">Analyse</div>
       <div class="gs-step-d">An FFT extracts the frequency fingerprint; a trained model reads it.</div></div>
  <div><div class="gs-step-n">Step 3</div><div class="gs-step-t">Act</div>
       <div class="gs-step-d">Get a health score, a verdict, and what to do about it.</div></div>
</div>""", unsafe_allow_html=True)

    samples = sample_files()
    if samples:
        st.markdown('<div class="gs-h">Try it now, no recording needed</div>'
                    '<div class="gs-sub">Load a real sample from the generated dataset. '
                    'Start with a failing one to see the full diagnosis.</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(samples))
        nice = {"healthy": ("Healthy motor", HEALTHY),
                "friction": ("Bearing friction", CRITICAL),
                "imbalance": ("Mechanical imbalance", WARNING)}
        for col, (cls, files) in zip(cols, samples.items()):
            label, dot = nice.get(cls, (cls.title(), MUTED))
            with col:
                if st.button(f"●  {label}", key=f"s_{cls}", width="stretch"):
                    st.session_state["sample"] = random.choice(files)
                    st.rerun()
    else:
        st.info("No sample audio found. Run `python generate_synthetic_data.py` first.")


def results(y, d, source_label):
    s = STATUS[d["status"]]
    color = d["color"]
    health = d["health"]

    # Verdict
    st.markdown(f"""<div class="gs-verdict">
  <div class="gs-verdict-top">
    <span style="color:{color};font-size:1.1rem;line-height:1">●</span>
    <span class="gs-status" style="color:{color}">{s['label']}</span>
  </div>
  <div class="gs-read">{plain_reading(d)}</div>
  <div class="gs-meter-head">
    <span class="gs-meter-label">Machine Health Index</span>
    <span class="gs-meter-val" style="color:{color}">{health:.0f}<span style="font-size:.9rem;color:{MUTED}">/100</span></span>
  </div>
  <div class="gs-track"><div class="gs-fill" style="width:{max(health,1.5):.1f}%;background:{color}"></div></div>
  <div class="gs-ticks">
    <span class="gs-tick" style="left:0%;transform:none">0</span>
    <span class="gs-tick" style="left:40%">40 · warning</span>
    <span class="gs-tick" style="left:75%">75 · healthy</span>
    <span class="gs-tick" style="left:100%;transform:translateX(-100%)">100</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Measurements
    st.markdown(f"""<div class="gs-strip">
  <div class="gs-cell"><div class="gs-cell-k">Detected condition</div>
    <div class="gs-cell-v" style="color:{color}">{d['predicted'].title()}</div></div>
  <div class="gs-cell"><div class="gs-cell-k">Dominant frequency</div>
    <div class="gs-cell-v">{d['dominant_hz']:.0f} <small>Hz</small></div></div>
  <div class="gs-cell"><div class="gs-cell-k">Energy above 3 kHz</div>
    <div class="gs-cell-v">{d['hf_ratio']*100:.0f} <small>%</small></div></div>
  <div class="gs-cell"><div class="gs-cell-k">Model confidence</div>
    <div class="gs-cell-v">{d['confidence']:.0f} <small>%</small></div></div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown('<div class="gs-h">Signal analysis</div>'
                    '<div class="gs-sub">The waveform shows loudness over time. The spectrogram '
                    'shows which frequencies carry the energy: bright bands high up mean '
                    'grinding; a slow pulse means wobble.</div>', unsafe_allow_html=True)
        st.pyplot(plot_waveform(y, color), width="stretch")
        st.pyplot(plot_spectrogram(y), width="stretch")

    with right:
        st.markdown('<div class="gs-h">Confidence breakdown</div>'
                    '<div class="gs-sub">How strongly the sound matches each known condition.</div>',
                    unsafe_allow_html=True)
        order = sorted(d["prob_map"].items(), key=lambda kv: -kv[1])
        cmap = {"healthy": HEALTHY, "friction": CRITICAL, "imbalance": WARNING}
        rows = "".join(
            f'<div class="gs-row"><div class="gs-row-k">{k.title()}</div>'
            f'<div class="gs-row-t"><div class="gs-row-f" style="width:{v*100:.1f}%;'
            f'background:{cmap.get(k, MUTED)}"></div></div>'
            f'<div class="gs-row-v">{v*100:.0f}%</div></div>'
            for k, v in order)
        st.markdown(f'<div class="gs-conf">{rows}</div>', unsafe_allow_html=True)

        st.markdown('<div class="gs-h">Listen</div>', unsafe_allow_html=True)
        st.audio(st.session_state.get("audio_bytes"), format="audio/wav")
        st.caption(f"Source: {source_label}")

    st.markdown('<div class="gs-h">Recommended actions</div>'
                '<div class="gs-sub">Ordered by priority.</div>', unsafe_allow_html=True)
    for title, detail in recommendations(d):
        st.markdown(f'<div class="gs-rec"><div class="gs-rec-t">{title}</div>'
                    f'<div class="gs-rec-d">{detail}</div></div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Analyse another sample", key="reset"):
        for k in ("sample", "audio_bytes", "source_label"):
            st.session_state.pop(k, None)
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
inject_css()
style_plots()
masthead()

bundle = load_model()
if bundle is None:
    st.error("No trained model found. Run `python generate_synthetic_data.py` "
             "then `python train_model.py`, and reload this page.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="gs-h" style="margin-top:0">Audio input</div>', unsafe_allow_html=True)
    rec = st.audio_input("Record a machine")
    up = st.file_uploader("or upload a .wav file", type=["wav"])

    st.markdown('<div class="gs-h">Load a sample</div>', unsafe_allow_html=True)
    for cls, files in sample_files().items():
        if st.button(cls.title(), key=f"sb_{cls}", width="stretch"):
            st.session_state["sample"] = random.choice(files)
            st.rerun()

    st.markdown(f'<div class="gs-sub" style="margin-top:1.5rem;font-size:.8rem;color:{FAINT}">'
                'Detects bearing friction and mechanical imbalance from sound alone. '
                'Runs entirely on this machine.</div>', unsafe_allow_html=True)

# Precedence: a fresh recording or upload beats a previously loaded sample.
audio_bytes, source_label = None, None
if rec is not None:
    audio_bytes, source_label = rec.read(), "Live microphone recording"
    st.session_state.pop("sample", None)
elif up is not None:
    audio_bytes, source_label = up.read(), f"Uploaded file: {up.name}"
    st.session_state.pop("sample", None)
elif st.session_state.get("sample"):
    path = st.session_state["sample"]
    with open(path, "rb") as fh:
        audio_bytes = fh.read()
    source_label = f"Sample: {os.path.basename(path)}"

if audio_bytes is None:
    empty_state()
    st.stop()

st.session_state["audio_bytes"] = audio_bytes
y = decode_audio(audio_bytes)

if y.size < SAMPLE_RATE // 2:
    st.warning("That clip is under half a second, so the diagnosis may be unreliable. "
               "Aim for 2-3 seconds of steady running noise.")

results(y, diagnose(y, bundle), source_label)
