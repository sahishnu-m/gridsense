# GridSense — Devpost Submission Copy

Copy-paste ready text for the **Global Tech Innovation Challenge** submission.
Judging criterion: **Overall Innovation** (creativity, execution, impact) · peer-voted.

---

## Tagline (one line)
> Hear failure before it happens — acoustic predictive maintenance from any microphone, 100% offline.

---

## Short summary — the real-world problem (required field)

Industrial machines fail loudly long before they fail catastrophically: a
bearing whines, a rotor wobbles, a fan grinds. Large factories catch this with
vibration-sensor networks and cloud analytics costing **thousands of dollars per
machine** — pricing out small workshops, clinics, schools, farms, and the
billions of small motors and micro-hardware devices that simply run until they
break.

**GridSense turns any microphone into a trained diagnostic ear.** It records a
few seconds of machine sound, runs Fast Fourier Transform signal processing to
extract the acoustic fingerprint, and uses a machine-learning classifier to
detect early **bearing friction** and **mechanical imbalance** — outputting a
0–100% Machine Health Index and specific maintenance actions. It runs entirely
on a laptop with no internet, no subscriptions, and no data ever leaving the
device.

---

## What it does

- **Listens:** live microphone recording or `.wav` upload.
- **Analyses:** FFT / spectrogram + feature extraction (MFCCs, spectral
  centroid, spectral rolloff, zero-crossing rate, RMS energy).
- **Diagnoses:** a RandomForest classifier labels the sound `healthy`,
  `friction`, or `imbalance` with **98.3% held-out accuracy**.
- **Advises:** a Machine Health Index (0–100%), a 🟢/🟡/🔴 status badge, and
  concrete recommendations (e.g. "re-lubricate bearings", "check rotor balance").

## How we built it

Python 3 · Librosa & SciPy (signal processing) · scikit-learn (RandomForest) ·
Streamlit (dashboard) · Matplotlib (waveform + spectrogram). A physics-based
synthetic-audio generator lets the entire system be trained and demoed with
**zero physical hardware** — motor hum, high-frequency grinding, and
low-frequency wobble are synthesised with controllable severity and realistic
background noise.

## What makes it innovative

- **Sensor-free:** no accelerometers or proprietary hardware — just a mic.
- **Fully offline & private:** audio never leaves the device.
- **Accessible:** democratises a capability normally reserved for large industry.
- **Honest engineering:** its single misclassification is a *mild* friction case
  read as healthy — surfacing the genuine hard problem of early fault detection.

## Impact & sustainability

Catching faults early extends equipment life, prevents catastrophic breakdowns,
and reduces electronic waste. Because it runs on commodity hardware with no
recurring cost, it's deployable in resource-constrained settings worldwide —
factory motors, HVAC, medical devices, farm pumps, and micro-hardware alike.

## What's next

Field recordings from real machinery to validate accuracy beyond synthetic data;
more fault classes (electrical arcing, cavitation, gear-mesh wear); a trend view
that tracks a machine's Health Index over time; and a Raspberry Pi edge build.

---

## Submission checklist

- [ ] **Public code repo** — push this folder to GitHub (see run instructions).
- [ ] **Demo** — 1–3 min video (see `DEMO_SCRIPT.md`) OR the detailed text above.
- [ ] **Problem summary** — paste the "Short summary" section above.
