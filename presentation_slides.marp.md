---
marp: true
theme: default
paginate: true
backgroundColor: #0f172a
color: #e2e8f0
style: |
  section { font-family: 'Segoe UI', sans-serif; }
  h1 { color: #38bdf8; }
  h2 { color: #7dd3fc; }
  strong { color: #fbbf24; }
  a { color: #38bdf8; }
---

<!-- _class: lead -->

# 🎧 GridSense

## Hear failure before it happens.

**Acoustic predictive maintenance** powered by FFT signal processing and
machine learning — detecting early bearing friction and mechanical imbalance
from nothing but a **microphone**.

*100% local · offline · privacy-first*

---

# The Problem

**Industrial predictive maintenance is expensive and exclusive.**

- Vibration sensors, SCADA systems and cloud analytics cost **thousands per machine**.
- Small workshops, clinics, schools and micro-hardware get **nothing** — they run to failure.
- Failures are **audible long before they're catastrophic**: a whine, a grind, a wobble.

> Every technician already diagnoses by ear. GridSense turns *any* microphone
> into that trained ear — for **industrial motors down to micro-hardware fans**.

---

# Technical Architecture & Signal Pipeline

```
 🎙️ Audio  ──►  FFT / STFT  ──►  Feature Extraction  ──►  RandomForest  ──►  Health Index
  (mic/wav)      (spectrum)     MFCC · Centroid ·          Classifier        0–100% + badge
                                Rolloff · ZCR · RMS
```

- **Signal processing:** Librosa / SciPy — time-domain + frequency spectrogram.
- **ML model:** Scikit-learn RandomForest → `healthy` / `friction` / `imbalance`.
- **Dashboard:** Streamlit — live mic or `.wav`, plots, and actionable advice.
- **Data:** synthetic generator = train & demo with **zero physical hardware**.

> **98.3%** classification accuracy on held-out data. The one miss? A *mild*
> friction case read as healthy — the exact early-detection challenge that
> makes this problem worth solving.

---

# Impact & Sustainability

- ♻️ **Extends equipment life** — catch faults early, avoid catastrophic breakdowns and e-waste.
- 🌍 **Democratises maintenance** — no cloud, no subscriptions, runs on a laptop.
- 🔒 **Privacy-first & offline** — audio never leaves the device.
- 🏥 **Broadly applicable** — factory motors, HVAC, medical devices, farm pumps, micro-hardware.

## GridSense: predictive maintenance for **everyone**, not just the Fortune 500.
