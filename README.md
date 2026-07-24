# 🎧 GridSense

**Acoustic predictive maintenance** — detect early micro-hardware and motor
failures (bearing friction, mechanical imbalance) from a simple audio feed
using **FFT signal processing + machine learning**.

Runs **100% locally and offline**. No cloud, no API keys, no third-party SaaS.

---

## 📁 Project Structure

```
GridSense/
├── requirements.txt              # local Python dependencies
├── features.py                   # shared feature extraction (train == inference)
├── generate_synthetic_data.py    # synthesise the .wav dataset locally
├── train_model.py                # extract features + train RandomForest
├── app.py                        # Streamlit dashboard (mic / upload + diagnostics)
├── presentation_slides.marp.md   # 4-slide Marp deck
├── README.md                     # this file
├── sample_data/                  # generated audio (created by the generator)
│   ├── healthy/*.wav
│   ├── friction/*.wav
│   └── imbalance/*.wav
└── gridsense_model.pkl           # trained model (created by train_model.py)
```

---

## 🧰 Tech Stack

Python 3.10+ · Streamlit · Librosa / SciPy · Scikit-learn · Matplotlib · NumPy

---

## 🚀 Setup & Run (step by step)

### 1. Create a virtual environment & install dependencies

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate the synthetic audio dataset
```bash
python generate_synthetic_data.py
```
Creates `sample_data/{healthy,friction,imbalance}/*.wav` (40 clips per class by default).
Customise with `--per-class 60 --seed 7`.

### 3. Train the model
```bash
python train_model.py
```
Extracts features, trains a RandomForest, prints accuracy, and saves
`gridsense_model.pkl`.

### 4. Launch the dashboard
```bash
python run.py
```
Opens in your browser. Record live audio or upload a `.wav`, then read the
Health Index, status badge, spectrogram, and maintenance recommendations.

> **Use `python run.py`, not `streamlit run app.py`.** On Windows + Python 3.14,
> the plain Streamlit command can freeze for a long time on a slow WMI system
> call at startup, and it also prompts for your email on first run. `run.py`
> sidesteps both and boots in about a second.

---

## 🏠 Interactive demo (`demo.html`)

A self-contained walkthrough — **just double-click `demo.html`**, no server, no install.

A landing page scrolls into a top-down floor plan of a house with seven running
appliances. Click any one to hear it and see the diagnosis. Every sound is
synthesised live in the browser with the Web Audio API, and the verdict is
*computed* from that audio by an in-page FFT — not hardcoded. The decision
thresholds were calibrated against the same 240-clip dataset that trains the
scikit-learn model.

Two appliances are failing, three show early warnings, two are healthy.

## 🖥️ Presentation

Preview `presentation_slides.marp.md` with the **Marp for VS Code** extension,
or export locally with the Marp CLI:

```bash
npx @marp-team/marp-cli presentation_slides.marp.md -o slides.html
```

---

## 🔊 The Three Conditions

| Class       | Acoustic signature                          | Likely cause                     |
|-------------|---------------------------------------------|----------------------------------|
| `healthy`   | Clean motor hum + harmonics, low noise      | Normal operation                 |
| `friction`  | High-frequency grinding / metallic squeal   | Bearing wear, poor lubrication   |
| `imbalance` | Low-frequency amplitude wobble + sub-harmonic | Rotor imbalance, misalignment  |

---

## 🔒 Privacy

All processing happens on your machine. Captured audio never leaves the device.
