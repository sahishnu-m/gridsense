# GridSense
**Hear failure before it happens.**
Acoustic predictive maintenance from any microphone, 100% offline.

---

## The problem
Machines fail loudly long before they fail catastrophically: a bearing
whines, a rotor wobbles, a fan grinds. Catching that early normally means
vibration sensors and cloud analytics costing **thousands of dollars per
machine**, out of reach for small workshops, clinics, schools, farms, and
the billions of small motors and micro-hardware devices that just run until
they break.

## The solution
GridSense turns any microphone into a trained diagnostic ear. It records a
few seconds of machine sound, runs an FFT to extract the acoustic
fingerprint, and uses a machine-learning classifier to detect **bearing
friction** and **mechanical imbalance** early, well before either one causes
a breakdown.

| Step | What happens |
|---|---|
| Capture | Live mic recording or `.wav` upload. 2-3 seconds is enough |
| Analyze | FFT / spectrogram to MFCC, spectral centroid, rolloff, ZCR, RMS |
| Diagnose | RandomForest classifier to healthy / friction / imbalance |
| Act | 0-100 Machine Health Index, a status badge, and a concrete next step |

## Results
**98.3%** held-out classification accuracy. Its one miss in testing was a
*mild* friction case read as healthy: the genuine hard problem in this
space, catching a fault in its earliest, quietest stage. We're being
upfront about that limitation rather than hiding it.

## Why it's different
- **Sensor-free.** A laptop or phone microphone, no accelerometers or proprietary hardware.
- **Fully offline and private.** Every computation runs on-device; audio never leaves it.
- **Accessible by design.** Democratizes a capability normally reserved for large industry.
- **Honest engineering.** Transparent about the model's real limitations.

## Impact
Catching faults early extends equipment life, prevents catastrophic
breakdowns, and reduces electronic waste. Zero recurring cost means it's
deployable anywhere: factory motors, HVAC, medical devices, farm pumps, and
micro-hardware, in resource-constrained settings worldwide.

## What's next
Field recordings from real machinery to validate accuracy beyond the
synthetic dataset · more fault classes (electrical arcing, cavitation,
gear-mesh wear) · a Health Index trend view over time · a Raspberry Pi edge
build.

## Tech stack
Python 3 · Librosa & SciPy (signal processing) · scikit-learn (RandomForest)
· Streamlit (dashboard) · Matplotlib (waveform + spectrogram). A
physics-based synthetic-audio generator trains and demos the entire system
with zero physical hardware.

---

**Try it:** `python run.py` for the live dashboard, or open `demo.html` for
a no-install interactive walkthrough. **Full deck:** `pitch_deck.html`.
