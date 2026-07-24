"""
features.py: single source of truth for audio feature extraction.

Both train_model.py and app.py import from here so that the features used
at training time EXACTLY match the features used at inference time.

All processing is 100% local: librosa / numpy only. No network, no APIs.
"""

import numpy as np
import librosa

# ---- Global audio config (keep identical everywhere) ----------------------
SAMPLE_RATE = 22050          # Hz, librosa default, good balance for machinery
N_MFCC = 20                  # number of MFCC coefficients
DURATION = 3.0               # seconds each clip is normalised to

# Human-readable names for every column produced by extract_features().
FEATURE_NAMES = (
    [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    + [f"mfcc_std_{i}" for i in range(N_MFCC)]
    + ["spectral_centroid_mean", "spectral_centroid_std",
       "spectral_rolloff_mean", "spectral_rolloff_std",
       "zero_crossing_rate_mean", "rms_mean"]
)


def load_audio(path_or_array, sr=SAMPLE_RATE):
    """Load a .wav file (path) OR accept a raw numpy array already in memory.

    Returns a mono float32 signal resampled to `sr`.
    """
    if isinstance(path_or_array, (str, bytes)) or hasattr(path_or_array, "read"):
        y, _ = librosa.load(path_or_array, sr=sr, mono=True)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)
    # Normalise amplitude to [-1, 1] to be robust to recording gain.
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def extract_features(y, sr=SAMPLE_RATE):
    """Turn a 1-D audio signal into a fixed-length feature vector.

    Features:
      - MFCCs (mean + std of each coefficient)  -> timbre / spectral shape
      - Spectral Centroid (mean + std)          -> "brightness" of the sound
      - Spectral Rolloff (mean + std)           -> high-frequency energy edge
      - Zero-Crossing Rate (mean)               -> noisiness / friction hint
      - RMS energy (mean)                        -> overall loudness

    Returns a 1-D numpy array aligned with FEATURE_NAMES.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    vector = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [centroid.mean(), centroid.std()],
        [rolloff.mean(), rolloff.std()],
        [zcr.mean()],
        [rms.mean()],
    ]).astype(np.float32)

    return vector
