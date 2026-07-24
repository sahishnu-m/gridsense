"""
generate_synthetic_data.py — mathematically synthesise a local audio dataset.

Creates .wav clips for three machine conditions so the whole GridSense
pipeline can be trained and demoed WITHOUT any physical hardware:

  1. healthy   -> clean motor hum: fundamental + harmonics, tiny noise floor
  2. friction  -> healthy hum + high-frequency metallic grinding / hiss
  3. imbalance -> healthy hum + slow low-frequency amplitude wobble (beat)

Everything is generated with numpy and written with scipy.io.wavfile.
No downloads, no APIs — 100% offline.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --per-class 40 --seed 7
"""

import os
import argparse
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050          # must match features.SAMPLE_RATE
DURATION = 3.0               # seconds per clip
CLASSES = ("healthy", "friction", "imbalance")
OUTPUT_DIR = "sample_data"


def _time_axis(sr=SAMPLE_RATE, duration=DURATION):
    n = int(sr * duration)
    return np.linspace(0.0, duration, n, endpoint=False)


def _motor_hum(t, rng, fundamental=60.0):
    """Base motor tone: fundamental + a few decaying harmonics."""
    signal = np.zeros_like(t)
    # Slight per-sample jitter in fundamental to feel real, not perfectly pure.
    f0 = fundamental + rng.uniform(-2.0, 2.0)
    for k, amp in enumerate([1.0, 0.5, 0.25, 0.12], start=1):
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * k * t + phase)
    return signal


def _background_noise(t, rng):
    """Realistic mixed environment noise: white + low-frequency 'pink-ish'
    rumble. Every real recording has a shifting noise floor."""
    white = rng.standard_normal(t.shape)
    # crude coloured noise: low-pass a random walk for room/machine rumble
    rumble = np.cumsum(rng.standard_normal(t.shape))
    rumble = rumble / (np.max(np.abs(rumble)) + 1e-9)
    level = rng.uniform(0.02, 0.12)          # varying ambient level per clip
    return level * white + 0.5 * level * rumble


def make_healthy(t, rng):
    """Clean hum with a low, but variable, broadband noise floor.

    Real 'healthy' machines aren't silent — some run louder, and a few even
    have a faint high-frequency tick. This overlap is what stops the model
    from being trivially perfect."""
    sig = _motor_hum(t, rng)

    # Occasionally a healthy unit has a faint hint of high-freq content,
    # blurring the boundary with early friction.
    if rng.random() < 0.3:
        f = rng.uniform(3000, 6000)
        sig += rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * f * t)

    sig += _background_noise(t, rng)
    return sig


def make_friction(t, rng):
    """Healthy hum PLUS high-frequency grinding, at a RANDOM SEVERITY.

    Mild cases (low severity) look almost healthy; severe cases are obvious.
    This spread is what produces realistic, sub-100% accuracy."""
    sig = _motor_hum(t, rng)
    severity = rng.uniform(0.25, 1.0)        # 0.25 = subtle, 1.0 = severe

    # High-frequency hiss: white noise emphasised by squaring high sine carrier.
    hiss = rng.standard_normal(t.shape)
    carrier = np.sin(2 * np.pi * rng.uniform(3000, 5000) * t)
    sig += severity * 0.35 * hiss * (0.5 + 0.5 * np.abs(carrier))

    # A few sharp metallic squeal partials that come and go.
    for _ in range(rng.integers(1, 4)):
        f = rng.uniform(3500, 7000)
        env = np.clip(np.sin(2 * np.pi * rng.uniform(1.5, 4.0) * t), 0, None)
        sig += severity * 0.15 * env * np.sin(2 * np.pi * f * t)

    sig += _background_noise(t, rng)
    return sig


def make_imbalance(t, rng):
    """Healthy hum with a slow low-frequency amplitude wobble, RANDOM SEVERITY.

    Wobble depth and sub-harmonic strength vary per clip so mild imbalance
    overlaps with healthy."""
    sig = _motor_hum(t, rng)
    severity = rng.uniform(0.25, 1.0)

    # Low-frequency amplitude modulation (the "wobble"), 1.5–4 Hz.
    wobble_hz = rng.uniform(1.5, 4.0)
    am = 1.0 + severity * 0.6 * np.sin(2 * np.pi * wobble_hz * t)
    sig *= am

    # Low sub-harmonic thump characteristic of imbalance, scaled by severity.
    sig += severity * 0.3 * np.sin(2 * np.pi * rng.uniform(20, 35) * t)
    sig += _background_noise(t, rng)
    return sig


GENERATORS = {
    "healthy": make_healthy,
    "friction": make_friction,
    "imbalance": make_imbalance,
}


def _to_int16(sig):
    """Normalise to [-1, 1] then convert to 16-bit PCM for a valid .wav."""
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak
    sig = 0.9 * sig                      # small headroom to avoid clipping
    return np.int16(sig * 32767)


def generate(per_class=40, seed=42, output_dir=OUTPUT_DIR):
    rng = np.random.default_rng(seed)
    t = _time_axis()
    os.makedirs(output_dir, exist_ok=True)

    total = 0
    for label in CLASSES:
        class_dir = os.path.join(output_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(per_class):
            sig = GENERATORS[label](t, rng)
            audio = _to_int16(sig)
            fname = os.path.join(class_dir, f"{label}_{i:03d}.wav")
            wavfile.write(fname, SAMPLE_RATE, audio)
            total += 1
        print(f"  [{label:9s}] wrote {per_class} clips -> {class_dir}")

    print(f"\nDone. {total} .wav files generated under '{output_dir}/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic GridSense audio dataset.")
    parser.add_argument("--per-class", type=int, default=40,
                        help="number of clips per class (default: 40)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="output folder")
    args = parser.parse_args()

    print("GridSense — synthetic dataset generator")
    print(f"Classes: {', '.join(CLASSES)}  |  {args.per_class} clips each\n")
    generate(args.per_class, args.seed, args.output_dir)
