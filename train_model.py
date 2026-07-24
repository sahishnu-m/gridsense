"""
train_model.py — train the GridSense acoustic fault classifier.

Pipeline:
  1. Walk sample_data/<class>/*.wav
  2. Extract features (MFCC + spectral centroid + spectral rolloff + extras)
     via features.py  (shared with app.py to avoid train/inference drift)
  3. Train a RandomForestClassifier
  4. Report accuracy on a held-out split
  5. Save the fitted model to gridsense_model.pkl with joblib

Runs 100% locally. No network access required.

Usage:
    python train_model.py
"""

import os
import glob
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from features import load_audio, extract_features, FEATURE_NAMES

DATA_DIR = "sample_data"
MODEL_PATH = "gridsense_model.pkl"


def build_dataset(data_dir=DATA_DIR):
    """Load every .wav under data_dir/<label>/ and return (X, y, labels)."""
    X, y = [], []
    labels = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    if not labels:
        raise SystemExit(
            f"No class folders found in '{data_dir}'. "
            "Run:  python generate_synthetic_data.py"
        )

    for label in labels:
        files = sorted(glob.glob(os.path.join(data_dir, label, "*.wav")))
        if not files:
            print(f"  [warn] no .wav files for class '{label}'")
        for path in files:
            signal = load_audio(path)
            X.append(extract_features(signal))
            y.append(label)
        print(f"  [{label:9s}] {len(files)} clips")

    return np.array(X), np.array(y), labels


def main():
    print("GridSense — model training\n")
    print("Extracting features from local dataset...")
    X, y, labels = build_dataset()
    print(f"\nDataset: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Classes: {', '.join(labels)}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    print("Training RandomForestClassifier...")
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"\nHeld-out accuracy: {acc * 100:.1f}%\n")
    print("Classification report:")
    print(classification_report(y_test, clf.predict(X_test)))
    print("Confusion matrix (rows=true, cols=pred):")
    print("labels:", list(clf.classes_))
    print(confusion_matrix(y_test, clf.predict(X_test)))

    # Persist the model AND metadata needed by the dashboard.
    bundle = {
        "model": clf,
        "classes": list(clf.classes_),
        "feature_names": list(FEATURE_NAMES),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
