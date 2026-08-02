"""
train_face_db.py -- Module A3 training script.

Builds the initial LBPH face-recognition database used for returning-
customer detection.

DATASET NOTE: we cannot download a real face dataset (e.g. LFW) or accept
webcam captures inside this offline build environment, and Haar-cascade
face detection will not fire on procedurally drawn shapes. So this script
seeds the recognizer directly with structured synthetic grayscale
templates for a few demo customer IDs, enough to demonstrate the
enrollment -> recognition -> "returning customer" logging pipeline
end-to-end via the API.

FOR A REAL SUBMISSION: don't use this seed data as your evaluation
dataset -- instead collect a few consenting sample photos (or use the
LFW subset for practice) and call POST /register-face for each one.

ETHICS: see the note at the top of app/services/cv_service.py regarding
consent, data privacy, and bias considerations before deploying facial
recognition against real customers.
"""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config

DEMO_CUSTOMERS = ["cust_alice01", "cust_bob02", "cust_carol03"]
SAMPLES_PER_CUSTOMER = 15
FACE_SIZE = 200


def _synthetic_face_template(seed: int, rng: np.random.Generator) -> np.ndarray:
    base = np.random.RandomState(seed).randint(80, 180, (FACE_SIZE, FACE_SIZE), dtype=np.uint8)
    smoothed = cv2.GaussianBlur(base, (15, 15), 0)
    noise = rng.normal(0, 4, smoothed.shape)
    return np.clip(smoothed.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(7)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, labels, label_map = [], [], {}
    for label_idx, customer_id in enumerate(DEMO_CUSTOMERS):
        label_map[label_idx] = customer_id
        for _ in range(SAMPLES_PER_CUSTOMER):
            faces.append(_synthetic_face_template(seed=label_idx * 1000, rng=rng))
            labels.append(label_idx)

    recognizer.train(faces, np.array(labels))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(config.FACE_RECOGNIZER_PATH))
    with open(config.FACE_LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"Seeded demo face DB with {len(DEMO_CUSTOMERS)} customers -> {config.FACE_RECOGNIZER_PATH}")
    print("Enroll real customers via POST /register-face for an actual evaluation.")


if __name__ == "__main__":
    main()
