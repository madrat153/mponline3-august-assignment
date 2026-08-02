"""
cv_service.py -- Module A2 (image classification) + A3 (face recognition).

1. ProductClassifierService: HOG features + sklearn SVM (default,
   dependency-light path). Auto-detects an optional Keras/MobileNetV2
   model bundle if you trained one with training/train_product_classifier_tf.py.

2. FaceRecognitionService: OpenCV LBPH face recognizer for returning-
   customer detection. Haar-cascade detection -> LBPH encoding -> compare
   against enrolled customer templates -> log a visit.

   ETHICS NOTE: Facial recognition on real customers raises consent,
   data-privacy, and bias concerns. This demo only ever operates on
   images the caller explicitly uploads/enrolls (no covert capture).
   Any real deployment needs explicit opt-in consent, a documented
   retention/deletion policy, and bias/accuracy audits across
   demographic groups.
"""
import json
import pickle
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import cv2
import joblib
import numpy as np

from app import config
from app.services import cv_utils


class ProductClassifierService:
    def __init__(self, model_path=config.PRODUCT_CLASSIFIER_PATH):
        self.model_path = model_path
        self.kind = None
        self.pipeline = None
        self.keras_model = None
        self.class_names = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            self.pipeline = None
            return
        bundle = joblib.load(self.model_path)
        self.kind = bundle.get("kind", "sklearn")
        self.class_names = bundle["class_names"]
        if self.kind == "sklearn":
            self.pipeline = bundle["pipeline"]
        elif self.kind == "keras":
            from tensorflow import keras
            self.keras_model = keras.models.load_model(bundle["keras_path"])

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None or self.keras_model is not None

    def predict(self, image_bgr: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_ready:
            raise RuntimeError(
                "Product classifier model not found. Run "
                "training/train_product_classifier.py first."
            )
        if self.kind == "sklearn":
            feats = _hog_features(image_bgr).reshape(1, -1)
            probs = self.pipeline.predict_proba(feats)[0]
        else:
            resized = cv_utils.resize_image(image_bgr, 224, 224)
            arr = np.expand_dims(resized.astype("float32") / 255.0, axis=0)
            probs = self.keras_model.predict(arr, verbose=0)[0]

        scores = {cls: float(p) for cls, p in zip(self.class_names, probs)}
        best_idx = int(np.argmax(probs))
        return self.class_names[best_idx], float(probs[best_idx]), scores


def _hog_features(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv_utils.to_grayscale(image_bgr)
    gray = cv_utils.resize_image(gray, 128, 128)
    hog = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(32, 32),
                             _blockStride=(16, 16), _cellSize=(16, 16), _nbins=9)
    return hog.compute(gray).flatten()


class FaceRecognitionService:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels: Dict[int, str] = {}
        self._loaded = False
        self._load()

    def _load(self):
        if config.FACE_RECOGNIZER_PATH.exists() and config.FACE_LABELS_PATH.exists():
            self.recognizer.read(str(config.FACE_RECOGNIZER_PATH))
            with open(config.FACE_LABELS_PATH, "rb") as f:
                self.labels = pickle.load(f)
            self._loaded = True

    @property
    def is_ready(self) -> bool:
        return self._loaded and len(self.labels) > 0

    def _log_visit(self, customer_id: Optional[str], status: str) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        record = {"customer_id": customer_id, "status": status, "timestamp": ts}
        _append_json_log(config.VISITS_LOG, record)
        return ts

    def recognize(self, image_bgr: np.ndarray):
        boxes = cv_utils.detect_faces(image_bgr)
        if not boxes:
            return {"matched": False, "customer_id": None, "status": "no_face_detected",
                     "confidence": None, "faces_detected": 0, "visit_timestamp": None}

        box = max(boxes, key=lambda b: b[2] * b[3])
        face = cv_utils.crop_face(image_bgr, box)
        face_gray = cv_utils.to_grayscale(face)
        face_gray = cv_utils.resize_image(face_gray, 200, 200)

        if not self.is_ready:
            ts = self._log_visit(None, "new_customer")
            return {"matched": False, "customer_id": None, "status": "new_customer",
                     "confidence": None, "faces_detected": len(boxes), "visit_timestamp": ts}

        label, distance = self.recognizer.predict(face_gray)
        if distance <= config.FACE_MATCH_THRESHOLD and label in self.labels:
            customer_id = self.labels[label]
            ts = self._log_visit(customer_id, "returning_customer")
            confidence = max(0.0, 1.0 - (distance / 100.0))
            return {"matched": True, "customer_id": customer_id, "status": "returning_customer",
                     "confidence": round(confidence, 3), "faces_detected": len(boxes), "visit_timestamp": ts}

        ts = self._log_visit(None, "new_customer")
        confidence = max(0.0, 1.0 - (distance / 100.0))
        return {"matched": False, "customer_id": None, "status": "new_customer",
                 "confidence": round(confidence, 3), "faces_detected": len(boxes), "visit_timestamp": ts}

    def register_face(self, image_bgr: np.ndarray, customer_id: Optional[str] = None) -> str:
        boxes = cv_utils.detect_faces(image_bgr)
        if not boxes:
            raise ValueError("No face detected in the provided image.")
        box = max(boxes, key=lambda b: b[2] * b[3])
        face = cv_utils.crop_face(image_bgr, box)
        face_gray = cv_utils.resize_image(cv_utils.to_grayscale(face), 200, 200)

        customer_id = customer_id or f"cust_{uuid.uuid4().hex[:8]}"
        next_label = (max(self.labels.keys()) + 1) if self.labels else 0

        if self.is_ready:
            self.recognizer.update([face_gray], np.array([next_label]))
        else:
            self.recognizer.train([face_gray], np.array([next_label]))

        self.labels[next_label] = customer_id
        self.recognizer.write(str(config.FACE_RECOGNIZER_PATH))
        with open(config.FACE_LABELS_PATH, "wb") as f:
            pickle.dump(self.labels, f)
        self._loaded = True
        return customer_id


def _append_json_log(path, record: dict):
    data = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
    data.append(record)
    path.write_text(json.dumps(data, indent=2))
