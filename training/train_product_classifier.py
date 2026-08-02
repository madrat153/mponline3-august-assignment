"""
train_product_classifier.py -- Module A2 training script (default path).

Trains HOG features + a linear SVM: fast, CPU-only, no dataset download
needed. See training/train_product_classifier_tf.py for the MobileNetV2
stretch-goal upgrade once you have a real labeled dataset.

DATASET NOTE: generates a small SYNTHETIC placeholder dataset (colored
geometric shapes standing in for 5 product categories) so the pipeline
is runnable end-to-end offline. For a real submission, use Fashion-MNIST,
Kaggle's "Retail Product Checkout Dataset", or your own scraped photos.
"""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.services.cv_service import _hog_features

CLASS_NAMES = ["shoes", "bags", "electronics", "clothing", "groceries"]
IMG_SIZE = 128


def _synthetic_image(category: str, rng: np.random.Generator) -> np.ndarray:
    img = np.full((IMG_SIZE, IMG_SIZE, 3), 245, dtype=np.uint8)
    color = tuple(int(c) for c in rng.integers(30, 220, size=3))
    jitter = lambda v: v + int(rng.integers(-8, 8))

    if category == "shoes":
        cv2.ellipse(img, (64, 80), (jitter(45), jitter(20)), 0, 0, 360, color, -1)
        cv2.rectangle(img, (30, 60), (100, 85), color, 3)
    elif category == "bags":
        cv2.rectangle(img, (jitter(30), jitter(50)), (jitter(98), jitter(110)), color, -1)
        cv2.ellipse(img, (64, 45), (25, 15), 0, 180, 360, color, 4)
    elif category == "electronics":
        cv2.rectangle(img, (jitter(25), jitter(25)), (jitter(103), jitter(103)), color, -1)
        cv2.line(img, (25, 60), (103, 60), (255, 255, 255), 2)
        cv2.line(img, (64, 25), (64, 103), (255, 255, 255), 2)
    elif category == "clothing":
        pts = np.array([[jitter(40), 20], [jitter(88), 20], [jitter(100), 108], [jitter(28), 108]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    else:
        for _ in range(4):
            cx, cy = rng.integers(30, 98, size=2)
            cv2.circle(img, (int(cx), int(cy)), int(rng.integers(10, 18)), color, -1)

    noise = rng.normal(0, 6, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def load_synthetic_dataset(samples_per_class: int = 80):
    rng = np.random.default_rng(42)
    X, y = [], []
    for label_idx, category in enumerate(CLASS_NAMES):
        for _ in range(samples_per_class):
            img = _synthetic_image(category, rng)
            X.append(_hog_features(img))
            y.append(label_idx)
    return np.array(X), np.array(y)


def main():
    print("Generating synthetic demo dataset (replace with a real dataset for production)...")
    X, y = load_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": "sklearn", "pipeline": pipeline, "class_names": CLASS_NAMES},
                config.PRODUCT_CLASSIFIER_PATH)
    print(f"Saved product classifier to {config.PRODUCT_CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
