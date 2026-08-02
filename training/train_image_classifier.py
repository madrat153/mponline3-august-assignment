"""
Train the product image classifier (Module A2).

Default (no external dataset required): generates a small synthetic
product-silhouette dataset -- five shape-coded categories drawn with
OpenCV, with randomized rotation/scale/color/noise per sample -- then
trains HOG features + a RandomForestClassifier on it. This keeps the demo
fully offline and fast (seconds, not GPU-hours).

For a real capstone submission, replace `generate_synthetic_dataset()`
with a loader over a real labeled image folder (e.g. Fashion-MNIST, or
the Kaggle "Retail Product Checkout Dataset" mentioned in the brief) --
everything downstream (feature extraction, training, serialization)
stays the same. See train_image_classifier_tf.py for the MobileNetV2 /
TensorFlow upgrade path over such a real dataset.

Usage:
    python training/train_image_classifier.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.config import PRODUCT_CATEGORIES, PRODUCT_CLASSIFIER_PATH
from app.services.cv_service import extract_features

IMG_SIZE = 96
SAMPLES_PER_CATEGORY = 120


def _random_color():
    return (random.randint(20, 235), random.randint(20, 235), random.randint(20, 235))


def _blank_canvas():
    bg = random.randint(230, 255)
    return np.full((IMG_SIZE, IMG_SIZE, 3), bg, dtype=np.uint8)


def _draw_clothing(canvas):
    # A simple T-shirt silhouette: torso rectangle + two sleeve triangles.
    color = _random_color()
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    w = random.randint(28, 40)
    h = random.randint(36, 50)
    cv2.rectangle(canvas, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, -1)
    sleeve_len = random.randint(10, 18)
    pts_left = np.array([[cx - w // 2, cy - h // 2], [cx - w // 2 - sleeve_len, cy - h // 4],
                          [cx - w // 2, cy]], np.int32)
    pts_right = np.array([[cx + w // 2, cy - h // 2], [cx + w // 2 + sleeve_len, cy - h // 4],
                           [cx + w // 2, cy]], np.int32)
    cv2.fillPoly(canvas, [pts_left], color)
    cv2.fillPoly(canvas, [pts_right], color)
    return canvas


def _draw_shoes(canvas):
    # An elongated rounded sole + curved upper.
    color = _random_color()
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2 + random.randint(-5, 5)
    axes = (random.randint(28, 38), random.randint(10, 16))
    cv2.ellipse(canvas, (cx, cy), axes, 0, 0, 360, color, -1)
    cv2.ellipse(canvas, (cx - axes[0] // 3, cy - axes[1]), (axes[0] // 2, axes[1]), 0, 180, 360, color, -1)
    return canvas


def _draw_bags(canvas):
    # A rounded rectangle body + an arch handle on top.
    color = _random_color()
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2 + 8
    w, h = random.randint(32, 44), random.randint(28, 38)
    cv2.rectangle(canvas, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, -1)
    handle_r = random.randint(10, 16)
    cv2.ellipse(canvas, (cx, cy - h // 2), (handle_r, handle_r), 0, 180, 360, color, 4)
    return canvas


def _draw_electronics(canvas):
    # An outer rectangle "device" with an inner "screen" rectangle.
    color = _random_color()
    screen_color = tuple(max(0, c - 80) for c in color)
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    w, h = random.randint(34, 46), random.randint(24, 34)
    cv2.rectangle(canvas, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, -1)
    pad = 4
    cv2.rectangle(canvas, (cx - w // 2 + pad, cy - h // 2 + pad),
                   (cx + w // 2 - pad, cy + h // 2 - pad), screen_color, -1)
    return canvas


def _draw_groceries(canvas):
    # A round "produce" shape with a small stem.
    color = _random_color()
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = random.randint(18, 26)
    cv2.circle(canvas, (cx, cy), r, color, -1)
    cv2.line(canvas, (cx, cy - r), (cx, cy - r - 6), (60, 90, 40), 2)
    return canvas


_DRAW_FUNCS = {
    "clothing": _draw_clothing,
    "shoes": _draw_shoes,
    "bags": _draw_bags,
    "electronics": _draw_electronics,
    "groceries": _draw_groceries,
}


def _augment(image):
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), angle, random.uniform(0.9, 1.1))
    image = cv2.warpAffine(image, M, (IMG_SIZE, IMG_SIZE), borderValue=(245, 245, 245))
    noise = np.random.randint(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def generate_synthetic_dataset(samples_per_category: int = SAMPLES_PER_CATEGORY):
    """Yields (image, label) pairs for every category in PRODUCT_CATEGORIES."""
    for category in PRODUCT_CATEGORIES:
        draw_fn = _DRAW_FUNCS[category]
        for _ in range(samples_per_category):
            canvas = _blank_canvas()
            canvas = draw_fn(canvas)
            canvas = _augment(canvas)
            yield canvas, category


def main():
    random.seed(7)
    np.random.seed(7)

    print(f"Generating synthetic dataset ({SAMPLES_PER_CATEGORY} samples x {len(PRODUCT_CATEGORIES)} categories) ...")
    X, y = [], []
    for image, label in generate_synthetic_dataset():
        X.append(extract_features(image))
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"  dataset shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    print("Training RandomForestClassifier on HOG features ...")
    model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=7)
    model.fit(X_train, y_train)

    print("\nEvaluation on held-out test set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    bundle = {"model": model, "categories": PRODUCT_CATEGORIES}
    joblib.dump(bundle, PRODUCT_CLASSIFIER_PATH)
    print(f"\nSaved model -> {PRODUCT_CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
