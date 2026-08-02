"""
cv_utils.py -- Module A1 deliverable.

Reusable OpenCV preprocessing utilities used across the vision pipeline:
grayscale conversion, resizing, blurring, edge detection, and Haar-cascade
face detection.
"""
from typing import List, Tuple
import cv2
import numpy as np

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes -- is this a valid image file?")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_image(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny_edges(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    gray = to_grayscale(image)
    return cv2.Canny(gray, low_threshold, high_threshold)


def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    gray = to_grayscale(image)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(60, 60)
    )
    return [tuple(int(v) for v in f) for f in faces]


def crop_face(image: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
    x, y, w, h = box
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(image.shape[1], x + w + mx), min(image.shape[0], y + h + my)
    return image[y0:y1, x0:x1]


def draw_bounding_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=(0, 255, 0)) -> np.ndarray:
    out = image.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out
