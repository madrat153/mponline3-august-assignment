"""
test_endpoints.py -- smoke tests for every API endpoint.
Run with: pytest -v
"""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import API_KEY, API_KEY_HEADER_NAME
from app.main import app

client = TestClient(app)
AUTH_HEADERS = {API_KEY_HEADER_NAME: API_KEY}


def _dummy_image_bytes(width=200, height=200) -> bytes:
    img = np.full((height, width, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200


def test_missing_api_key_rejected():
    resp = client.post("/analyze-sentiment", json={"text": "great product"})
    assert resp.status_code == 401


def test_wrong_api_key_rejected():
    resp = client.post(
        "/analyze-sentiment", json={"text": "great product"},
        headers={API_KEY_HEADER_NAME: "wrong-key"},
    )
    assert resp.status_code == 401


def test_classify_product_no_face_no_error_on_plain_image():
    files = {"file": ("product.jpg", _dummy_image_bytes(), "image/jpeg")}
    resp = client.post("/classify-product", files=files, headers=AUTH_HEADERS)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert "category" in body and "confidence" in body


def test_recognize_face_handles_no_face_gracefully():
    files = {"file": ("frame.jpg", _dummy_image_bytes(), "image/jpeg")}
    resp = client.post("/recognize-face", files=files, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("no_face_detected", "new_customer", "returning_customer")


def test_classify_product_rejects_non_image():
    files = {"file": ("note.txt", b"not an image", "text/plain")}
    resp = client.post("/classify-product", files=files, headers=AUTH_HEADERS)
    assert resp.status_code == 400


@pytest.mark.parametrize("text", [
    "The delivery was fast and the quality is amazing!",
    "This product broke after one use, very disappointed.",
    "It's an average product, does the job.",
])
def test_analyze_sentiment(text):
    resp = client.post("/analyze-sentiment", json={"text": text}, headers=AUTH_HEADERS)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        body = resp.json()
        assert body["sentiment"] in ("positive", "negative", "neutral")
        assert 0.0 <= body["confidence"] <= 1.0


def test_analyze_sentiment_rejects_empty_text():
    resp = client.post("/analyze-sentiment", json={"text": ""}, headers=AUTH_HEADERS)
    assert resp.status_code == 422


def test_chatbot_return_policy_intent():
    resp = client.post(
        "/chatbot", json={"message": "What is your return policy?", "session_id": "test"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "return_policy"
    assert body["matched_via"] in ("rule", "ml_fallback")


def test_chatbot_unknown_message_gets_fallback_or_low_confidence():
    resp = client.post(
        "/chatbot", json={"message": "purple elephants dance at midnight", "session_id": "test"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert "reply" in resp.json()


def test_dashboard_stats_shape():
    resp = client.get("/dashboard/stats", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    for key in ("total_visits", "unique_customers", "returning_customer_rate",
                "total_reviews_analyzed", "sentiment_breakdown", "total_chat_messages",
                "top_chat_intents"):
        assert key in body
