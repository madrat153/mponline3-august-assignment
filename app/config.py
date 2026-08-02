"""
Central configuration for the Smart Retail & Customer Intelligence Platform.
Values can be overridden via environment variables (see .env.example).
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

API_KEY = os.getenv("SMART_RETAIL_API_KEY", "dev-demo-key-123")
API_KEY_HEADER_NAME = "X-API-Key"

MODELS_DIR = BASE_DIR / "app" / "models"
DATA_DIR = BASE_DIR / "data"

PRODUCT_CLASSIFIER_PATH = MODELS_DIR / "product_classifier.pkl"
FACE_RECOGNIZER_PATH = MODELS_DIR / "face_recognizer.yml"
FACE_LABELS_PATH = MODELS_DIR / "face_labels.pkl"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
SENTIMENT_VECTORIZER_PATH = MODELS_DIR / "sentiment_vectorizer.pkl"
CHATBOT_MODEL_PATH = MODELS_DIR / "chatbot_model.pkl"
CHATBOT_VECTORIZER_PATH = MODELS_DIR / "chatbot_vectorizer.pkl"
INTENTS_PATH = DATA_DIR / "intents.json"
REVIEWS_CSV_PATH = DATA_DIR / "reviews.csv"

RUNTIME_DIR = BASE_DIR / "runtime_logs"
VISITS_LOG = RUNTIME_DIR / "customer_visits.json"
SENTIMENT_LOG = RUNTIME_DIR / "reviews_log.json"
CHAT_LOG = RUNTIME_DIR / "chat_logs.json"

FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "70.0"))
CHATBOT_CONFIDENCE_THRESHOLD = float(os.getenv("CHATBOT_CONFIDENCE_THRESHOLD", "0.30"))

os.makedirs(RUNTIME_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
