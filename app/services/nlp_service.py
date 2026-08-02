"""
nlp_service.py -- Module B2: sentiment analysis.
Baseline: TF-IDF + Logistic Regression, trained by
training/train_sentiment_model.py on data/reviews.csv.
"""
from typing import Dict, Tuple

import joblib

from app import config
from app.services.text_utils import preprocess


class SentimentService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load()

    def _load(self):
        if config.SENTIMENT_MODEL_PATH.exists() and config.SENTIMENT_VECTORIZER_PATH.exists():
            self.model = joblib.load(config.SENTIMENT_MODEL_PATH)
            self.vectorizer = joblib.load(config.SENTIMENT_VECTORIZER_PATH)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not self.is_ready:
            raise RuntimeError(
                "Sentiment model not found. Run training/train_sentiment_model.py first."
            )
        cleaned = preprocess(text)
        X = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        scores = {cls: float(p) for cls, p in zip(classes, probs)}
        best_idx = probs.argmax()
        return classes[best_idx], float(probs[best_idx]), scores
