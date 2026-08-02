"""
chatbot_service.py -- Module B3: hybrid FAQ chatbot.

1. Rule-based keyword/pattern matching against data/intents.json first.
2. ML fallback: TF-IDF + Logistic Regression intent classifier.
3. Graceful "connect to a human" fallback below confidence threshold.

Every exchange is logged to runtime_logs/chat_logs.json so
GET /dashboard/stats can report top intents.
"""
import json
import random
from datetime import datetime, timezone
from typing import Dict

import joblib

from app import config
from app.services.text_utils import preprocess


class ChatbotService:
    def __init__(self):
        self.intents = []
        self.model = None
        self.vectorizer = None
        self._load()

    def _load(self):
        if config.INTENTS_PATH.exists():
            self.intents = json.loads(config.INTENTS_PATH.read_text())["intents"]
        if config.CHATBOT_MODEL_PATH.exists() and config.CHATBOT_VECTORIZER_PATH.exists():
            self.model = joblib.load(config.CHATBOT_MODEL_PATH)
            self.vectorizer = joblib.load(config.CHATBOT_VECTORIZER_PATH)

    @property
    def is_ready(self) -> bool:
        return len(self.intents) > 0

    def _rule_match(self, message: str):
        cleaned = preprocess(message, stem=False)
        tokens = set(cleaned.split())
        best_tag, best_overlap = None, 0
        for intent in self.intents:
            for pattern in intent["patterns"]:
                pattern_tokens = set(preprocess(pattern, stem=False).split())
                if not pattern_tokens:
                    continue
                overlap = len(tokens & pattern_tokens) / len(pattern_tokens)
                if overlap >= 0.6 and overlap > best_overlap:
                    best_tag, best_overlap = intent["tag"], overlap
        return best_tag

    def _ml_match(self, message: str):
        if self.model is None or self.vectorizer is None:
            return None, 0.0
        cleaned = preprocess(message)
        X = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(X)[0]
        best_idx = probs.argmax()
        return self.model.classes_[best_idx], float(probs[best_idx])

    def _response_for(self, tag: str) -> str:
        for intent in self.intents:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        return "I'm not sure I understood that -- could you rephrase?"

    def get_reply(self, message: str, session_id: str = "anonymous") -> Dict:
        if not self.is_ready:
            raise RuntimeError("Chatbot intents not found. Ensure data/intents.json exists.")

        tag = self._rule_match(message)
        confidence = 1.0
        matched_via = "rule"

        if tag is None:
            ml_tag, ml_conf = self._ml_match(message)
            if ml_tag is not None and ml_conf >= config.CHATBOT_CONFIDENCE_THRESHOLD:
                tag, confidence, matched_via = ml_tag, ml_conf, "ml_fallback"
            else:
                tag, confidence, matched_via = "fallback", ml_conf, "default"

        reply = self._response_for(tag) if tag != "fallback" else (
            "I'm not 100% sure about that one -- I've flagged it for a human "
            "agent to follow up, or you can rephrase your question."
        )

        self._log_exchange(session_id, message, tag, reply, matched_via, confidence)
        return {"reply": reply, "intent": tag, "confidence": round(confidence, 3), "matched_via": matched_via}

    def _log_exchange(self, session_id, message, tag, reply, matched_via, confidence):
        record = {
            "session_id": session_id,
            "message": message,
            "intent": tag,
            "matched_via": matched_via,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        data = []
        if config.CHAT_LOG.exists():
            try:
                data = json.loads(config.CHAT_LOG.read_text())
            except json.JSONDecodeError:
                data = []
        data.append(record)
        config.CHAT_LOG.write_text(json.dumps(data, indent=2))
