"""
train_chatbot_model.py -- Module B3 training script.

Trains the ML fallback intent classifier (TF-IDF + Logistic Regression)
on the patterns in data/intents.json, used whenever the rule-based
keyword matcher doesn't find a confident match.
"""
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.services.text_utils import preprocess


def main():
    if not config.INTENTS_PATH.exists():
        raise SystemExit(f"intents.json not found at {config.INTENTS_PATH}")

    intents = json.loads(config.INTENTS_PATH.read_text())["intents"]

    texts, labels = [], []
    for intent in intents:
        for pattern in intent["patterns"]:
            texts.append(preprocess(pattern))
            labels.append(intent["tag"])

    print(f"Training on {len(texts)} patterns across {len(intents)} intents.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.CHATBOT_MODEL_PATH)
    joblib.dump(vectorizer, config.CHATBOT_VECTORIZER_PATH)
    print(f"Saved chatbot model to {config.CHATBOT_MODEL_PATH}")
    print(f"Saved chatbot vectorizer to {config.CHATBOT_VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
