"""
train_sentiment_model.py -- Module B2 training script.

Trains TF-IDF + Logistic Regression on data/reviews.csv. Ships with a
small generated sample so the pipeline runs offline; swap in a real
dataset (e.g. Kaggle "Women's E-Commerce Clothing Reviews") for a real
submission -- same review_text,sentiment column schema.

Stretch goal: swap LogisticRegression for a fine-tuned DistilBERT --
see the docstring in app/services/nlp_service.py.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.services.text_utils import preprocess


def load_dataset() -> pd.DataFrame:
    if not config.REVIEWS_CSV_PATH.exists():
        raise SystemExit(f"Dataset not found at {config.REVIEWS_CSV_PATH}")
    df = pd.read_csv(config.REVIEWS_CSV_PATH)
    df = df.dropna(subset=["review_text", "sentiment"])
    return df


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} labeled reviews. Class distribution:\n{df['sentiment'].value_counts()}")

    df["cleaned"] = df["review_text"].apply(preprocess)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.SENTIMENT_MODEL_PATH)
    joblib.dump(vectorizer, config.SENTIMENT_VECTORIZER_PATH)
    print(f"Saved model to {config.SENTIMENT_MODEL_PATH}")
    print(f"Saved vectorizer to {config.SENTIMENT_VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
