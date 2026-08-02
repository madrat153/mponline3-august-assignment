"""
dashboard_service.py -- aggregates runtime logs written by the CV, NLP,
and chatbot services into the stats payload served at GET /dashboard/stats.
"""
import json
from collections import Counter
from typing import Dict

from app import config


def _load(path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def get_stats() -> Dict:
    visits = _load(config.VISITS_LOG)
    chats = _load(config.CHAT_LOG)
    sentiment_records = _load(config.SENTIMENT_LOG)

    total_visits = len(visits)
    customer_ids = {v["customer_id"] for v in visits if v.get("customer_id")}
    returning = [v for v in visits if v.get("status") == "returning_customer"]
    returning_rate = (len(returning) / total_visits) if total_visits else 0.0

    sentiment_counts = Counter(r["sentiment"] for r in sentiment_records)
    intent_counts = Counter(c["intent"] for c in chats)
    top_intents = [{"intent": tag, "count": count} for tag, count in intent_counts.most_common(5)]

    return {
        "total_visits": total_visits,
        "unique_customers": len(customer_ids),
        "returning_customer_rate": round(returning_rate, 3),
        "total_reviews_analyzed": len(sentiment_records),
        "sentiment_breakdown": dict(sentiment_counts),
        "total_chat_messages": len(chats),
        "top_chat_intents": top_intents,
    }


def log_sentiment_result(text: str, sentiment: str, confidence: float):
    from datetime import datetime, timezone
    record = {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    data = _load(config.SENTIMENT_LOG)
    data.append(record)
    config.SENTIMENT_LOG.write_text(json.dumps(data, indent=2))
