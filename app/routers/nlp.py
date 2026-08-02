from fastapi import APIRouter, Depends, HTTPException

from app.schemas import SentimentRequest, SentimentResponse
from app.security import require_api_key
from app.services import dashboard_service
from app.services.pipeline import pipeline

router = APIRouter(tags=["NLP"])


@router.post("/analyze-sentiment", response_model=SentimentResponse, dependencies=[Depends(require_api_key)])
async def analyze_sentiment(payload: SentimentRequest):
    """Runs a customer review/chat message through the sentiment model."""
    try:
        sentiment, confidence, scores = pipeline.sentiment_service.predict(payload.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    dashboard_service.log_sentiment_result(payload.text, sentiment, confidence)
    return {
        "text": payload.text,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 3) for k, v in scores.items()},
    }
