from fastapi import APIRouter, Depends, HTTPException

from app.schemas import ChatbotRequest, ChatbotResponse
from app.security import require_api_key
from app.services.pipeline import pipeline

router = APIRouter(tags=["Chatbot"])


@router.post("/chatbot", response_model=ChatbotResponse, dependencies=[Depends(require_api_key)])
async def chatbot(payload: ChatbotRequest):
    """Hybrid FAQ chatbot: rule-based intent matching first, ML fallback second."""
    try:
        result = pipeline.chatbot_service.get_reply(payload.message, payload.session_id or "anonymous")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result
