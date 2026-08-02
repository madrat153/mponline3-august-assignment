"""Pydantic models used for request/response validation across the API."""
from typing import List, Optional
from pydantic import BaseModel, Field


class FaceRecognitionResponse(BaseModel):
    matched: bool
    customer_id: Optional[str] = None
    status: str = Field(..., description="'returning_customer' | 'new_customer' | 'no_face_detected'")
    confidence: Optional[float] = None
    faces_detected: int = 0
    visit_timestamp: Optional[str] = None


class FaceRegisterResponse(BaseModel):
    customer_id: str
    message: str


class ProductClassificationResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["The delivery was fast and the quality is great!"])


class SentimentResponse(BaseModel):
    text: str
    sentiment: str = Field(..., description="'positive' | 'negative' | 'neutral'")
    confidence: float
    scores: dict


class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["What is your return policy?"])
    session_id: Optional[str] = "anonymous"


class ChatbotResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    matched_via: str = Field(..., description="'rule' | 'ml_fallback' | 'default'")


class DashboardStats(BaseModel):
    total_visits: int
    unique_customers: int
    returning_customer_rate: float
    total_reviews_analyzed: int
    sentiment_breakdown: dict
    total_chat_messages: int
    top_chat_intents: List[dict]
