"""
pipeline.py -- Module C1: unified pipeline. Loads all models ONCE at
API startup rather than per-request. Routers pull instances from here.
"""
from app.services.cv_service import FaceRecognitionService, ProductClassifierService
from app.services.nlp_service import SentimentService
from app.services.chatbot_service import ChatbotService


class SmartRetailPipeline:
    def __init__(self):
        self.face_service = FaceRecognitionService()
        self.product_classifier = ProductClassifierService()
        self.sentiment_service = SentimentService()
        self.chatbot_service = ChatbotService()

    def readiness(self) -> dict:
        return {
            "face_recognition": self.face_service.is_ready,
            "product_classifier": self.product_classifier.is_ready,
            "sentiment_analysis": self.sentiment_service.is_ready,
            "chatbot": self.chatbot_service.is_ready,
        }


pipeline = SmartRetailPipeline()
