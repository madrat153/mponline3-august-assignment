from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException

from app.schemas import FaceRecognitionResponse, FaceRegisterResponse, ProductClassificationResponse
from app.security import require_api_key
from app.services import cv_utils
from app.services.pipeline import pipeline

router = APIRouter(tags=["Computer Vision"])


async def _read_upload_as_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    raw = await file.read()
    try:
        return cv_utils.read_image_from_bytes(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/recognize-face", response_model=FaceRecognitionResponse, dependencies=[Depends(require_api_key)])
async def recognize_face(file: UploadFile = File(..., description="A frame/photo containing a customer's face")):
    """Detects a face, compares it against enrolled customer templates, and
    logs a visit. Returns whether this is a returning or new customer."""
    image = await _read_upload_as_image(file)
    result = pipeline.face_service.recognize(image)
    return result


@router.post("/register-face", response_model=FaceRegisterResponse, dependencies=[Depends(require_api_key)])
async def register_face(
    file: UploadFile = File(..., description="A clear photo of the customer's face to enroll"),
    customer_id: Optional[str] = Form(None, description="Optional existing customer ID to attach this face to"),
):
    """Enrolls a new face template for loyalty/returning-customer detection.
    Requires explicit customer consent before calling in a real deployment."""
    image = await _read_upload_as_image(file)
    try:
        assigned_id = pipeline.face_service.register_face(image, customer_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"customer_id": assigned_id, "message": "Face enrolled successfully."}


@router.post("/classify-product", response_model=ProductClassificationResponse, dependencies=[Depends(require_api_key)])
async def classify_product(file: UploadFile = File(..., description="Product photo to categorize")):
    """Classifies a product photo into one of the trained catalog categories."""
    image = await _read_upload_as_image(file)
    try:
        category, confidence, scores = pipeline.product_classifier.predict(image)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"category": category, "confidence": round(confidence, 3), "all_scores": scores}
