from fastapi import APIRouter, Depends

from app.schemas import DashboardStats
from app.security import require_api_key
from app.services import dashboard_service

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard/stats", response_model=DashboardStats, dependencies=[Depends(require_api_key)])
async def dashboard_stats():
    """Aggregate visit/sentiment/chatbot stats for a simple frontend chart."""
    return dashboard_service.get_stats()
