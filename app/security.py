"""Lightweight API-key auth to simulate production-style security (C4)."""
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import API_KEY, API_KEY_HEADER_NAME

_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


async def require_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key is None or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid API key. Send it in the '{API_KEY_HEADER_NAME}' header.",
        )
    return api_key
