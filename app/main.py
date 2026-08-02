"""
main.py -- FastAPI entrypoint for the AI-Powered Smart Retail &
Customer Intelligence Platform.

Run locally:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
All endpoints (except /health and /) require an `X-API-Key` header.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import vision, nlp, chatbot, dashboard
from app.services.pipeline import pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Smart Retail Platform starting up. Model readiness:", pipeline.readiness())
    yield
    print("Smart Retail Platform shutting down.")


app = FastAPI(
    title="AI-Powered Smart Retail & Customer Intelligence Platform",
    description=(
        "A production-style API unifying computer vision (face recognition, "
        "product classification), NLP (sentiment analysis), and a hybrid "
        "FAQ chatbot for retail/e-commerce customer intelligence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)
app.include_router(dashboard.router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "models_ready": pipeline.readiness()}


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "Smart Retail & Customer Intelligence Platform is running.",
        "docs": "/docs",
        "health": "/health",
    }
