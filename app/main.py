"""
AI Phishing Detector v2.0 — Hybrid AI+ML Cybersecurity System
FastAPI entry point with CORS, routing, and startup events.

Architecture:
  🤖 ML Layer  → scikit-learn models for fast classification
  🧠 AI Layer  → Ollama/Mistral LLM for reasoning & explanations
  ⚙️ Rule Layer → Fallback detection when models aren't available
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.services.predictor import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — load models and check AI services on startup."""
    print("")
    print("🛡️  ═══════════════════════════════════════════")
    print("🛡️  AI PHISHING DETECTOR v2.0")
    print("🛡️  Hybrid AI + ML Cybersecurity System")
    print("🛡️  ═══════════════════════════════════════════")
    print("")
    predictor.load_models()
    print("")
    print("✅ System ready! Docs at http://localhost:8000/docs")
    print("═" * 50)
    yield
    print("👋 Shutting down AI Phishing Detector...")


# ============================================
# Create FastAPI Application
# ============================================

settings = get_settings()

app = FastAPI(
    title="🛡️ AI Phishing Detector — Hybrid AI+ML System",
    description=(
        "Detect phishing emails and malicious URLs using a **hybrid AI + ML** approach.\n\n"
        "## Architecture\n"
        "- 🤖 **ML Layer** — scikit-learn models for fast, accurate classification\n"
        "- 🧠 **AI Layer** — LLM-powered reasoning via Ollama (Mistral)\n"
        "- 📊 **Risk Scoring** — 0-100 threat rating with confidence metrics\n"
        "- 🔍 **Indicator Analysis** — Detailed breakdown of suspicious patterns\n"
        "- 📦 **Batch Processing** — Analyze multiple items at once\n\n"
        "## Authentication\n"
        "Use either:\n"
        "- `X-API-Key` header with your API key\n"
        "- `Bearer` token from `/api/v1/auth/login`\n"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ============================================
# CORS Middleware
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Include Routes
# ============================================

app.include_router(router)


# ============================================
# Root Endpoint
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API information."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict_email": "POST /api/v1/predict/email",
            "predict_url": "POST /api/v1/predict/url",
            "batch_emails": "POST /api/v1/predict/batch/emails",
            "batch_urls": "POST /api/v1/predict/batch/urls",
            "register": "POST /api/v1/auth/register",
            "login": "POST /api/v1/auth/login",
        },
    }
