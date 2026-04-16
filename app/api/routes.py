"""
AI Phishing Detector — API Routes
All REST API endpoints for phishing detection, auth, and health checks.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import get_settings
from app.core.security import (
    verify_api_key_or_token,
    register_user,
    authenticate_user,
    create_access_token,
)
from app.schemas.email_schema import (
    EmailRequest,
    URLRequest,
    PredictionResponse,
    BatchEmailRequest,
    BatchURLRequest,
    BatchPredictionResponse,
    UserRegisterRequest,
    UserLoginRequest,
    AuthResponse,
    UserResponse,
)
from app.services.predictor import predictor
from app.services.llm_service import check_ollama_status, generate_threat_summary

# ============================================
# Router Setup
# ============================================
router = APIRouter()


# ============================================
# Health Check & System Status
# ============================================

@router.get("/health", tags=["Health"])
async def health_check():
    """Check if the API is running and models are loaded."""
    system_status = predictor.get_system_status()
    return {
        "status": "healthy",
        "models_loaded": predictor.is_loaded(),
        "service": "AI Phishing Detector",
        "version": "2.0.0",
        "system": {
            "ml_email_model": system_status["ml_email_model"],
            "ml_url_model": system_status["ml_url_model"],
            "ai_llm_available": system_status["ai_llm_available"],
            "ai_llm_model": system_status["ai_llm_model"],
        },
    }


@router.get("/api/v1/status", tags=["Health"])
async def system_status():
    """Get detailed system status including ML and AI layer info."""
    return {
        "service": "AI Phishing Detector — Hybrid AI+ML System",
        "version": "2.0.0",
        "layers": {
            "ml": {
                "email_model_loaded": predictor.email_model is not None,
                "url_model_loaded": predictor.url_model is not None,
                "description": "scikit-learn models for fast classification",
            },
            "ai": {
                **check_ollama_status(),
                "description": "LLM reasoning layer for human-readable explanations",
            },
            "rules": {
                "active": True,
                "description": "Rule-based fallback when ML models aren't available",
            },
        },
    }


# ============================================
# Authentication Endpoints
# ============================================

@router.post(
    "/api/v1/auth/register",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Register a new user",
)
async def register(request: UserRegisterRequest):
    """Create a new user account and get an access token."""
    user = register_user(request.username, request.email, request.password)

    settings = get_settings()
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return AuthResponse(
        access_token=access_token,
        username=user["username"],
        message="Registration successful! Use this token for authenticated requests.",
    )


@router.post(
    "/api/v1/auth/login",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Login and get access token",
)
async def login(request: UserLoginRequest):
    """Authenticate with username/password and receive a JWT token."""
    user = authenticate_user(request.username, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    settings = get_settings()
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return AuthResponse(
        access_token=access_token,
        username=user["username"],
        message="Login successful!",
    )


# ============================================
# Email Phishing Detection
# ============================================

@router.post(
    "/api/v1/predict/email",
    response_model=PredictionResponse,
    tags=["Phishing Detection"],
    summary="Analyze an email for phishing (ML + AI)",
)
async def predict_email(
    request: EmailRequest,
    auth: str = Depends(verify_api_key_or_token),
):
    """
    Analyze an email's subject and body for phishing indicators.

    **Hybrid Analysis:**
    - 🤖 **ML Layer**: Fast classification with confidence score
    - 🧠 **AI Layer**: LLM-powered explanation of findings (if Ollama is running)
    - 📊 **Risk Score**: 0-100 threat rating

    **Authentication:** Requires either `X-API-Key` header or `Bearer` JWT token.
    """
    result = predictor.predict_email(
        subject=request.subject,
        body=request.body,
        sender=request.sender,
    )

    return PredictionResponse(**result)


# ============================================
# URL Phishing Detection
# ============================================

@router.post(
    "/api/v1/predict/url",
    response_model=PredictionResponse,
    tags=["Phishing Detection"],
    summary="Check if a URL is phishing (ML + AI)",
)
async def predict_url(
    request: URLRequest,
    auth: str = Depends(verify_api_key_or_token),
):
    """
    Analyze a URL to determine if it's a phishing/malicious website.

    **Hybrid Analysis:**
    - 🤖 **ML Layer**: Feature extraction + classification
    - 🧠 **AI Layer**: Domain analysis with LLM reasoning
    - 📊 **Risk Score**: 0-100 threat rating

    **Authentication:** Requires either `X-API-Key` header or `Bearer` JWT token.
    """
    result = predictor.predict_url(url=request.url)

    return PredictionResponse(**result)


# ============================================
# Batch Processing
# ============================================

@router.post(
    "/api/v1/predict/batch/emails",
    response_model=BatchPredictionResponse,
    tags=["Batch Processing"],
    summary="Analyze multiple emails at once",
)
async def predict_batch_emails(
    request: BatchEmailRequest,
    auth: str = Depends(verify_api_key_or_token),
):
    """
    Analyze multiple emails for phishing in a single request (max 50).

    **Note:** AI explanations are disabled for batch processing to save time.
    A threat summary is generated instead.

    **Authentication:** Requires either `X-API-Key` header or `Bearer` JWT token.
    """
    results = []
    for email in request.emails:
        result = predictor.predict_email(
            subject=email.subject,
            body=email.body,
            sender=email.sender,
        )
        results.append(PredictionResponse(**result))

    phishing_count = sum(1 for r in results if r.is_phishing)

    # Generate AI threat summary for the batch
    threat_summary = generate_threat_summary(
        [r.model_dump() for r in results]
    )

    return BatchPredictionResponse(
        results=results,
        total=len(results),
        phishing_count=phishing_count,
        safe_count=len(results) - phishing_count,
        threat_summary=threat_summary,
    )


@router.post(
    "/api/v1/predict/batch/urls",
    response_model=BatchPredictionResponse,
    tags=["Batch Processing"],
    summary="Check multiple URLs at once",
)
async def predict_batch_urls(
    request: BatchURLRequest,
    auth: str = Depends(verify_api_key_or_token),
):
    """
    Check multiple URLs for phishing in a single request (max 100).

    **Authentication:** Requires either `X-API-Key` header or `Bearer` JWT token.
    """
    results = []
    for url_req in request.urls:
        result = predictor.predict_url(url=url_req.url)
        results.append(PredictionResponse(**result))

    phishing_count = sum(1 for r in results if r.is_phishing)

    threat_summary = generate_threat_summary(
        [r.model_dump() for r in results]
    )

    return BatchPredictionResponse(
        results=results,
        total=len(results),
        phishing_count=phishing_count,
        safe_count=len(results) - phishing_count,
        threat_summary=threat_summary,
    )
