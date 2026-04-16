"""
AI Phishing Detector — Pydantic Schemas
Request/Response models for the API.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================
# Auth Schemas
# ============================================

class UserRegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")


class UserLoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class AuthResponse(BaseModel):
    """Authentication response with JWT token."""
    access_token: str
    token_type: str = "bearer"
    username: str
    message: str


class UserResponse(BaseModel):
    """User info response."""
    username: str
    email: str


# ============================================
# Email Phishing Schemas
# ============================================

class EmailRequest(BaseModel):
    """Request to analyze an email for phishing."""
    subject: str = Field(
        ...,
        min_length=1,
        description="Email subject line",
        examples=["Your account has been compromised!"],
    )
    body: str = Field(
        ...,
        min_length=1,
        description="Email body text",
        examples=["Click here to verify your account immediately or it will be suspended."],
    )
    sender: Optional[str] = Field(
        None,
        description="Sender email address (optional)",
        examples=["security@paypa1.com"],
    )


# ============================================
# URL Phishing Schemas
# ============================================

class URLRequest(BaseModel):
    """Request to check if a URL is phishing."""
    url: str = Field(
        ...,
        min_length=1,
        description="URL to analyze",
        examples=["http://paypa1-secure.xyz/login/verify"],
    )


# ============================================
# Prediction Response Schemas
# ============================================

class PredictionResponse(BaseModel):
    """Prediction result for phishing analysis — ML + AI hybrid."""
    is_phishing: bool = Field(..., description="Whether the input is classified as phishing")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score from 0 (safe) to 100 (dangerous)")
    indicators: list[str] = Field(default_factory=list, description="Suspicious indicators found")
    suspicious_words: list[str] = Field(default_factory=list, description="Highlighted suspicious words/phrases")
    recommendation: str = Field(..., description="Recommended action for the user")
    ai_explanation: Optional[str] = Field(None, description="AI-generated explanation of the analysis (from LLM)")
    analysis_mode: str = Field(
        default="ml_only",
        description="Analysis mode: 'ml_only', 'ai_hybrid' (ML + LLM), or 'rule_based'"
    )


# ============================================
# Batch Schemas
# ============================================

class BatchEmailRequest(BaseModel):
    """Batch email analysis request."""
    emails: list[EmailRequest] = Field(..., min_length=1, max_length=50)


class BatchURLRequest(BaseModel):
    """Batch URL analysis request."""
    urls: list[URLRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""
    results: list[PredictionResponse]
    total: int
    phishing_count: int
    safe_count: int
    threat_summary: Optional[str] = Field(None, description="AI-generated threat summary for the batch")
