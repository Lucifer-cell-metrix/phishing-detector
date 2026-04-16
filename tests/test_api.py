"""
AI Phishing Detector v2.0 — API Tests
Tests for all API endpoints including health, auth, prediction, and AI features.
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.core.security import USERS_FILE

client = TestClient(app)
settings = get_settings()


@pytest.fixture(autouse=True, scope="session")
def cleanup_test_users():
    """Clean up test user data before and after test session."""
    # Remove users.json before tests
    if os.path.exists(USERS_FILE):
        os.remove(USERS_FILE)
    yield
    # Remove users.json after tests
    if os.path.exists(USERS_FILE):
        os.remove(USERS_FILE)


# ============================================
# Health Check Tests
# ============================================


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_endpoint(self):
        """Health endpoint should return 200 with system status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "system" in data
        assert "ml_email_model" in data["system"]
        assert "ai_llm_available" in data["system"]

    def test_root_endpoint(self):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data

    def test_system_status(self):
        """Should return detailed system status."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "layers" in data
        assert "ml" in data["layers"]
        assert "ai" in data["layers"]
        assert "rules" in data["layers"]


# ============================================
# Authentication Tests
# ============================================


class TestAuthentication:
    """Tests for auth endpoints."""

    def test_register_user(self):
        """Should register a new user successfully."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "testuser_api",
                "email": "testapi@example.com",
                "password": "testpass123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["username"] == "testuser_api"

    def test_register_duplicate_user(self):
        """Should reject duplicate username."""
        # First registration
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "duplicate_user",
                "email": "dup@example.com",
                "password": "testpass123",
            },
        )
        # Second registration with same username
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": "duplicate_user",
                "email": "dup2@example.com",
                "password": "testpass123",
            },
        )
        assert response.status_code == 400

    def test_login_success(self):
        """Should login with valid credentials."""
        # Register first
        client.post(
            "/api/v1/auth/register",
            json={
                "username": "logintest",
                "email": "login@example.com",
                "password": "testpass123",
            },
        )
        # Login
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "logintest", "password": "testpass123"},
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_invalid_credentials(self):
        """Should reject invalid credentials."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "nonexistent", "password": "wrongpass"},
        )
        assert response.status_code == 401


# ============================================
# Email Prediction Tests (v2.0 Hybrid)
# ============================================


class TestEmailPrediction:
    """Tests for email phishing prediction with hybrid AI+ML."""

    def test_predict_phishing_email(self):
        """Should detect a phishing email with all v2 fields."""
        response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": settings.API_KEY},
            json={
                "subject": "URGENT: Your account has been compromised!",
                "body": "Click here immediately to verify your identity or your account will be suspended. http://paypa1-login.xyz/verify",
                "sender": "security@paypa1.xyz",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Core fields
        assert "is_phishing" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert data["is_phishing"] is True

        # v2.0 fields
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100
        assert "suspicious_words" in data
        assert isinstance(data["suspicious_words"], list)
        assert "analysis_mode" in data
        assert data["analysis_mode"] in ["ml_only", "ai_hybrid", "ai_only", "rule_based"]

        # AI explanation (may be None if Ollama isn't running)
        assert "ai_explanation" in data

    def test_predict_legitimate_email(self):
        """Should recognize a legitimate email."""
        response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": settings.API_KEY},
            json={
                "subject": "Team meeting notes - March 15",
                "body": "Hi team, here are the notes from today's standup. Please update your Jira tickets by Friday.",
                "sender": "manager@company.com",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_phishing"] is False
        assert data["risk_score"] < 50  # Should be low risk

    def test_predict_email_risk_score(self):
        """Risk score should be higher for obviously phishing emails."""
        # Phishing email
        phish_response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": settings.API_KEY},
            json={
                "subject": "URGENT: Verify your password immediately!",
                "body": "Your credit card has been compromised! Click here to verify: http://192.168.1.1/verify. Confirm your SSN now!",
                "sender": "security@paypa1.tk",
            },
        )
        # Legitimate email
        safe_response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": settings.API_KEY},
            json={
                "subject": "Team lunch Friday",
                "body": "Hey, let's grab lunch at the Italian place. See you at noon!",
            },
        )

        phish_data = phish_response.json()
        safe_data = safe_response.json()

        assert phish_data["risk_score"] > safe_data["risk_score"]

    def test_predict_email_no_auth(self):
        """Should reject requests without authentication."""
        response = client.post(
            "/api/v1/predict/email",
            json={
                "subject": "Test",
                "body": "Test body",
            },
        )
        assert response.status_code == 401

    def test_predict_email_invalid_api_key(self):
        """Should reject invalid API key."""
        response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": "invalid-key"},
            json={
                "subject": "Test",
                "body": "Test body",
            },
        )
        assert response.status_code == 401

    def test_predict_email_with_jwt(self):
        """Should accept JWT token authentication."""
        # Register a unique user for this test
        import time
        username = f"jwttest_email_{int(time.time())}"
        reg_response = client.post(
            "/api/v1/auth/register",
            json={
                "username": username,
                "email": f"{username}@test.com",
                "password": "testpass123",
            },
        )
        assert reg_response.status_code == 200, f"Registration failed: {reg_response.json()}"
        token = reg_response.json()["access_token"]

        # Use token for prediction
        response = client.post(
            "/api/v1/predict/email",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "subject": "Meeting tomorrow",
                "body": "Don't forget we have a team sync at 2 PM.",
            },
        )
        assert response.status_code == 200


# ============================================
# URL Prediction Tests (v2.0 Hybrid)
# ============================================


class TestURLPrediction:
    """Tests for URL phishing prediction with hybrid AI+ML."""

    def test_predict_phishing_url(self):
        """Should detect a phishing URL with v2 fields."""
        response = client.post(
            "/api/v1/predict/url",
            headers={"X-API-Key": settings.API_KEY},
            json={"url": "http://192.168.1.100/bank-login"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_phishing"] is True
        assert data["risk_level"] in ["MEDIUM", "HIGH", "CRITICAL"]
        assert "risk_score" in data
        assert "suspicious_words" in data
        assert "analysis_mode" in data

    def test_predict_legitimate_url(self):
        """Should recognize a legitimate URL."""
        response = client.post(
            "/api/v1/predict/url",
            headers={"X-API-Key": settings.API_KEY},
            json={"url": "https://www.google.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_phishing"] is False

    def test_predict_url_no_auth(self):
        """Should reject requests without auth."""
        response = client.post(
            "/api/v1/predict/url",
            json={"url": "https://www.google.com"},
        )
        assert response.status_code == 401


# ============================================
# Batch Processing Tests
# ============================================


class TestBatchProcessing:
    """Tests for batch prediction endpoints."""

    def test_batch_emails(self):
        """Should process multiple emails with v2 response."""
        response = client.post(
            "/api/v1/predict/batch/emails",
            headers={"X-API-Key": settings.API_KEY},
            json={
                "emails": [
                    {
                        "subject": "URGENT: Verify now!",
                        "body": "Your account will be deleted. Click http://scam.xyz/verify",
                    },
                    {
                        "subject": "Team lunch Friday",
                        "body": "Hey, let's grab lunch at the Italian place this Friday.",
                    },
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert "results" in data
        assert len(data["results"]) == 2
        # Each result should have v2 fields
        for result in data["results"]:
            assert "risk_score" in result
            assert "analysis_mode" in result

    def test_batch_urls(self):
        """Should process multiple URLs with v2 response."""
        response = client.post(
            "/api/v1/predict/batch/urls",
            headers={"X-API-Key": settings.API_KEY},
            json={
                "urls": [
                    {"url": "http://paypa1-login.tk/verify"},
                    {"url": "https://www.github.com"},
                    {"url": "http://free-bitcoin.xyz/claim"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert "phishing_count" in data
        assert "safe_count" in data


# ============================================
# Validation Tests
# ============================================


class TestValidation:
    """Tests for request validation."""

    def test_empty_email_subject(self):
        """Should reject empty email subject."""
        response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": settings.API_KEY},
            json={"subject": "", "body": "Some body"},
        )
        assert response.status_code == 422

    def test_missing_url(self):
        """Should reject missing URL field."""
        response = client.post(
            "/api/v1/predict/url",
            headers={"X-API-Key": settings.API_KEY},
            json={},
        )
        assert response.status_code == 422

    def test_malformed_json(self):
        """Should reject malformed request body."""
        response = client.post(
            "/api/v1/predict/email",
            headers={"X-API-Key": settings.API_KEY},
            content="not valid json",
        )
        assert response.status_code == 422
