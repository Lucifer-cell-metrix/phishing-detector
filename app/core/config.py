"""
AI Phishing Detector — Application Configuration
Loads settings from .env file using Pydantic Settings.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # App
    APP_NAME: str = "AI Phishing Detector"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    # Security
    API_KEY: str = "phish-detect-api-key-2024-secure"
    JWT_SECRET_KEY: str = "super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Model paths
    MODEL_PATH: str = "app/models/phishing_model.pkl"
    URL_MODEL_PATH: str = "app/models/url_model.pkl"

    # Database
    DATABASE_URL: str = "sqlite:///./phishing_detector.db"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
