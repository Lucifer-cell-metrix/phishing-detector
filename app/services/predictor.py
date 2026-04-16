"""
AI Phishing Detector — Hybrid Prediction Service (ML + LLM)
Combines trained ML models with LLM reasoning for comprehensive phishing detection.

Architecture:
  1. ML Layer  → Fast classification + confidence score
  2. AI Layer  → LLM explanation of WHY (via Ollama/Mistral)
  3. Rule Layer → Fallback when models aren't available
"""

import os
import re
import joblib
import numpy as np
from typing import Optional

from app.core.config import get_settings
from app.services.preprocess import (
    preprocess_email,
    extract_email_features,
    get_email_indicators,
    extract_url_features,
    get_url_indicators,
    URGENCY_KEYWORDS,
    PHISHING_KEYWORDS,
)
from app.services.llm_service import (
    analyze_email_with_llm,
    analyze_url_with_llm,
    check_ollama_status,
)


# --- Suspicious words to highlight ---
SUSPICIOUS_WORDS = [
    "urgent", "immediately", "act now", "verify", "suspended",
    "click here", "confirm", "password", "credit card", "ssn",
    "social security", "bank account", "wire transfer", "bitcoin",
    "gift card", "lottery", "prize", "congratulations", "winner",
    "unauthorized", "unusual activity", "expires", "locked",
    "warning", "alert", "deadline", "limited time",
]


def _find_suspicious_words(text: str) -> list[str]:
    """Find and return all suspicious words/phrases present in the text."""
    text_lower = text.lower()
    found = []
    for word in SUSPICIOUS_WORDS:
        if word in text_lower:
            found.append(word)
    return found


def _calculate_risk_score(
    is_phishing: bool,
    confidence: float,
    indicators: list[str],
    features: dict = None,
) -> int:
    """
    Calculate a 0-100 risk score based on multiple signals.

    Scoring breakdown:
    - Base score from ML confidence:   0-50 points
    - Indicator count bonus:           0-25 points
    - Feature-based bonuses:           0-25 points
    """
    if not is_phishing:
        # Safe emails get a low risk score (inverse of confidence)
        base = max(0, int((1.0 - confidence) * 30))
        indicator_penalty = min(len(indicators) * 3, 15)
        return min(base + indicator_penalty, 40)  # Cap at 40 for safe

    # Phishing scoring
    base = int(confidence * 50)  # 0-50 from confidence
    indicator_bonus = min(len(indicators) * 5, 25)  # 0-25 from indicators

    feature_bonus = 0
    if features:
        if features.get("has_ip_url") or features.get("has_ip"):
            feature_bonus += 8
        if features.get("sender_suspicious"):
            feature_bonus += 7
        if features.get("urgency_count", 0) > 2:
            feature_bonus += 5
        if features.get("phishing_keyword_count", 0) > 2:
            feature_bonus += 5
    feature_bonus = min(feature_bonus, 25)  # Cap at 25

    return min(base + indicator_bonus + feature_bonus, 100)


class PhishingPredictor:
    """
    Hybrid ML + AI prediction engine.

    Combines:
    - scikit-learn models for fast classification
    - Ollama LLM for human-readable explanations
    - Rule-based fallback for when models aren't available
    """

    def __init__(self):
        self.email_model = None
        self.email_vectorizer = None
        self.url_model = None
        self.url_scaler = None
        self._loaded = False

    def load_models(self):
        """Load trained models from disk."""
        settings = get_settings()

        # Load email model
        if os.path.exists(settings.MODEL_PATH):
            model_data = joblib.load(settings.MODEL_PATH)
            self.email_model = model_data["model"]
            self.email_vectorizer = model_data["vectorizer"]
            print(f"✅ Email phishing model loaded from {settings.MODEL_PATH}")
        else:
            print(f"⚠️  Email model not found at {settings.MODEL_PATH}. Run training first!")

        # Load URL model
        if os.path.exists(settings.URL_MODEL_PATH):
            url_data = joblib.load(settings.URL_MODEL_PATH)
            self.url_model = url_data["model"]
            self.url_scaler = url_data.get("scaler")
            print(f"✅ URL phishing model loaded from {settings.URL_MODEL_PATH}")
        else:
            print(f"⚠️  URL model not found at {settings.URL_MODEL_PATH}. Run training first!")

        # Check Ollama status
        llm_status = check_ollama_status()
        if llm_status["ollama_running"]:
            if llm_status["model_available"]:
                print(f"✅ Ollama LLM ready (model: {llm_status['model']})")
            else:
                print(f"⚠️  Ollama running but model '{llm_status['model']}' not found.")
                print(f"   Available models: {llm_status['available_models']}")
                print(f"   Pull it with: ollama pull {llm_status['model']}")
        else:
            print("⚠️  Ollama not running. AI explanations disabled. Start with: ollama serve")

        self._loaded = True

    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded and (self.email_model is not None or self.url_model is not None)

    def get_system_status(self) -> dict:
        """Get full system status including ML and AI layers."""
        llm_status = check_ollama_status()
        return {
            "ml_email_model": self.email_model is not None,
            "ml_url_model": self.url_model is not None,
            "ai_llm_available": llm_status["ollama_running"] and llm_status["model_available"],
            "ai_llm_model": llm_status["model"],
            "ai_llm_status": llm_status,
        }

    def predict_email(
        self, subject: str, body: str, sender: Optional[str] = None
    ) -> dict:
        """
        Predict if an email is phishing using ML + AI hybrid approach.

        Returns dict with: is_phishing, confidence, risk_level, risk_score,
        indicators, suspicious_words, recommendation, ai_explanation, analysis_mode
        """
        combined_text = f"{subject} {body}"

        # Get indicators regardless of model
        indicators = get_email_indicators(subject, body, sender)

        # Find suspicious words
        suspicious_words = _find_suspicious_words(combined_text)

        # Extract features for risk scoring
        hand_features = extract_email_features(subject, body, sender)

        if self.email_model is None or self.email_vectorizer is None:
            # Fallback: rule-based prediction
            result = self._rule_based_email_prediction(subject, body, sender, indicators)
            result["suspicious_words"] = suspicious_words
            result["risk_score"] = _calculate_risk_score(
                result["is_phishing"], result["confidence"], indicators, hand_features
            )
            result["analysis_mode"] = "rule_based"

            # Still try LLM even without ML model
            ai_explanation = analyze_email_with_llm(
                subject, body, sender,
                ml_prediction=result["is_phishing"],
                ml_confidence=result["confidence"],
                indicators=indicators,
            )
            result["ai_explanation"] = ai_explanation
            if ai_explanation:
                result["analysis_mode"] = "ai_only"

            return result

        # --- ML Prediction ---
        cleaned_text = preprocess_email(subject, body, sender)

        # TF-IDF vectorization
        text_features = self.email_vectorizer.transform([cleaned_text])

        # Get hand-crafted features
        hand_feature_values = np.array(list(hand_features.values())).reshape(1, -1)

        # Combine features (TF-IDF + hand-crafted)
        from scipy.sparse import hstack
        combined_features = hstack([text_features, hand_feature_values])

        # Predict
        prediction = self.email_model.predict(combined_features)[0]
        probability = self.email_model.predict_proba(combined_features)[0]

        # Confidence is the probability of the predicted class
        confidence = float(max(probability))
        is_phishing = bool(prediction == 1)

        # Determine risk level & score
        risk_level = self._get_risk_level(confidence, is_phishing)
        risk_score = _calculate_risk_score(is_phishing, confidence, indicators, hand_features)
        recommendation = self._get_recommendation(risk_level, is_phishing)

        # --- AI Layer: Get LLM explanation ---
        ai_explanation = analyze_email_with_llm(
            subject, body, sender,
            ml_prediction=is_phishing,
            ml_confidence=confidence,
            indicators=indicators,
        )

        analysis_mode = "ai_hybrid" if ai_explanation else "ml_only"

        return {
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "indicators": indicators,
            "suspicious_words": suspicious_words,
            "recommendation": recommendation,
            "ai_explanation": ai_explanation,
            "analysis_mode": analysis_mode,
        }

    def predict_url(self, url: str) -> dict:
        """
        Predict if a URL is phishing using ML + AI hybrid approach.

        Returns dict with: is_phishing, confidence, risk_level, risk_score,
        indicators, suspicious_words, recommendation, ai_explanation, analysis_mode
        """
        # Get indicators
        indicators = get_url_indicators(url)

        # Find suspicious words in URL
        suspicious_words = _find_suspicious_words(url)

        # Extract URL features
        features = extract_url_features(url)
        feature_values = np.array(list(features.values())).reshape(1, -1)

        if self.url_model is None:
            # Fallback: rule-based prediction
            result = self._rule_based_url_prediction(url, features, indicators)
            result["suspicious_words"] = suspicious_words
            result["risk_score"] = _calculate_risk_score(
                result["is_phishing"], result["confidence"], indicators, features
            )
            result["analysis_mode"] = "rule_based"

            ai_explanation = analyze_url_with_llm(
                url,
                ml_prediction=result["is_phishing"],
                ml_confidence=result["confidence"],
                indicators=indicators,
            )
            result["ai_explanation"] = ai_explanation
            if ai_explanation:
                result["analysis_mode"] = "ai_only"

            return result

        # Scale features if scaler exists
        if self.url_scaler is not None:
            feature_values = self.url_scaler.transform(feature_values)

        # Predict
        prediction = self.url_model.predict(feature_values)[0]
        probability = self.url_model.predict_proba(feature_values)[0]

        confidence = float(max(probability))
        is_phishing = bool(prediction == 1)

        risk_level = self._get_risk_level(confidence, is_phishing)
        risk_score = _calculate_risk_score(is_phishing, confidence, indicators, features)
        recommendation = self._get_recommendation(risk_level, is_phishing, is_url=True)

        # --- AI Layer ---
        ai_explanation = analyze_url_with_llm(
            url,
            ml_prediction=is_phishing,
            ml_confidence=confidence,
            indicators=indicators,
        )

        analysis_mode = "ai_hybrid" if ai_explanation else "ml_only"

        return {
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "risk_score": risk_score,
            "indicators": indicators,
            "suspicious_words": suspicious_words,
            "recommendation": recommendation,
            "ai_explanation": ai_explanation,
            "analysis_mode": analysis_mode,
        }

    def _rule_based_email_prediction(
        self, subject: str, body: str, sender: Optional[str], indicators: list[str]
    ) -> dict:
        """Fallback rule-based email prediction when model isn't available."""
        features = extract_email_features(subject, body, sender)

        # Simple scoring
        score = 0
        score += min(features["urgency_count"] * 0.15, 0.3)
        score += min(features["phishing_keyword_count"] * 0.15, 0.3)
        score += features["has_ip_url"] * 0.15
        score += min(features["url_count"] * 0.05, 0.1)
        score += features["sender_suspicious"] * 0.2
        score += min(features["exclamation_count"] * 0.02, 0.1)
        score += min(features["caps_word_count"] * 0.03, 0.1)

        confidence = min(max(score, 0.1), 0.95)
        is_phishing = confidence > 0.5

        if not is_phishing:
            confidence = 1.0 - confidence

        risk_level = self._get_risk_level(confidence, is_phishing)
        recommendation = self._get_recommendation(risk_level, is_phishing)

        if not indicators:
            indicators = ["No suspicious patterns detected"] if not is_phishing else ["Multiple risk factors detected"]

        return {
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "risk_score": 0,  # Will be calculated by caller
            "indicators": indicators,
            "suspicious_words": [],
            "recommendation": recommendation,
            "ai_explanation": None,
            "analysis_mode": "rule_based",
        }

    def _rule_based_url_prediction(
        self, url: str, features: dict, indicators: list[str]
    ) -> dict:
        """Fallback rule-based URL prediction when model isn't available."""
        score = 0
        score += (1 - features["has_https"]) * 0.1
        score += features["has_ip"] * 0.2
        score += features["has_at"] * 0.15
        score += features["has_suspicious_tld"] * 0.15
        score += features["is_shortener"] * 0.1
        score += features["has_suspicious_keywords"] * 0.1
        score += min(features["subdomain_count"] * 0.05, 0.15)
        score += min(features["special_char_count"] * 0.03, 0.1)
        score += (1 if features["url_length"] > 75 else 0) * 0.1
        score += min(features["digit_count"] * 0.02, 0.1)

        confidence = min(max(score, 0.1), 0.95)
        is_phishing = confidence > 0.45

        if not is_phishing:
            confidence = 1.0 - confidence

        risk_level = self._get_risk_level(confidence, is_phishing)
        recommendation = self._get_recommendation(risk_level, is_phishing, is_url=True)

        if not indicators:
            indicators = ["URL appears legitimate"] if not is_phishing else ["Multiple risk factors detected"]

        return {
            "is_phishing": is_phishing,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "risk_score": 0,
            "indicators": indicators,
            "suspicious_words": [],
            "recommendation": recommendation,
            "ai_explanation": None,
            "analysis_mode": "rule_based",
        }

    @staticmethod
    def _get_risk_level(confidence: float, is_phishing: bool) -> str:
        """Determine risk level from confidence and prediction."""
        if not is_phishing:
            return "LOW"
        if confidence >= 0.9:
            return "CRITICAL"
        elif confidence >= 0.75:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def _get_recommendation(risk_level: str, is_phishing: bool, is_url: bool = False) -> str:
        """Generate recommendation based on risk level."""
        target = "URL" if is_url else "email"

        recommendations = {
            "CRITICAL": f"🚨 DO NOT interact with this {target}! This is almost certainly a phishing attempt. Delete immediately and report it.",
            "HIGH": f"⚠️ This {target} is very likely a phishing attempt. Do not click any links or provide personal information. Report as phishing.",
            "MEDIUM": f"⚡ This {target} shows some suspicious patterns. Exercise caution — verify the sender/source through official channels before taking any action.",
            "LOW": f"✅ This {target} appears to be safe. However, always verify unexpected requests through official channels.",
        }

        return recommendations.get(risk_level, recommendations["LOW"])


# --- Singleton Instance ---
predictor = PhishingPredictor()
