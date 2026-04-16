"""
AI Phishing Detector — LLM Service (Ollama Integration)
Uses a local LLM (Mistral via Ollama) to provide human-readable
explanations of WHY an email or URL is phishing.

This adds an AI reasoning layer on top of the ML classification.
If Ollama isn't running, the system gracefully falls back to rule-based explanations.
"""

import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# --- Ollama Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
OLLAMA_TIMEOUT = 30  # seconds


def _call_ollama(prompt: str) -> Optional[str]:
    """
    Send a prompt to Ollama and return the response.
    Returns None if Ollama is unavailable or errors out.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,      # Low temp for consistent analysis
                    "num_predict": 500,       # Limit response length
                    "top_p": 0.9,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        logger.warning("Ollama is not running. LLM analysis skipped. Start Ollama with: ollama serve")
        return None
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out after %ds", OLLAMA_TIMEOUT)
        return None
    except Exception as e:
        logger.error("Ollama error: %s", str(e))
        return None


def check_ollama_status() -> dict:
    """Check if Ollama is running and the model is available."""
    try:
        # Check if Ollama is running
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            model_available = any(OLLAMA_MODEL in m for m in models)
            return {
                "ollama_running": True,
                "model": OLLAMA_MODEL,
                "model_available": model_available,
                "available_models": models,
            }
    except Exception:
        pass

    return {
        "ollama_running": False,
        "model": OLLAMA_MODEL,
        "model_available": False,
        "available_models": [],
    }


def analyze_email_with_llm(
    subject: str, body: str, sender: str = None,
    ml_prediction: bool = None, ml_confidence: float = None,
    indicators: list = None,
) -> Optional[str]:
    """
    Use LLM to analyze an email and explain why it's phishing or safe.

    Provides the ML prediction as context so the LLM can give a coherent explanation.
    """
    sender_line = f"\nFrom: {sender}" if sender else ""
    ml_context = ""
    if ml_prediction is not None:
        verdict = "PHISHING" if ml_prediction else "SAFE"
        ml_context = f"""
--- ML Model Result ---
Verdict: {verdict}
Confidence: {ml_confidence:.1%}
Indicators found: {', '.join(indicators or ['None'])}
"""

    prompt = f"""You are a cybersecurity expert analyzing emails for phishing threats.

Analyze the following email and provide:
1. **Verdict**: Is it PHISHING or SAFE?
2. **Risk Score**: Rate 0 (safe) to 100 (definite phishing)
3. **Key Findings**: List the top 3-5 suspicious or safe elements
4. **Explanation**: Brief paragraph explaining your reasoning
5. **Recommended Action**: What should the user do?

Be concise and professional. Use bullet points.
{ml_context}
--- Email ---{sender_line}
Subject: {subject}
Body: {body}
--- End ---

Your analysis:"""

    return _call_ollama(prompt)


def analyze_url_with_llm(
    url: str,
    ml_prediction: bool = None, ml_confidence: float = None,
    indicators: list = None,
) -> Optional[str]:
    """
    Use LLM to analyze a URL and explain why it's phishing or safe.
    """
    ml_context = ""
    if ml_prediction is not None:
        verdict = "PHISHING" if ml_prediction else "SAFE"
        ml_context = f"""
--- ML Model Result ---
Verdict: {verdict}
Confidence: {ml_confidence:.1%}
Indicators found: {', '.join(indicators or ['None'])}
"""

    prompt = f"""You are a cybersecurity expert analyzing URLs for phishing threats.

Analyze the following URL and determine:
1. **Verdict**: Is it PHISHING or SAFE?
2. **Risk Score**: Rate 0 (safe) to 100 (definite phishing)
3. **Domain Analysis**: Break down the domain, TLD, and any suspicious patterns
4. **Key Findings**: List top 3-5 suspicious or safe elements
5. **Recommended Action**: What should the user do?

Be concise and professional. Use bullet points.
{ml_context}
--- URL ---
{url}
--- End ---

Your analysis:"""

    return _call_ollama(prompt)


def generate_threat_summary(results: list) -> Optional[str]:
    """
    Generate a summary threat report for batch analysis results.
    """
    items = []
    for i, r in enumerate(results[:10], 1):  # Limit to 10 for prompt size
        status = "PHISHING" if r.get("is_phishing") else "SAFE"
        items.append(f"  {i}. [{status}] Confidence: {r.get('confidence', 0):.0%}")

    results_text = "\n".join(items)

    prompt = f"""You are a cybersecurity analyst writing a brief threat summary.

The following items were scanned for phishing:
{results_text}

Write a 2-3 sentence executive summary of the threat landscape.
Include: total threats found, overall risk assessment, and one key recommendation.

Summary:"""

    return _call_ollama(prompt)
