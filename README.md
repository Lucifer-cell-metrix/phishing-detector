# 🛡️ AI Phishing Detector v2.0 — Hybrid AI+ML Cybersecurity System

> **Detect phishing emails and malicious URLs using a hybrid AI + Machine Learning approach.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![Ollama](https://img.shields.io/badge/Ollama-Mistral_LLM-blueviolet)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

This is NOT a basic project — it's a **real cybersecurity system** with three detection layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| 🤖 **ML Layer** | scikit-learn (Random Forest + Gradient Boosting) | Fast classification with confidence scoring |
| 🧠 **AI Layer** | Ollama + Mistral LLM | Human-readable threat explanations |
| 📋 **Rule Layer** | Custom heuristics | Fallback detection when models aren't available |

---

## 🚀 Features

- **📧 Email Phishing Detection** — Analyze email content with ML + AI explanation
- **🔗 URL/Website Checker** — Verify if a URL is legitimate or phishing
- **📊 Risk Score (0-100)** — Quantified threat rating
- **🧠 AI Explanation** — LLM tells you WHY it's phishing
- **🔴 Suspicious Word Highlighting** — Flagged words/phrases
- **🔍 Indicator Analysis** — Detailed breakdown of suspicious patterns
- **🔐 Dual Authentication** — API key for developers + JWT login for users
- **📦 Batch Processing** — Analyze multiple emails/URLs at once
- **🎨 Two Frontends** — Streamlit (demo) + React (production)

---

## 📁 Project Structure

```
ai-phishing-detector/
├── app/                          # FastAPI Backend
│   ├── api/routes.py             # API endpoints
│   ├── core/config.py            # Configuration
│   ├── core/security.py          # Auth (API key + JWT)
│   ├── models/                   # Trained ML models (.pkl)
│   ├── schemas/email_schema.py   # Request/Response models
│   ├── services/predictor.py     # Hybrid ML+AI prediction engine
│   ├── services/preprocess.py    # Text & URL preprocessing
│   ├── services/llm_service.py   # 🧠 LLM integration (Ollama)
│   └── main.py                   # Entry point
├── data/                         # Training datasets
│   ├── emails.csv
│   └── urls.csv
├── notebooks/train.py            # Model training script
├── frontend/
│   ├── app.py                    # Streamlit demo UI
│   └── react-app/                # React production UI
├── tests/test_api.py
├── requirements.txt
├── Dockerfile
└── .env
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/ai-phishing-detector.git
cd ai-phishing-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Models

```bash
python notebooks/train.py
```

### 3. (Optional) Setup Ollama for AI Layer

```bash
# Install Ollama: https://ollama.com
ollama serve          # Start Ollama server
ollama pull mistral   # Download Mistral model
```

> Without Ollama, the system works with ML-only mode. AI explanations are disabled.

### 4. Start the Backend

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 5. Start Frontend

**Streamlit (Demo):**
```bash
python -m streamlit run frontend/app.py
```

**React (Production):**
```bash
cd frontend/react-app
npm install
npm run dev
```

---

## 📡 API Documentation

Once the backend is running, visit **http://localhost:8000/docs** for interactive Swagger documentation.

### Key Endpoints

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/api/v1/predict/email` | Analyze email (ML + AI) | API Key / JWT |
| `POST` | `/api/v1/predict/url` | Check URL (ML + AI) | API Key / JWT |
| `POST` | `/api/v1/predict/batch/emails` | Batch email analysis | API Key / JWT |
| `POST` | `/api/v1/predict/batch/urls` | Batch URL analysis | API Key / JWT |
| `POST` | `/api/v1/auth/register` | User registration | None |
| `POST` | `/api/v1/auth/login` | User login | None |
| `GET` | `/health` | Health check + system status | None |
| `GET` | `/api/v1/status` | Detailed system status | None |

### Example Response (v2.0)

```json
{
  "is_phishing": true,
  "confidence": 0.94,
  "risk_level": "CRITICAL",
  "risk_score": 87,
  "indicators": [
    "Urgency language detected: urgent, verify, immediately",
    "Suspicious sender address: security@paypa1.xyz",
    "URL contains IP address instead of domain name"
  ],
  "suspicious_words": ["urgent", "verify", "immediately", "password"],
  "recommendation": "🚨 DO NOT interact with this email! Delete immediately.",
  "ai_explanation": "This email is phishing because it uses urgency tactics...",
  "analysis_mode": "ai_hybrid"
}
```

---

## 🐳 Docker

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

---

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

---

## 📄 License

MIT License — feel free to use for personal and commercial projects.
