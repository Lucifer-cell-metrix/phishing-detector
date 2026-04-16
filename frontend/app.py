"""
AI Phishing Detector — Streamlit Demo Frontend
Interactive web UI for testing email and URL phishing detection.

Run with: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import json
import time

# ============================================
# Configuration
# ============================================

API_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "phish-detect-api-key-2024-secure"

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="🛡️ AI Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(147,51,234,0.1));
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #00d4ff, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* Risk level cards */
    .risk-critical {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        animation: pulse 2s infinite;
    }

    .risk-high {
        background: linear-gradient(135deg, #ea580c, #c2410c);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }

    .risk-medium {
        background: linear-gradient(135deg, #d97706, #b45309);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }

    .risk-low {
        background: linear-gradient(135deg, #059669, #047857);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Indicator styling */
    .indicator-item {
        background: rgba(255,255,255,0.05);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #dc2626;
        color: #e2e8f0;
    }

    .indicator-safe {
        border-left-color: #059669;
    }

    /* Stats cards */
    .stat-card {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .stat-card h3 {
        color: #00d4ff;
        font-size: 2rem;
        margin: 0;
    }

    .stat-card p {
        color: #94a3b8;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Helper Functions
# ============================================

def check_api_health():
    """Check if the API is running."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def predict_email(subject: str, body: str, sender: str = None):
    """Send email for phishing prediction."""
    payload = {"subject": subject, "body": body}
    if sender:
        payload["sender"] = sender

    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/predict/email",
            headers={"X-API-Key": st.session_state.get("api_key", DEFAULT_API_KEY)},
            json=payload,
            timeout=30,
        )
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the backend is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


def predict_url(url: str):
    """Send URL for phishing prediction."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/predict/url",
            headers={"X-API-Key": st.session_state.get("api_key", DEFAULT_API_KEY)},
            json={"url": url},
            timeout=30,
        )
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the backend is running on port 8000."}
    except Exception as e:
        return {"error": str(e)}


def display_result(result: dict):
    """Display prediction result with rich formatting."""
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        return

    # Risk level card
    risk_level = result.get("risk_level", "LOW")
    risk_class = f"risk-{risk_level.lower()}"
    risk_emoji = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "✅"}.get(risk_level, "ℹ️")

    st.markdown(
        f'<div class="{risk_class}">{risk_emoji} RISK LEVEL: {risk_level}</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        phishing_label = "🚫 PHISHING" if result["is_phishing"] else "✅ SAFE"
        st.metric("Classification", phishing_label)

    with col2:
        confidence_pct = f"{result['confidence'] * 100:.1f}%"
        st.metric("Confidence", confidence_pct)

    with col3:
        st.metric("Risk Level", risk_level)

    # Confidence progress bar
    st.write("**Confidence Score:**")
    confidence_color = (
        "red" if result["is_phishing"] and result["confidence"] > 0.7
        else "orange" if result["is_phishing"]
        else "green"
    )
    st.progress(result["confidence"])

    # Indicators
    st.write("")
    st.subheader("🔍 Indicators Found")

    if result.get("indicators"):
        for indicator in result["indicators"]:
            safe_class = "indicator-safe" if not result["is_phishing"] else ""
            icon = "⚠️" if result["is_phishing"] else "✅"
            st.markdown(
                f'<div class="indicator-item {safe_class}">{icon} {indicator}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No specific indicators detected.")

    # Recommendation
    st.write("")
    st.subheader("💡 Recommendation")
    st.info(result.get("recommendation", "Exercise caution with unfamiliar communications."))


# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # API Key
    api_key = st.text_input(
        "API Key",
        value=DEFAULT_API_KEY,
        type="password",
        help="Enter your API key for authentication",
    )
    st.session_state["api_key"] = api_key

    # API Status
    st.markdown("---")
    st.markdown("### 📡 API Status")

    if check_api_health():
        st.success("✅ API is running")
    else:
        st.error("❌ API is offline")
        st.caption("Start the API with:")
        st.code("uvicorn app.main:app --reload", language="bash")

    # About
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **AI Phishing Detector** uses Machine Learning to detect phishing emails and malicious URLs in real-time.

    **Features:**
    - 📧 Email analysis
    - 🔗 URL verification
    - 📊 Confidence scoring
    - 🔍 Indicator breakdown

    **Tech Stack:**
    - FastAPI (Backend)
    - scikit-learn (ML)
    - Streamlit (Frontend)
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using AI")


# ============================================
# Main Content
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI Phishing Detector</h1>
    <p>Detect phishing emails and malicious URLs using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Detection mode tabs
tab_email, tab_url, tab_batch = st.tabs(["📧 Email Analysis", "🔗 URL Checker", "📦 Batch Analysis"])

# ============================================
# Email Analysis Tab
# ============================================

with tab_email:
    st.markdown("### 📧 Analyze an Email for Phishing")
    st.markdown("Paste the email content below and click **Analyze** to check for phishing.")

    with st.form("email_form"):
        col1, col2 = st.columns([3, 1])

        with col1:
            email_subject = st.text_input(
                "Email Subject",
                placeholder="e.g., Your account has been compromised!",
            )

        with col2:
            email_sender = st.text_input(
                "Sender (optional)",
                placeholder="e.g., security@paypa1.xyz",
            )

        email_body = st.text_area(
            "Email Body",
            height=200,
            placeholder="Paste the email content here...",
        )

        submit_email = st.form_submit_button("🔍 Analyze Email", use_container_width=True)

    if submit_email:
        if not email_subject or not email_body:
            st.warning("⚠️ Please enter both subject and body.")
        else:
            with st.spinner("🔄 Analyzing email..."):
                result = predict_email(email_subject, email_body, email_sender or None)
                time.sleep(0.5)  # Brief delay for UX

            st.markdown("---")
            display_result(result)

    # Example emails
    with st.expander("📋 Try Example Emails"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚫 Phishing Example:**")
            st.code("""Subject: URGENT: Your PayPal account is locked!
Body: We detected unauthorized access. Click here to verify: http://paypa1-secure.xyz/verify
Sender: security@paypa1.xyz""")

        with col2:
            st.markdown("**✅ Legitimate Example:**")
            st.code("""Subject: Team standup notes - March 15
Body: Hi team, here are today's meeting notes. Sprint review is Friday at 2 PM.
Sender: manager@company.com""")


# ============================================
# URL Checker Tab
# ============================================

with tab_url:
    st.markdown("### 🔗 Check if a URL is Safe")
    st.markdown("Enter a URL below to verify if it's legitimate or a phishing website.")

    with st.form("url_form"):
        url_input = st.text_input(
            "URL to Check",
            placeholder="e.g., http://paypa1-secure.xyz/login/verify",
        )

        submit_url = st.form_submit_button("🔍 Check URL", use_container_width=True)

    if submit_url:
        if not url_input:
            st.warning("⚠️ Please enter a URL.")
        else:
            with st.spinner("🔄 Analyzing URL..."):
                result = predict_url(url_input)
                time.sleep(0.5)

            st.markdown("---")
            display_result(result)

    # Example URLs
    with st.expander("📋 Try Example URLs"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚫 Phishing URLs:**")
            examples_phishing = [
                "http://paypa1-secure.xyz/login",
                "http://amaz0n-deals.club/signin",
                "http://192.168.1.100/bank-login",
                "http://g00gle-security.top/reset",
            ]
            for url in examples_phishing:
                st.code(url)

        with col2:
            st.markdown("**✅ Legitimate URLs:**")
            examples_legit = [
                "https://www.google.com",
                "https://www.paypal.com",
                "https://www.github.com",
                "https://www.amazon.com",
            ]
            for url in examples_legit:
                st.code(url)


# ============================================
# Batch Analysis Tab
# ============================================

with tab_batch:
    st.markdown("### 📦 Batch URL Analysis")
    st.markdown("Enter multiple URLs (one per line) to check them all at once.")

    with st.form("batch_form"):
        batch_urls = st.text_area(
            "URLs (one per line)",
            height=200,
            placeholder="https://www.google.com\nhttp://suspicious-site.xyz/login\nhttps://www.github.com",
        )

        submit_batch = st.form_submit_button("🔍 Analyze All URLs", use_container_width=True)

    if submit_batch:
        if not batch_urls.strip():
            st.warning("⚠️ Please enter at least one URL.")
        else:
            urls = [u.strip() for u in batch_urls.strip().split("\n") if u.strip()]

            with st.spinner(f"🔄 Analyzing {len(urls)} URLs..."):
                results = []
                progress_bar = st.progress(0)

                for i, url in enumerate(urls):
                    result = predict_url(url)
                    result["url"] = url
                    results.append(result)
                    progress_bar.progress((i + 1) / len(urls))

            st.markdown("---")

            # Summary stats
            if results and "error" not in results[0]:
                phishing = sum(1 for r in results if r.get("is_phishing", False))
                safe = len(results) - phishing

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyzed", len(results))
                with col2:
                    st.metric("🚫 Phishing", phishing)
                with col3:
                    st.metric("✅ Safe", safe)

                st.write("")

                # Results table
                for result in results:
                    status = "🚫 PHISHING" if result.get("is_phishing") else "✅ SAFE"
                    confidence = f"{result.get('confidence', 0) * 100:.1f}%"
                    risk = result.get("risk_level", "N/A")

                    with st.expander(f"{status} | {result.get('url', 'N/A')} ({confidence})"):
                        display_result(result)


# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #64748b;">🛡️ AI Phishing Detector v1.0 | '
    'Built with FastAPI + scikit-learn + Streamlit</p>',
    unsafe_allow_html=True,
)
