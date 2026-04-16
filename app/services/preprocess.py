"""
AI Phishing Detector — Text & URL Preprocessing
Cleans and extracts features from emails and URLs for ML prediction.
"""

import re
from urllib.parse import urlparse
from typing import Optional


# ============================================
# Phishing Indicator Keywords
# ============================================

URGENCY_KEYWORDS = [
    "urgent", "immediately", "act now", "expires", "suspended",
    "verify", "confirm", "update your", "click here", "limited time",
    "warning", "alert", "attention", "important", "deadline",
    "account will be", "unauthorized", "unusual activity", "security notice",
    "within 24 hours", "within 48 hours", "action required", "respond immediately",
]

PHISHING_KEYWORDS = [
    "password", "credit card", "social security", "bank account",
    "login credentials", "personal information", "ssn", "pin number",
    "wire transfer", "bitcoin", "cryptocurrency", "gift card",
    "winning", "lottery", "prize", "congratulations", "selected",
    "inheritance", "beneficiary", "million dollars",
]

SUSPICIOUS_TLDS = [
    ".xyz", ".top", ".club", ".online", ".site", ".buzz",
    ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".cc",
    ".icu", ".fun", ".space", ".website",
]

LEGITIMATE_DOMAINS = [
    "google.com", "gmail.com", "microsoft.com", "apple.com",
    "amazon.com", "facebook.com", "twitter.com", "linkedin.com",
    "github.com", "paypal.com", "netflix.com", "spotify.com",
    "yahoo.com", "outlook.com", "instagram.com", "youtube.com",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "tiny.cc", "lnkd.in",
]


# ============================================
# Email Text Preprocessing
# ============================================

def clean_email_text(text: str) -> str:
    """Clean email text for ML processing."""
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs (but we count them separately)
    text = re.sub(r"https?://\S+|www\.\S+", " [URL] ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " [EMAIL] ", text)

    # Remove phone numbers
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", " [PHONE] ", text)

    # Remove special characters but keep spaces
    text = re.sub(r"[^a-zA-Z\s\[\]]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_email(subject: str, body: str, sender: Optional[str] = None) -> str:
    """Combine and preprocess email components into a single text string."""
    combined = f"{subject} {body}"
    if sender:
        combined = f"{sender} {combined}"
    return clean_email_text(combined)


def extract_email_features(subject: str, body: str, sender: Optional[str] = None) -> dict:
    """Extract hand-crafted features from email for enhanced prediction."""
    combined = f"{subject} {body}".lower()

    features = {}

    # Count urgency keywords
    features["urgency_count"] = sum(
        1 for keyword in URGENCY_KEYWORDS if keyword in combined
    )

    # Count phishing keywords
    features["phishing_keyword_count"] = sum(
        1 for keyword in PHISHING_KEYWORDS if keyword in combined
    )

    # Count URLs in the text
    urls = re.findall(r"https?://\S+|www\.\S+", f"{subject} {body}")
    features["url_count"] = len(urls)

    # Check for IP addresses in URLs
    features["has_ip_url"] = int(
        any(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) for url in urls)
    )

    # Email length
    features["body_length"] = len(body)
    features["subject_length"] = len(subject)

    # Exclamation marks
    features["exclamation_count"] = combined.count("!")

    # ALL CAPS words
    words = f"{subject} {body}".split()
    features["caps_word_count"] = sum(
        1 for word in words if word.isupper() and len(word) > 2
    )

    # Sender domain analysis
    if sender:
        features["sender_suspicious"] = _analyze_sender(sender)
    else:
        features["sender_suspicious"] = 0

    return features


def _analyze_sender(sender: str) -> int:
    """Analyze if the sender email looks suspicious."""
    sender = sender.lower()

    # Check for misspelled legitimate domains
    suspicious_patterns = [
        r"paypa[l1]", r"amaz[o0]n", r"g[o0]{2}gle", r"micr[o0]s[o0]ft",
        r"app[l1]e", r"netf[l1]ix", r"faceb[o0]{2}k",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, sender) and not any(
            legit in sender for legit in LEGITIMATE_DOMAINS
        ):
            return 1

    # Check for suspicious TLDs
    for tld in SUSPICIOUS_TLDS:
        if sender.endswith(tld):
            return 1

    return 0


def get_email_indicators(subject: str, body: str, sender: Optional[str] = None) -> list[str]:
    """Get human-readable phishing indicators found in the email."""
    indicators = []
    combined = f"{subject} {body}".lower()

    # Check urgency
    found_urgency = [kw for kw in URGENCY_KEYWORDS if kw in combined]
    if found_urgency:
        indicators.append(f"Urgency language detected: {', '.join(found_urgency[:3])}")

    # Check phishing keywords
    found_phishing = [kw for kw in PHISHING_KEYWORDS if kw in combined]
    if found_phishing:
        indicators.append(f"Sensitive information requested: {', '.join(found_phishing[:3])}")

    # Check URLs
    urls = re.findall(r"https?://\S+|www\.\S+", f"{subject} {body}")
    if urls:
        indicators.append(f"Contains {len(urls)} URL(s) — verify before clicking")

    # Check for IP URLs
    for url in urls:
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            indicators.append("URL contains IP address instead of domain name")
            break

    # Check excessive caps
    words = f"{subject} {body}".split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) > 2)
    if caps_count > 3:
        indicators.append(f"Excessive use of capital letters ({caps_count} ALL-CAPS words)")

    # Check sender
    if sender:
        if _analyze_sender(sender):
            indicators.append(f"Suspicious sender address: {sender}")

    # Check exclamation marks
    if combined.count("!") > 3:
        indicators.append(f"Excessive exclamation marks ({combined.count('!')} found)")

    return indicators


# ============================================
# URL Preprocessing
# ============================================

def extract_url_features(url: str) -> dict:
    """Extract features from a URL for phishing detection."""
    features = {}

    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        # If URL can't be parsed, mark everything as suspicious
        return {
            "url_length": len(url),
            "has_https": 0,
            "dot_count": url.count("."),
            "has_ip": 1,
            "has_at": 1,
            "has_double_slash": 1,
            "has_dash": 1,
            "subdomain_count": 0,
            "path_length": 0,
            "has_suspicious_tld": 1,
            "is_shortener": 0,
            "digit_count": sum(c.isdigit() for c in url),
            "special_char_count": sum(not c.isalnum() and c not in ":/." for c in url),
            "has_suspicious_keywords": 1,
            "domain_length": 0,
        }

    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # Basic length features
    features["url_length"] = len(url)
    features["domain_length"] = len(hostname)
    features["path_length"] = len(path)

    # Protocol
    features["has_https"] = int(parsed.scheme == "https")

    # Dot count in hostname
    features["dot_count"] = hostname.count(".")

    # Has IP address instead of domain
    features["has_ip"] = int(bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname)))

    # Has @ symbol (redirect trick)
    features["has_at"] = int("@" in url)

    # Has double slash in path (redirect)
    features["has_double_slash"] = int("//" in path)

    # Has dash in domain
    features["has_dash"] = int("-" in hostname)

    # Subdomain count
    features["subdomain_count"] = max(0, hostname.count(".") - 1)

    # Suspicious TLD
    features["has_suspicious_tld"] = int(
        any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS)
    )

    # URL shortener
    features["is_shortener"] = int(
        any(shortener in hostname for shortener in URL_SHORTENERS)
    )

    # Digit count in hostname
    features["digit_count"] = sum(c.isdigit() for c in hostname)

    # Special characters
    features["special_char_count"] = sum(
        not c.isalnum() and c not in ":/.-" for c in url
    )

    # Suspicious keywords in URL
    suspicious_url_keywords = [
        "login", "signin", "verify", "update", "secure", "account",
        "banking", "confirm", "password", "credential", "auth",
        "suspend", "locked", "unusual", "alert",
    ]
    features["has_suspicious_keywords"] = int(
        any(kw in url.lower() for kw in suspicious_url_keywords)
    )

    return features


def get_url_indicators(url: str) -> list[str]:
    """Get human-readable phishing indicators from a URL."""
    indicators = []
    features = extract_url_features(url)

    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        hostname = parsed.hostname or ""
    except Exception:
        return ["Unable to parse URL — this is suspicious"]

    # Length check
    if features["url_length"] > 75:
        indicators.append(f"URL is unusually long ({features['url_length']} characters)")

    # HTTPS check
    if not features["has_https"]:
        indicators.append("URL does not use HTTPS encryption")

    # IP address
    if features["has_ip"]:
        indicators.append("URL uses IP address instead of domain name")

    # @ symbol
    if features["has_at"]:
        indicators.append("URL contains @ symbol (potential redirect trick)")

    # Suspicious TLD
    if features["has_suspicious_tld"]:
        indicators.append(f"URL uses suspicious top-level domain")

    # URL shortener
    if features["is_shortener"]:
        indicators.append("URL uses a URL shortening service (hides real destination)")

    # Too many subdomains
    if features["subdomain_count"] > 2:
        indicators.append(f"URL has {features['subdomain_count']} subdomains (may be impersonating)")

    # Suspicious keywords
    if features["has_suspicious_keywords"]:
        indicators.append("URL contains login/verify/account keywords")

    # Many dashes
    if hostname.count("-") > 2:
        indicators.append("Domain name contains multiple hyphens (common in phishing)")

    # Check for misspelled domains
    misspell_checks = {
        "paypal": ["paypa1", "paypai", "paypaI", "pay-pal", "paypol"],
        "google": ["g00gle", "googIe", "go0gle", "gooogle"],
        "apple": ["appIe", "app1e", "appl3"],
        "amazon": ["amaz0n", "amazom", "arnazon"],
        "microsoft": ["micros0ft", "mlcrosoft", "rnicrosoft"],
        "facebook": ["faceb00k", "facebock", "faceb0ok"],
        "netflix": ["netf1ix", "netfllx", "netfl1x"],
    }

    for brand, misspellings in misspell_checks.items():
        for misspell in misspellings:
            if misspell in hostname.lower() and brand not in hostname.lower():
                indicators.append(f"Domain appears to mimic '{brand}' (typosquatting)")
                break

    return indicators
