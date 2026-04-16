"""
AI Phishing Detector — Model Training Script
Trains both email phishing and URL phishing detection models.

Usage:
    python notebooks/train.py

Output:
    - app/models/phishing_model.pkl  (Email model + TF-IDF vectorizer)
    - app/models/url_model.pkl       (URL model + feature scaler)
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.preprocess import (
    preprocess_email,
    extract_email_features,
    extract_url_features,
)

warnings.filterwarnings("ignore")


def train_email_model():
    """Train the email phishing detection model."""
    print("=" * 60)
    print("📧 TRAINING EMAIL PHISHING MODEL")
    print("=" * 60)

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "emails.csv")
    df = pd.read_csv(data_path)
    print(f"\n📊 Dataset: {len(df)} emails ({df['label'].sum()} phishing, {(df['label'] == 0).sum()} legitimate)")

    # Preprocess text
    print("\n🔄 Preprocessing emails...")
    df["cleaned_text"] = df.apply(
        lambda row: preprocess_email(row["subject"], row["body"]),
        axis=1,
    )

    # Extract hand-crafted features
    print("🔧 Extracting features...")
    hand_crafted_features = df.apply(
        lambda row: extract_email_features(row["subject"], row["body"]),
        axis=1,
    )
    hand_crafted_df = pd.DataFrame(hand_crafted_features.tolist())

    # TF-IDF Vectorization
    print("📝 TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_features = vectorizer.fit_transform(df["cleaned_text"])

    # Combine features
    hand_crafted_array = hand_crafted_df.values
    X = hstack([tfidf_features, hand_crafted_array])
    y = df["label"].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📂 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Train model
    print("\n🤖 Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📈 Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print(f"🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    print(f"\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"   CV F1 Scores: {cv_scores}")
    print(f"   Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "app", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "phishing_model.pkl")

    joblib.dump(
        {"model": model, "vectorizer": vectorizer},
        model_path,
    )
    print(f"\n💾 Email model saved to: {model_path}")

    return accuracy, f1


def train_url_model():
    """Train the URL phishing detection model."""
    print("\n" + "=" * 60)
    print("🔗 TRAINING URL PHISHING MODEL")
    print("=" * 60)

    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "urls.csv")
    df = pd.read_csv(data_path)
    print(f"\n📊 Dataset: {len(df)} URLs ({df['label'].sum()} phishing, {(df['label'] == 0).sum()} legitimate)")

    # Extract URL features
    print("\n🔧 Extracting URL features...")
    features = df["url"].apply(extract_url_features)
    features_df = pd.DataFrame(features.tolist())

    X = features_df.values
    y = df["label"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📂 Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Train model
    print("\n🤖 Training Gradient Boosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n📈 Results:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print(f"🔢 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation
    print(f"\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1")
    print(f"   CV F1 Scores: {cv_scores}")
    print(f"   Mean F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    print(f"\n📊 Top Feature Importances:")
    feature_names = list(features_df.columns)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in range(min(10, len(feature_names))):
        idx = sorted_idx[i]
        print(f"   {feature_names[idx]:25s} → {importances[idx]:.4f}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), "..", "app", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "url_model.pkl")

    joblib.dump(
        {"model": model, "scaler": scaler},
        model_path,
    )
    print(f"\n💾 URL model saved to: {model_path}")

    return accuracy, f1


def main():
    """Run full training pipeline."""
    print("🛡️  AI PHISHING DETECTOR — MODEL TRAINING")
    print("=" * 60)

    # Train email model
    email_acc, email_f1 = train_email_model()

    # Train URL model
    url_acc, url_f1 = train_url_model()

    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"   📧 Email Model:  Accuracy={email_acc:.4f}  F1={email_f1:.4f}")
    print(f"   🔗 URL Model:    Accuracy={url_acc:.4f}  F1={url_f1:.4f}")
    print(f"\n   Models saved to: app/models/")
    print(f"   Ready to start the API with: uvicorn app.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
