"""
MediScan AI – Model Training Script
Trains a Decision Tree Classifier on symptom → disease dataset.
Saves model.pkl, vectorizer.pkl, label_map.pkl, and model_meta.json.
"""

import os
import json
import random
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ============================================================
# Configuration
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "disease_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def symptom_tokenizer(text):
    """Tokenizer function for CountVectorizer. Must be module-level for pickling."""
    return [s.strip().lower() for s in text.split(",")]


def load_dataset():
    """Load and validate the disease dataset."""
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    required_cols = ["symptoms", "disease", "precautions", "medicines"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"   ✅ Loaded {len(df)} rows, {df['disease'].nunique()} unique diseases")
    print(f"   📊 Disease distribution:")
    for disease, count in df["disease"].value_counts().items():
        print(f"      - {disease}: {count} samples")

    return df


def augment_data(df, augment_factor=3):
    """
    Augment the dataset by generating partial symptom subsets.
    For each row, randomly sample subsets of symptoms to create new training rows.
    This simulates real-world usage where patients may not report all symptoms.
    """
    print(f"\n🔄 Augmenting dataset (factor={augment_factor})...")
    augmented_rows = []

    for _, row in df.iterrows():
        symptoms = [s.strip().lower() for s in row["symptoms"].split(",")]

        for _ in range(augment_factor):
            # Sample between 2 and len(symptoms) symptoms
            n_sample = random.randint(max(2, len(symptoms) // 2), len(symptoms))
            sampled = random.sample(symptoms, min(n_sample, len(symptoms)))
            random.shuffle(sampled)

            augmented_rows.append({
                "symptoms": ",".join(sampled),
                "disease": row["disease"],
                "precautions": row["precautions"],
                "medicines": row["medicines"],
            })

    augmented_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([df, augmented_df], ignore_index=True)

    # Drop exact duplicates in the symptom+disease columns
    combined = combined.drop_duplicates(subset=["symptoms", "disease"])

    print(f"   ✅ Augmented: {len(df)} → {len(combined)} rows")
    return combined


def preprocess_symptoms(df):
    """
    Convert symptom strings to bag-of-words features.
    Each symptom string like 'fever,cough,headache' is treated as a document.
    """
    print("\n🔧 Preprocessing symptoms (bag-of-words)...")

    # Use CountVectorizer with comma as token separator
    vectorizer = CountVectorizer(
        tokenizer=symptom_tokenizer,
        token_pattern=None,  # Disable default pattern since we use custom tokenizer
    )

    X = vectorizer.fit_transform(df["symptoms"])
    feature_names = vectorizer.get_feature_names_out()

    print(f"   ✅ Created {len(feature_names)} features: {list(feature_names)}")

    return X, vectorizer


def train_model(X_train, y_train, X_test, y_test):
    """Train a Decision Tree Classifier with optimized hyperparameters."""
    print("\n🧠 Training Decision Tree Classifier...")

    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=1,
        min_samples_split=2,
        criterion="gini",
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # Cross-validation score on training data
    cv_folds = min(5, len(np.unique(y_train)))
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy")
    print(f"   ✅ Training complete!")
    print(f"   📊 Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
    print(f"   🌳 Tree depth: {model.get_depth()}")
    print(f"   🍃 Number of leaves: {model.get_n_leaves()}")

    # Training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"   🎯 Training accuracy: {train_accuracy:.2%}")

    # Test accuracy
    test_accuracy = model.score(X_test, y_test)
    print(f"   🎯 Test accuracy: {test_accuracy:.2%}")

    # Detailed classification report
    y_pred = model.predict(X_test)
    print(f"\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        "train_accuracy": round(train_accuracy * 100, 2),
        "test_accuracy": round(test_accuracy * 100, 2),
        "cv_mean_accuracy": round(cv_scores.mean() * 100, 2),
        "cv_std": round(cv_scores.std() * 100, 2),
        "tree_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
    }

    return model, metrics


def save_artifacts(model, vectorizer, label_map, metrics, feature_count, disease_count):
    """Save trained model, vectorizer, label map, and metadata to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)

    # Save model metadata
    meta = {
        "model_type": "DecisionTreeClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": feature_count,
        "disease_count": disease_count,
        "diseases": list(label_map.values()),
        "metrics": metrics,
        "version": "2.0.0",
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n💾 Saved artifacts:")
    print(f"   - Model:      {MODEL_PATH}")
    print(f"   - Vectorizer:  {VECTORIZER_PATH}")
    print(f"   - Label map:   {LABEL_MAP_PATH}")
    print(f"   - Metadata:    {META_PATH}")


def main():
    print("=" * 60)
    print("  MediScan AI – Model Training Pipeline v2.0")
    print("=" * 60)

    # 1. Load data
    df = load_dataset()

    # 2. Augment data
    df = augment_data(df, augment_factor=3)

    # 3. Preprocess
    X, vectorizer = preprocess_symptoms(df)
    y = df["disease"].values

    # Create label map for reverse lookup
    unique_diseases = sorted(df["disease"].unique())
    label_map = {i: disease for i, disease in enumerate(unique_diseases)}

    # 4. Train/test split
    print(f"\n📊 Splitting data: 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # 5. Train
    model, metrics = train_model(X_train, y_train, X_test, y_test)

    # 6. Save
    feature_names = vectorizer.get_feature_names_out()
    save_artifacts(model, vectorizer, label_map, metrics, len(feature_names), len(unique_diseases))

    # 7. Quick test prediction
    print("\n🧪 Quick test predictions:")
    test_cases = [
        "fever,cough,headache",
        "nausea,vomiting,diarrhea",
        "red eyes,itchy eyes",
        "restlessness,rapid heartbeat",
    ]
    for test_symptoms in test_cases:
        test_vector = vectorizer.transform([test_symptoms])
        prediction = model.predict(test_vector)[0]
        probabilities = model.predict_proba(test_vector)[0]
        max_prob = max(probabilities) * 100

        print(f"   Input: {test_symptoms:40s} → {prediction} ({max_prob:.1f}%)")

    print("\n" + "=" * 60)
    print("  ✅ Training complete! Model ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
