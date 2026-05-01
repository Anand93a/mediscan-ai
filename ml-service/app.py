"""
MediScan AI – Flask ML Service
Serves disease predictions via a REST API using a trained Decision Tree model.
"""

import os
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the tokenizer function so the pickled vectorizer can be deserialized
from train_model import symptom_tokenizer  # noqa: F401

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

PORT = int(os.environ.get("ML_PORT", 5002))

# ============================================================
# Logging Setup (Structured)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MediScan-ML")

# ============================================================
# Flask App
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# Load Model
# ============================================================
model = None
vectorizer = None
label_map = None
model_meta = None


def load_model():
    """Load the trained model, vectorizer, label map, and metadata from disk."""
    global model, vectorizer, label_map, model_meta

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        logger.error("Please run 'python train_model.py' first to train the model.")
        return False

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_map = joblib.load(LABEL_MAP_PATH)

        # Load metadata if available
        if os.path.exists(META_PATH):
            with open(META_PATH, "r") as f:
                model_meta = json.load(f)
            logger.info(f"📊 Model metadata loaded: v{model_meta.get('version', 'unknown')}")
        else:
            model_meta = {"version": "unknown", "metrics": {}}

        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        logger.info(f"   Classes: {list(label_map.values())}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def sanitize_symptoms(symptoms):
    """Sanitize and deduplicate symptom inputs."""
    seen = set()
    sanitized = []
    for s in symptoms:
        cleaned = s.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            sanitized.append(cleaned)
    return sanitized


def get_feature_importance_explanation(input_vector, feature_names, top_n=5):
    """
    Generate a human-readable explanation of why the model made this prediction
    using feature importances from the Decision Tree.
    """
    importances = model.feature_importances_

    # Get the features that are present in the input (non-zero values)
    input_array = input_vector.toarray().flatten()
    active_features = []

    for i, val in enumerate(input_array):
        if val > 0 and i < len(feature_names):
            active_features.append({
                "symptom": feature_names[i],
                "importance": float(importances[i]),
            })

    # Sort by importance descending
    active_features.sort(key=lambda x: x["importance"], reverse=True)
    top_features = active_features[:top_n]

    if not top_features:
        return "Prediction based on overall symptom pattern analysis."

    # Build explanation string
    parts = []
    for feat in top_features:
        level = "high" if feat["importance"] > 0.15 else "moderate" if feat["importance"] > 0.05 else "contributing"
        parts.append(f"{feat['symptom']} ({level} importance)")

    explanation = f"Key symptoms driving this prediction: {', '.join(parts)}."
    return explanation


# ============================================================
# API Routes
# ============================================================

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "MediScan AI ML Service",
        "model_loaded": model is not None,
        "version": model_meta.get("version", "unknown") if model_meta else "unknown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return model metadata including accuracy, diseases, and training info."""
    if model is None or model_meta is None:
        return jsonify({
            "error": "Model not loaded. Run train_model.py first."
        }), 503

    feature_names = list(vectorizer.get_feature_names_out()) if vectorizer else []

    return jsonify({
        "model_type": model_meta.get("model_type", "DecisionTreeClassifier"),
        "version": model_meta.get("version", "unknown"),
        "trained_at": model_meta.get("trained_at", "unknown"),
        "disease_count": model_meta.get("disease_count", 0),
        "diseases": model_meta.get("diseases", []),
        "feature_count": model_meta.get("feature_count", 0),
        "features": feature_names,
        "metrics": model_meta.get("metrics", {}),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict disease from symptoms.

    Input:  { "symptoms": ["fever", "cough", "headache"] }
    Output: {
        "disease": "Common Flu (Influenza)",
        "confidence": 87.5,
        "top_3": [...],
        "explanation": "..."
    }
    """
    start_time = datetime.now()

    # Validate model is loaded
    if model is None:
        logger.error("Prediction attempted but model is not loaded")
        return jsonify({
            "error": "ML model is not loaded. Please run train_model.py first."
        }), 503

    # Parse request
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({
            "error": "Invalid request. Please provide a 'symptoms' array.",
            "example": {"symptoms": ["fever", "cough", "headache"]},
        }), 400

    symptoms = data["symptoms"]
    if not isinstance(symptoms, list) or len(symptoms) == 0:
        return jsonify({"error": "Please provide at least one symptom."}), 400

    try:
        # Sanitize input
        symptoms = sanitize_symptoms(symptoms)
        if len(symptoms) == 0:
            return jsonify({"error": "No valid symptoms provided after sanitization."}), 400

        # Prepare input: join symptoms into comma-separated string (same as training format)
        symptom_string = ",".join(symptoms)

        # Vectorize
        input_vector = vectorizer.transform([symptom_string])

        # Predict with probabilities
        prediction = model.predict(input_vector)[0]
        probabilities = model.predict_proba(input_vector)[0]

        # Get class labels from the model
        classes = model.classes_

        # Build top-3 predictions
        prob_pairs = list(zip(classes, probabilities))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        top_3 = [
            {
                "disease": str(disease),
                "confidence": round(float(prob) * 100, 1),
            }
            for disease, prob in prob_pairs[:3]
        ]

        # Primary prediction confidence
        primary_confidence = round(float(max(probabilities)) * 100, 1)

        # Generate explanation
        feature_names = vectorizer.get_feature_names_out()
        explanation = get_feature_importance_explanation(input_vector, feature_names)

        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        # Log prediction (structured)
        logger.info(
            f"🔮 Prediction: symptoms={symptoms} → disease={prediction} "
            f"confidence={primary_confidence}% latency={latency_ms:.0f}ms"
        )

        return jsonify({
            "disease": str(prediction),
            "confidence": primary_confidence,
            "top_3": top_3,
            "explanation": explanation,
            "source": "ml_model",
            "latency_ms": round(latency_ms, 1),
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  MediScan AI – ML Service")
    print("=" * 60)

    if load_model():
        print(f"\n🚀 Starting Flask server on port {PORT}...")
        print(f"   POST http://localhost:{PORT}/predict")
        print(f"   GET  http://localhost:{PORT}/health")
        print(f"   GET  http://localhost:{PORT}/model-info\n")
        app.run(host="0.0.0.0", port=PORT, debug=False)
    else:
        print("\n❌ Cannot start server: model not loaded.")
        print("   Run 'python train_model.py' first to train the model.\n")
