const mongoose = require("mongoose");

const predictionHistorySchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      default: null,
    },
    symptoms: [{ type: String, trim: true, lowercase: true }],
    prediction: { type: String, required: true },
    confidence: { type: Number, required: true },
    top3: [
      {
        disease: String,
        confidence: Number,
      },
    ],
    precautions: [String],
    medicines: [String],
    severity: String,
    source: {
      type: String,
      enum: ["ml_model", "fallback"],
      default: "ml_model",
    },
    explanation: String,
  },
  { timestamps: true }
);

// Index for user queries
predictionHistorySchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.model("PredictionHistory", predictionHistorySchema);
