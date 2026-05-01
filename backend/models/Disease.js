const mongoose = require("mongoose");

const diseaseSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, unique: true, trim: true },
    symptoms: [{ type: String, trim: true, lowercase: true }],
    precautions: [{ type: String, trim: true }],
    medicines: [{ type: String, trim: true }],
    severity: {
      type: String,
      enum: ["low", "moderate", "high"],
      default: "moderate",
    },
  },
  { timestamps: true }
);

// Text index for search
diseaseSchema.index({ name: "text", symptoms: "text" });

module.exports = mongoose.model("Disease", diseaseSchema);
