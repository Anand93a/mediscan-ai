// ============================================================
// server.js — MediScan AI Backend
// Express server with OCR (Tesseract.js) + Disease Prediction
// ============================================================

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

// Import custom services
const { extractTextFromImage } = require("./ocrService");
const { predictFromSymptoms, predictFromReport, mergePredictions } = require("./predictService");

// ============================================================
// App Configuration
// ============================================================
const app = express();
const PORT = process.env.PORT || 5001;

// ============================================================
// Middleware
// ============================================================
app.use(cors());                     // Allow cross-origin requests
app.use(express.json());             // Parse JSON bodies
app.use(express.urlencoded({ extended: true })); // Parse URL-encoded bodies

// Request logger — logs every incoming request
app.use((req, _res, next) => {
  console.log(`[Server] ${req.method} ${req.path} — ${new Date().toISOString()}`);
  next();
});

// ============================================================
// Multer Setup — File Upload Configuration
// ============================================================

// Ensure the uploads directory exists
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
  console.log("[Server] 📁 Created /uploads directory");
}

// Configure disk storage
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (_req, file, cb) => {
    // Generate unique filename: mediscan_<timestamp>_<random>.<ext>
    const ext = path.extname(file.originalname).toLowerCase();
    const uniqueName = `mediscan_${Date.now()}_${Math.random().toString(36).slice(2, 8)}${ext}`;
    cb(null, uniqueName);
  },
});

// File filter — only allow JPG and PNG images
const fileFilter = (_req, file, cb) => {
  const allowedTypes = /jpeg|jpg|png/;
  const extOk = allowedTypes.test(path.extname(file.originalname).toLowerCase());
  const mimeOk = allowedTypes.test(file.mimetype);

  if (extOk && mimeOk) {
    cb(null, true);
  } else {
    cb(new Error("Only .jpg and .png image files are allowed"), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB max
});

// ============================================================
// ROUTES
// ============================================================

// Health check
app.get("/api/health", (_req, res) => {
  res.json({
    status: "ok",
    service: "MediScan AI Backend",
    timestamp: new Date().toISOString(),
  });
});

// ----------------------------------------------------------
// POST /api/predict — Main Prediction Endpoint
// ----------------------------------------------------------
// Accepts multipart form-data:
//   • file      (optional) — An image file (jpg/png)
//   • symptoms  (optional) — JSON string array, e.g. '["fever","cough"]'
//
// Flow:
//   1. If file uploaded → OCR → extract text → report prediction
//   2. If symptoms provided → symptom prediction
//   3. Merge both results and return
// ----------------------------------------------------------
app.post("/api/predict", upload.single("file"), async (req, res) => {
  try {
    console.log("\n========================================");
    console.log("[Server] 🚀 New prediction request");
    console.log("========================================");

    // Guard: ensure req.body exists (undefined when no form data is sent)
    if (!req.body) req.body = {};

    let extractedText = "";
    let symptomResult = { disease: "Unknown", confidence: 0 };
    let reportResult = { disease: "Unknown", confidence: 0 };

    // ── Step 1: Handle file upload (OCR + Report Prediction) ──
    if (req.file) {
      console.log(`[Server] 📸 File received: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)} KB)`);

      // Run OCR to extract text from the image
      const ocrResult = await extractTextFromImage(req.file.path);
      extractedText = ocrResult.text;

      console.log(`[Server] 📝 Extracted text preview: "${extractedText.slice(0, 100)}..."`);

      // Run report-based prediction on the extracted text
      if (extractedText.length > 0) {
        reportResult = predictFromReport(extractedText);
      }

      // Clean up the uploaded file after processing (after 60s)
      setTimeout(() => {
        fs.unlink(req.file.path, (err) => {
          if (err) console.log(`[Server] ⚠️  Failed to delete file: ${err.message}`);
          else console.log(`[Server] 🗑️  Cleaned up: ${req.file.filename}`);
        });
      }, 60000);
    } else {
      console.log("[Server] ℹ️  No file uploaded");
    }

    // ── Step 2: Handle symptoms ──
    let parsedSymptoms = [];
    if (req.body.symptoms) {
      try {
        parsedSymptoms = JSON.parse(req.body.symptoms);
        if (!Array.isArray(parsedSymptoms)) {
          throw new Error("symptoms must be a JSON array");
        }
      } catch (parseErr) {
        console.error(`[Server] ❌ Failed to parse symptoms: ${parseErr.message}`);
        return res.status(400).json({
          error: "Invalid symptoms format. Must be a JSON string array, e.g. '[\"fever\",\"cough\"]'",
        });
      }

      console.log(`[Server] 🩺 Symptoms received: [${parsedSymptoms.join(", ")}]`);
      symptomResult = predictFromSymptoms(parsedSymptoms);
    } else {
      console.log("[Server] ℹ️  No symptoms provided");
    }

    // ── Step 3: Validate at least one input ──
    if (!req.file && parsedSymptoms.length === 0) {
      console.log("[Server] ⚠️  No input provided at all");
      return res.status(400).json({
        error: "Please provide at least a file (image) or symptoms.",
      });
    }

    // ── Step 4: Merge predictions ──
    const merged = mergePredictions(symptomResult, reportResult);

    // ── Step 5: Build and send response ──
    const response = {
      disease: merged.disease,
      confidence: merged.confidence,
      extractedText: extractedText,
      source: merged.source,
    };

    console.log("\n[Server] ✅ Final response:");
    console.log(JSON.stringify(response, null, 2));
    console.log("========================================\n");

    return res.json(response);
  } catch (error) {
    console.error(`[Server] ❌ Prediction failed: ${error.message}`);
    return res.status(500).json({
      error: "Internal server error during prediction",
      details: error.message,
    });
  }
});

// ============================================================
// Multer Error Handler
// ============================================================
app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    console.error(`[Server] ❌ Multer error: ${err.message}`);
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ error: "File too large. Maximum size is 10 MB." });
    }
    return res.status(400).json({ error: err.message });
  }
  if (err) {
    console.error(`[Server] ❌ Error: ${err.message}`);
    return res.status(400).json({ error: err.message });
  }
});

// ============================================================
// Start Server
// ============================================================
app.listen(PORT, () => {
  console.log(`
  ╔══════════════════════════════════════════════╗
  ║   🏥  MediScan AI Backend                    ║
  ║   🌐  http://localhost:${PORT}                 ║
  ║   📡  POST /api/predict                      ║
  ║   💚  GET  /api/health                       ║
  ╚══════════════════════════════════════════════╝
  `);
});
