require("dotenv").config();
const express = require("express");
const http = require("http");
const cors = require("cors");
const axios = require("axios");
const winston = require("winston");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");
const mongoose = require("mongoose");
const jwt = require("jsonwebtoken");
const Tesseract = require("tesseract.js");
const { Server: SocketServer } = require("socket.io");
const { extractText } = require("./fileParserService");

// Models
const User = require("./models/User");
const Disease = require("./models/Disease");
const PredictionHistory = require("./models/PredictionHistory");

// Middleware
const { authenticate, requireAuth, requireAdmin } = require("./middleware/auth");

const app = express();
const PORT = process.env.PORT || 5001;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:5002";
const JWT_SECRET = process.env.JWT_SECRET || "mediscan_secret";

// ============================================================
// Logger Setup (Winston)
// ============================================================
const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
      const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : "";
      return `${timestamp} [${level.toUpperCase()}] ${message}${metaStr}`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({
      filename: "logs/predictions.log",
      maxsize: 5 * 1024 * 1024,
      maxFiles: 3,
    }),
  ],
});

// ============================================================
// Multer Setup for Dataset Upload
// ============================================================
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = path.join(__dirname, "..", "ml-service", "data");
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, "disease_dataset.csv");
  },
});
const upload = multer({ storage });

// Medical image upload config
const medicalUploadDir = path.join(__dirname, "uploads");
if (!fs.existsSync(medicalUploadDir)) fs.mkdirSync(medicalUploadDir, { recursive: true });

const medicalStorage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, medicalUploadDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, `med_${Date.now()}_${Math.random().toString(36).slice(2, 8)}${ext}`);
  },
});
const medicalUpload = multer({
  storage: medicalStorage,
  limits: { fileSize: 15 * 1024 * 1024 }, // 15MB
  fileFilter: (req, file, cb) => {
    const allowedExts = /jpeg|jpg|png|gif|bmp|webp|pdf|doc|docx|txt|csv/;
    const allowedMimes = [
      "image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp", "image/webp",
      "application/pdf",
      "application/msword",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain", "text/csv",
    ];
    const extOk = allowedExts.test(path.extname(file.originalname).toLowerCase());
    const mimeOk = allowedMimes.includes(file.mimetype);
    const ok = extOk || mimeOk;
    cb(ok ? null : new Error("Unsupported file type. Allowed: images, PDF, DOC, DOCX, TXT, CSV"), ok);
  },
});

// ============================================================
// OCR & Report Analysis Engine
// ============================================================

// Medical keyword rules for report-based prediction
const REPORT_RULES = [
  {
    keywords: ["glucose", "sugar", "hba1c", "fasting glucose", "blood sugar", "diabetic", "insulin"],
    highIndicators: ["high", "elevated", "above normal", "increased", "positive"],
    disease: "Diabetes Symptoms",
    severity: "high",
    precautions: ["Monitor blood sugar regularly", "Follow a low-sugar diet", "Exercise daily", "Take prescribed medication"],
    medicines: ["Metformin", "Glimepiride", "Insulin", "Sitagliptin"],
  },
  {
    keywords: ["blood pressure", "bp", "systolic", "diastolic", "hypertension", "mmhg"],
    highIndicators: ["high", "elevated", "above normal", "stage 1", "stage 2"],
    disease: "Hypertension Alert",
    severity: "high",
    precautions: ["Monitor BP regularly", "Reduce salt intake", "Avoid stress", "Exercise regularly"],
    medicines: ["Amlodipine", "Losartan", "Metoprolol", "Hydrochlorothiazide"],
  },
  {
    keywords: ["hemoglobin", "hb", "rbc", "iron", "ferritin", "anemia", "anaemia"],
    highIndicators: ["low", "decreased", "below normal", "deficiency"],
    disease: "Anemia (Iron Deficiency)",
    severity: "moderate",
    precautions: ["Iron-rich diet", "Vitamin C for absorption", "Regular blood tests", "Avoid tea with meals"],
    medicines: ["Ferrous Sulfate", "Folic Acid", "Vitamin B12", "Iron Supplements"],
  },
  {
    keywords: ["cholesterol", "ldl", "hdl", "triglyceride", "lipid profile", "lipid"],
    highIndicators: ["high", "elevated", "above normal", "borderline high"],
    disease: "High Cholesterol",
    severity: "moderate",
    precautions: ["Low-fat diet", "Regular exercise", "Avoid processed food", "Regular lipid testing"],
    medicines: ["Atorvastatin", "Rosuvastatin", "Fenofibrate", "Omega-3"],
  },
  {
    keywords: ["thyroid", "tsh", "t3", "t4", "thyroxine"],
    highIndicators: ["high", "low", "abnormal", "elevated", "decreased"],
    disease: "Thyroid Disorder",
    severity: "moderate",
    precautions: ["Regular thyroid monitoring", "Medication compliance", "Avoid soy in excess", "Iodine-balanced diet"],
    medicines: ["Levothyroxine", "Methimazole", "Propylthiouracil"],
  },
  {
    keywords: ["uric acid", "gout", "urate"],
    highIndicators: ["high", "elevated", "above normal"],
    disease: "Gout / High Uric Acid",
    severity: "moderate",
    precautions: ["Avoid red meat", "Stay hydrated", "Limit alcohol", "Low-purine diet"],
    medicines: ["Allopurinol", "Colchicine", "Febuxostat", "Naproxen"],
  },
  {
    keywords: ["creatinine", "bun", "kidney", "renal", "gfr", "urea"],
    highIndicators: ["high", "elevated", "abnormal", "decreased gfr"],
    disease: "Kidney Function Concern",
    severity: "high",
    precautions: ["Consult nephrologist", "Limit protein intake", "Stay hydrated", "Monitor regularly"],
    medicines: ["Consult specialist for prescription"],
  },
  {
    keywords: ["wbc", "white blood cell", "neutrophil", "lymphocyte", "infection", "leukocyte"],
    highIndicators: ["high", "elevated", "increased"],
    disease: "Infection / Inflammatory Response",
    severity: "moderate",
    precautions: ["Complete antibiotics course", "Rest", "Stay hydrated", "Follow-up blood test"],
    medicines: ["Amoxicillin", "Azithromycin", "Paracetamol", "Probiotics"],
  },
];

// Skin condition keywords for image filenames
const SKIN_RULES = [
  { keywords: ["acne", "pimple", "zit"], disease: "Acne Vulgaris", severity: "low", precautions: ["Gentle cleansing", "Avoid touching face", "Non-comedogenic products", "Dermatologist consult"], medicines: ["Benzoyl Peroxide", "Adapalene", "Clindamycin gel", "Salicylic Acid"] },
  { keywords: ["rash", "eczema", "dermatitis"], disease: "Dermatitis / Eczema", severity: "moderate", precautions: ["Moisturize regularly", "Avoid irritants", "Cotton clothing", "Dermatologist consult"], medicines: ["Hydrocortisone", "Cetrizine", "Calamine lotion", "Emollients"] },
  { keywords: ["fungal", "ringworm", "tinea"], disease: "Fungal Skin Infection", severity: "low", precautions: ["Keep area dry", "Antifungal cream", "Avoid sharing towels", "Wear loose clothing"], medicines: ["Clotrimazole", "Terbinafine", "Ketoconazole", "Fluconazole"] },
  { keywords: ["psoriasis"], disease: "Psoriasis", severity: "moderate", precautions: ["Moisturize", "Sunlight exposure", "Avoid triggers", "Dermatologist consult"], medicines: ["Clobetasol", "Methotrexate", "Calcipotriol", "Coal Tar"] },
];

async function runOCR(filePath) {
  try {
    logger.info("Starting OCR processing...", { filePath });
    const { data } = await Tesseract.recognize(filePath, "eng", {
      logger: (m) => { if (m.status === "recognizing text") logger.info(`OCR progress: ${Math.round(m.progress * 100)}%`); },
    });
    logger.info("OCR completed", { textLength: data.text.length, confidence: data.confidence });
    return { text: data.text, confidence: data.confidence };
  } catch (err) {
    logger.error("OCR failed:", err.message);
    return { text: "", confidence: 0 };
  }
}

function analyzeReportText(text) {
  if (!text || text.trim().length < 10) return null;
  const lower = text.toLowerCase();
  const results = [];

  for (const rule of REPORT_RULES) {
    let keywordScore = 0;
    let matchedKeywords = [];
    let hasIndicator = false;

    for (const kw of rule.keywords) {
      if (lower.includes(kw)) {
        keywordScore++;
        matchedKeywords.push(kw);
      }
    }

    if (keywordScore === 0) continue;

    for (const indicator of rule.highIndicators) {
      if (lower.includes(indicator)) {
        hasIndicator = true;
        break;
      }
    }

    const confidence = Math.min(Math.round((keywordScore / rule.keywords.length) * (hasIndicator ? 90 : 65)), 95);

    results.push({
      disease: rule.disease,
      confidence,
      type: "report",
      severity: rule.severity,
      precautions: rule.precautions,
      medicines: rule.medicines,
      matchedKeywords,
      hasIndicator,
      explanation: `Found medical keywords: ${matchedKeywords.join(", ")}${hasIndicator ? " with abnormal indicators" : " in report text"}.`,
    });
  }

  results.sort((a, b) => b.confidence - a.confidence);
  return results.length > 0 ? results : null;
}

function classifySkinImage(filename) {
  const lower = (filename || "").toLowerCase();
  for (const rule of SKIN_RULES) {
    if (rule.keywords.some((kw) => lower.includes(kw))) {
      return {
        disease: rule.disease,
        confidence: 60,
        type: "image",
        severity: rule.severity,
        precautions: rule.precautions,
        medicines: rule.medicines,
        explanation: `Image filename suggests a skin condition. Please consult a dermatologist for accurate diagnosis.`,
      };
    }
  }
  return null;
}

// ============================================================
// MongoDB Connection
// ============================================================
const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/mediscan";

async function connectDB() {
  try {
    await mongoose.connect(MONGO_URI);
    logger.info("✅ Connected to MongoDB");
    await seedDiseases();
  } catch (err) {
    logger.error("❌ MongoDB connection failed:", err.message);
    logger.info("⚠️  Running in fallback mode (in-memory data)");
  }
}

// ============================================================
// Seed Initial Disease Data
// ============================================================
const INITIAL_DISEASES = [
  { name: "Common Flu (Influenza)", symptoms: ["fever", "cough", "sore throat", "body ache", "fatigue", "chills", "headache", "runny nose"], precautions: ["Rest and stay hydrated", "Take OTC flu medication", "Avoid contact", "Cover mouth", "Consult doctor"], medicines: ["Paracetamol", "Ibuprofen", "Cetirizine", "Dextromethorphan", "Vitamin C"], severity: "moderate" },
  { name: "Common Cold", symptoms: ["runny nose", "sneezing", "sore throat", "mild cough", "congestion", "watery eyes"], precautions: ["Rest", "Warm fluids", "Saline drops", "Salt water gargle", "Wash hands"], medicines: ["Antihistamines", "Nasal decongestant", "Cough syrup", "Throat lozenges"], severity: "low" },
  { name: "COVID-19 (Suspected)", symptoms: ["fever", "dry cough", "loss of taste", "loss of smell", "shortness of breath", "fatigue", "body ache"], precautions: ["Isolate", "Monitor oxygen", "Stay hydrated", "Seek emergency help if breathing difficult"], medicines: ["Paracetamol", "Dolo 650", "ORS", "Zinc", "Vitamin D"], severity: "high" },
  { name: "Migraine", symptoms: ["headache", "nausea", "sensitivity to light", "blurred vision", "dizziness", "vomiting"], precautions: ["Dark room rest", "Cold compress", "Stay hydrated", "Avoid triggers", "Medication"], medicines: ["Sumatriptan", "Ibuprofen", "Aspirin", "Ergotamine"], severity: "moderate" },
  { name: "Gastroenteritis (Stomach Flu)", symptoms: ["nausea", "vomiting", "diarrhea", "stomach pain", "fever", "dehydration", "cramps"], precautions: ["Stay hydrated (ORS)", "Bland foods", "Avoid dairy", "Rest"], medicines: ["ORS", "Ondansetron", "Loperamide", "Probiotics"], severity: "moderate" },
  { name: "Allergic Rhinitis", symptoms: ["sneezing", "runny nose", "itchy eyes", "watery eyes", "congestion", "itchy throat"], precautions: ["Avoid allergens", "Antihistamines", "Closed windows", "Air purifiers"], medicines: ["Cetirizine", "Loratadine", "Fluticasone", "Montelukast"], severity: "low" },
  { name: "Bronchitis", symptoms: ["persistent cough", "mucus production", "chest discomfort", "fatigue", "shortness of breath", "mild fever"], precautions: ["Avoid smoke", "Use humidifier", "Drink fluids", "Rest"], medicines: ["Ambroxol", "Dextromethorphan", "Salbutamol", "Acetylcysteine"], severity: "moderate" },
  { name: "Pneumonia (Suspected)", symptoms: ["high fever", "severe cough", "chest pain", "shortness of breath", "fatigue", "chills", "sweating"], precautions: ["Immediate MD attention", "Antibiotics", "Rest", "Hydrate"], medicines: ["Amoxicillin", "Azithromycin", "Levofloxacin", "Paracetamol"], severity: "high" },
  { name: "Urinary Tract Infection (UTI)", symptoms: ["burning urination", "frequent urination", "lower abdominal pain", "cloudy urine", "fever", "back pain"], precautions: ["Plenty of water", "Antibiotics", "Avoid caffeine", "Hygiene"], medicines: ["Nitrofurantoin", "Ciprofloxacin", "Trimethoprim", "Cranberry extract"], severity: "moderate" },
  { name: "Dehydration", symptoms: ["thirst", "dry mouth", "dizziness", "fatigue", "dark urine", "headache", "dry skin"], precautions: ["Water/Electrolytes", "Avoid caffeine", "Rest in cool area"], medicines: ["ORS", "Electrolyte sachets", "Coconut water"], severity: "moderate" },
  { name: "Hypertension Alert", symptoms: ["headache", "dizziness", "blurred vision", "chest pain", "shortness of breath", "nosebleed"], precautions: ["Monitor BP", "Reduce salt", "Medication", "Avoid stress"], medicines: ["Amlodipine", "Losartan", "Metoprolol", "Hydrochlorothiazide"], severity: "high" },
  { name: "Diabetes Symptoms", symptoms: ["frequent urination", "excessive thirst", "blurred vision", "fatigue", "slow healing wounds", "weight loss"], precautions: ["Test sugar", "Low-sugar diet", "Exercise", "Monitor glucose"], medicines: ["Metformin", "Glimepiride", "Insulin", "Sitagliptin"], severity: "high" },
  { name: "Anxiety Disorder", symptoms: ["restlessness", "rapid heartbeat", "sweating", "trembling", "difficulty concentrating", "insomnia", "irritability"], precautions: ["Deep breathing", "Exercise", "Limit caffeine", "Regular sleep"], medicines: ["Escitalopram", "Sertraline", "Propranolol", "Buspirone"], severity: "moderate" },
  { name: "Food Poisoning", symptoms: ["nausea", "vomiting", "diarrhea", "stomach cramps", "fever", "weakness"], precautions: ["ORS", "Avoid solid food", "Rest", "Gradual bland food"], medicines: ["ORS", "Ondansetron", "Loperamide", "Probiotics"], severity: "moderate" },
  { name: "Conjunctivitis (Pink Eye)", symptoms: ["red eyes", "itchy eyes", "watery eyes", "eye discharge", "swollen eyelids", "sensitivity to light"], precautions: ["Don't touch eyes", "Wash hands", "Antibiotic drops", "Warm compress"], medicines: ["Moxifloxacin", "Tobramycin", "Artificial tears", "Olopatadine"], severity: "low" },
];

async function seedDiseases() {
  try {
    const count = await Disease.countDocuments();
    if (count === 0) {
      await Disease.insertMany(INITIAL_DISEASES);
      logger.info(`🌱 Seeded ${INITIAL_DISEASES.length} diseases into MongoDB`);
    } else {
      logger.info(`📊 ${count} diseases already in database`);
    }
  } catch (err) {
    logger.error("Seeding failed:", err.message);
  }
}

// ============================================================
// Middleware
// ============================================================
app.use(cors());
app.use(express.json());
app.use(authenticate); // Soft auth on all routes

// Request Logging
app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    logger.info("HTTP Request", {
      method: req.method,
      path: req.path,
      status: res.statusCode,
      latency_ms: Date.now() - start,
    });
  });
  next();
});

// ============================================================
// Helper: Generate JWT
// ============================================================
function generateToken(user) {
  return jwt.sign(
    { id: user._id, role: user.role },
    JWT_SECRET,
    { expiresIn: "7d" }
  );
}

// ============================================================
// Helper: Fallback Prediction (when MongoDB or ML is offline)
// Uses weighted F1 scoring for accurate disease differentiation
// ============================================================
async function fallbackPredict(userSymptoms) {
  const normalized = [...new Set(userSymptoms.map((s) => s.toLowerCase().trim().replace(/[_\-]/g, " ")).filter(Boolean))];
  if (normalized.length === 0) return null;

  let diseases;
  try {
    diseases = await Disease.find().lean();
  } catch {
    diseases = INITIAL_DISEASES.map((d, i) => ({ ...d, _id: `fallback_${i}` }));
  }

  // Pre-compute how common each symptom is across all diseases (rarity weight)
  const symptomFrequency = {};
  diseases.forEach((d) => {
    (d.symptoms || []).forEach((s) => {
      const key = s.toLowerCase().trim();
      symptomFrequency[key] = (symptomFrequency[key] || 0) + 1;
    });
  });
  const totalDiseases = diseases.length || 1;

  // Check if symptom A matches symptom B (exact word match, not substring)
  function symptomsMatch(userSym, diseaseSym) {
    const u = userSym.toLowerCase().trim();
    const d = diseaseSym.toLowerCase().trim();
    if (u === d) return true;
    // Allow matching with underscores/hyphens replaced by spaces
    const uNorm = u.replace(/[_\-]/g, " ");
    const dNorm = d.replace(/[_\-]/g, " ");
    if (uNorm === dNorm) return true;
    // Allow partial word match only if one is fully contained AND length is similar
    if (uNorm.length >= 4 && dNorm.length >= 4) {
      if (dNorm.includes(uNorm) && uNorm.length >= dNorm.length * 0.5) return true;
      if (uNorm.includes(dNorm) && dNorm.length >= uNorm.length * 0.5) return true;
    }
    return false;
  }

  const scored = diseases.map((entry) => {
    const diseaseSymptoms = (entry.symptoms || []).map((s) => s.toLowerCase().trim());

    // Find which user symptoms match this disease
    let matchedUserSymptoms = 0;
    let matchedDiseaseSymptoms = 0;
    let rarityBonus = 0;
    const matchedNames = [];

    normalized.forEach((userSym) => {
      const found = diseaseSymptoms.find((ds) => symptomsMatch(userSym, ds));
      if (found) {
        matchedUserSymptoms++;
        matchedNames.push(userSym);
        // Rarer symptoms get higher weight (inverse document frequency)
        const freq = symptomFrequency[found] || 1;
        rarityBonus += (1 - freq / totalDiseases);
      }
    });

    // Count how many disease symptoms were matched by user input
    diseaseSymptoms.forEach((ds) => {
      if (normalized.some((userSym) => symptomsMatch(userSym, ds))) {
        matchedDiseaseSymptoms++;
      }
    });

    if (matchedUserSymptoms === 0) return null;

    // Precision: What fraction of user's symptoms match this disease
    const precision = matchedUserSymptoms / normalized.length;
    // Recall: What fraction of disease's symptoms were mentioned by user
    const recall = matchedDiseaseSymptoms / (diseaseSymptoms.length || 1);
    // F1-like combined score (weighted: recall matters more for specificity)
    const f1 = precision * 0.4 + recall * 0.4 + (rarityBonus / normalized.length) * 0.2;

    return {
      entry,
      matchedUserSymptoms,
      matchedDiseaseSymptoms,
      totalDiseaseSymptoms: diseaseSymptoms.length,
      precision,
      recall,
      score: f1,
      matchedNames,
    };
  }).filter(Boolean);

  if (scored.length === 0) return null;

  // Sort by composite score (higher is better)
  scored.sort((a, b) => b.score - a.score);

  const best = scored[0];
  const conf = Math.min(Math.round(best.score * 100), 95);

  const top3 = scored.slice(0, 3).map((s) => ({
    disease: s.entry.name,
    confidence: Math.min(Math.round(s.score * 100), 95),
  }));

  const matchPct = Math.round(best.precision * 100);
  const coveragePct = Math.round(best.recall * 100);

  return {
    disease: best.entry.name,
    confidence: conf,
    top_3: top3,
    precautions: best.entry.precautions,
    medicines: best.entry.medicines,
    severity: best.entry.severity,
    source: "fallback",
    explanation: `${matchPct}% of your symptoms match ${best.entry.name} (covering ${coveragePct}% of its known symptoms: ${best.matchedNames.join(", ")}).`,
  };
}

// ============================================================
// AUTH ROUTES
// ============================================================

// Register
app.post("/api/auth/register", async (req, res) => {
  try {
    const { name, email, password } = req.body;
    if (!name || !email || !password) {
      return res.status(400).json({ error: "Name, email, and password are required" });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters" });
    }

    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({ error: "Email already registered" });
    }

    const user = await User.create({ name, email: email.toLowerCase(), password });
    const token = generateToken(user);

    logger.info("New user registered", { userId: user._id, email: user.email });

    res.status(201).json({
      token,
      user: user.toJSON(),
    });
  } catch (err) {
    logger.error("Registration failed:", err.message);
    res.status(500).json({ error: "Registration failed" });
  }
});

// Login
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required" });
    }

    const user = await User.findOne({ email: email.toLowerCase() });
    if (!user) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ error: "Invalid email or password" });
    }

    const token = generateToken(user);
    logger.info("User logged in", { userId: user._id });

    res.json({ token, user: user.toJSON() });
  } catch (err) {
    logger.error("Login failed:", err.message);
    res.status(500).json({ error: "Login failed" });
  }
});

// Get current user profile
app.get("/api/auth/me", requireAuth, async (req, res) => {
  try {
    const user = await User.findById(req.user._id);
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json(user.toJSON());
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch profile" });
  }
});

// Update profile
app.put("/api/auth/me", requireAuth, async (req, res) => {
  try {
    const { name, avatar } = req.body;
    const user = await User.findByIdAndUpdate(
      req.user._id,
      { name, avatar },
      { new: true }
    );
    res.json(user.toJSON());
  } catch (err) {
    res.status(500).json({ error: "Failed to update profile" });
  }
});

// ============================================================
// HEALTH & INFO ROUTES
// ============================================================

app.get("/api/health", async (_req, res) => {
  let mlStatus = "unavailable";
  try {
    const mlResponse = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 2000 });
    mlStatus = mlResponse.data?.status === "ok" ? "healthy" : "degraded";
  } catch {}
  
  const dbStatus = mongoose.connection.readyState === 1 ? "connected" : "disconnected";
  
  res.json({ status: "ok", ml_service: mlStatus, database: dbStatus });
});

app.get("/api/model-info", async (_req, res) => {
  try {
    const mlResponse = await axios.get(`${ML_SERVICE_URL}/model-info`, { timeout: 3000 });
    res.json(mlResponse.data);
  } catch {
    const count = await Disease.countDocuments().catch(() => INITIAL_DISEASES.length);
    res.json({
      disease_count: count,
      source: "fallback",
    });
  }
});

app.get("/api/symptoms", async (_req, res) => {
  try {
    const diseases = await Disease.find();
    const allSymptoms = new Set();
    diseases.forEach((d) => d.symptoms.forEach((s) => allSymptoms.add(s)));
    res.json({ symptoms: Array.from(allSymptoms).sort() });
  } catch {
    const allSymptoms = new Set();
    INITIAL_DISEASES.forEach((d) => d.symptoms.forEach((s) => allSymptoms.add(s)));
    res.json({ symptoms: Array.from(allSymptoms).sort() });
  }
});

// ============================================================
// UPLOAD & OCR ROUTES
// ============================================================

app.post("/api/upload", medicalUpload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded" });

    const filePath = req.file.path;
    const fileType = req.file.mimetype;
    const result = {
      fileName: req.file.filename,
      filePath: `/uploads/${req.file.filename}`,
      fileType: fileType,
      originalName: req.file.originalname,
      size: req.file.size,
      extractedText: "",
      ocrConfidence: 0,
      extractionMethod: "",
      reportAnalysis: null,
      skinAnalysis: null,
    };

    // Use unified file parser for text extraction
    const extraction = await extractText(filePath, fileType, req.file.originalname);
    result.extractedText = extraction.text || "";
    result.ocrConfidence = extraction.confidence || 0;
    result.extractionMethod = extraction.method || "unknown";

    // Analyze extracted text for medical keywords
    if (result.extractedText.trim().length > 20) {
      const reportResults = analyzeReportText(result.extractedText);
      if (reportResults) {
        result.reportAnalysis = reportResults;
      }
    }

    // Check for skin condition by filename (images only)
    if (fileType.startsWith("image/")) {
      const skinResult = classifySkinImage(req.file.originalname);
      if (skinResult) {
        result.skinAnalysis = skinResult;
      }
    }

    // Clean up file after processing
    setTimeout(() => {
      fs.unlink(filePath, () => {});
    }, 120000);

    logger.info("File uploaded and processed", {
      filename: result.fileName,
      method: result.extractionMethod,
      textLength: result.extractedText.length,
      confidence: result.ocrConfidence,
    });
    res.json(result);
  } catch (err) {
    logger.error("Upload failed:", err.message);
    res.status(500).json({ error: "File processing failed: " + err.message });
  }
});

// Report text analysis endpoint (for pasted text)
app.post("/api/analyze-report", async (req, res) => {
  const { text } = req.body;
  if (!text || text.trim().length < 10) {
    return res.status(400).json({ error: "Please provide report text (at least 10 characters)" });
  }
  const results = analyzeReportText(text);
  if (results) {
    res.json({ analysis: results, source: "report" });
  } else {
    res.json({ analysis: [], source: "report", message: "No medical conditions detected in the text." });
  }
});

// ============================================================
// PREDICTION ROUTES (Multi-logic: symptoms + report + image)
// ============================================================

app.post("/api/predict", async (req, res) => {
  const { symptoms, reportText, reportAnalysis, imageAnalysis } = req.body;

  // Need at least one input source
  const hasSymptoms = symptoms && Array.isArray(symptoms) && symptoms.length > 0;
  const hasReport = reportText && reportText.trim().length > 10;
  const hasReportAnalysis = reportAnalysis && reportAnalysis.length > 0;
  const hasImageAnalysis = imageAnalysis && imageAnalysis.disease;

  if (!hasSymptoms && !hasReport && !hasReportAnalysis && !hasImageAnalysis) {
    return res.status(400).json({ error: "Please provide symptoms, a report, or an image" });
  }

  let finalResult = null;
  const allResults = [];

  // 1. Report-based analysis
  if (hasReport) {
    const reportResults = analyzeReportText(reportText);
    if (reportResults) {
      reportResults.forEach((r) => allResults.push({ ...r, source: "report" }));
    }
  }
  if (hasReportAnalysis) {
    reportAnalysis.forEach((r) => allResults.push({ ...r, source: "report" }));
  }

  // 2. Image-based analysis
  if (hasImageAnalysis) {
    allResults.push({ ...imageAnalysis, source: "image" });
  }

  // 3. Symptom-based analysis
  if (hasSymptoms) {
    const sanitized = [...new Set(symptoms.map((s) => s.trim().toLowerCase()).filter(Boolean))].slice(0, 20);

    try {
      const mlResponse = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        { symptoms: sanitized },
        { timeout: 10000 }
      );
      const prediction = mlResponse.data;
      let diseaseData = null;
      try { diseaseData = await Disease.findOne({ name: prediction.disease }); } catch {}
      allResults.push({
        ...prediction,
        type: "symptom",
        source: "ml_model",
        precautions: diseaseData?.precautions || prediction.precautions || [],
        medicines: diseaseData?.medicines || prediction.medicines || [],
        severity: diseaseData?.severity || prediction.severity || "moderate",
      });
    } catch {
      const fallback = await fallbackPredict(sanitized);
      if (fallback) {
        allResults.push({ ...fallback, type: "symptom", source: "fallback" });
      }
    }
  }

  if (allResults.length === 0) {
    return res.json({
      disease: "No Specific Condition Detected",
      confidence: 0,
      type: "unknown",
      source: "analyzer",
      top_3: [],
      precautions: ["Consult a physician for proper medical advice.", "Rest and monitor your symptoms."],
      medicines: [],
      severity: "low",
      explanation: "No specific medical conditions were identified from the provided input data.",
      allResults: []
    });
  }

  // Pick the best result (highest confidence)
  allResults.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  finalResult = allResults[0];

  // If multiple sources agree, boost confidence
  const agreeing = allResults.filter((r) => r.disease === finalResult.disease);
  if (agreeing.length > 1) {
    finalResult.confidence = Math.min(finalResult.confidence + 10, 98);
    finalResult.explanation = (finalResult.explanation || "") + ` Confirmed by ${agreeing.length} analysis methods.`;
  }

  // Aggregate top 3 from all results
  const uniqueDiseases = new Map();
  allResults.forEach((r) => {
    if (!uniqueDiseases.has(r.disease) || uniqueDiseases.get(r.disease).confidence < r.confidence) {
      uniqueDiseases.set(r.disease, r);
    }
  });
  const top3 = Array.from(uniqueDiseases.values())
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3)
    .map((r) => ({ disease: r.disease, confidence: r.confidence }));

  const response = {
    disease: finalResult.disease,
    confidence: finalResult.confidence,
    type: finalResult.type || "symptom",
    source: finalResult.source || "fallback",
    top_3: top3.length > 0 ? top3 : (finalResult.top_3 || []),
    precautions: finalResult.precautions || [],
    medicines: finalResult.medicines || [],
    severity: finalResult.severity || "moderate",
    explanation: finalResult.explanation || "",
    allResults: allResults.slice(0, 5),
  };

  // Emit real-time prediction via WebSocket
  if (global.io) {
    global.io.emit("new_prediction", {
      disease: response.disease,
      confidence: response.confidence,
      severity: response.severity,
      source: response.source,
      timestamp: Date.now(),
    });
    logger.info("📡 WebSocket: emitted new_prediction");
  }

  // Save to history
  try {
    await PredictionHistory.create({
      userId: req.user?._id || null,
      symptoms: hasSymptoms ? symptoms.map((s) => s.trim().toLowerCase()) : [],
      prediction: response.disease,
      confidence: response.confidence,
      top3: response.top_3,
      precautions: response.precautions,
      medicines: response.medicines,
      severity: response.severity,
      source: response.source,
      explanation: response.explanation,
    });
  } catch (err) {
    logger.error("Failed to save prediction history:", err.message);
  }

  res.json(response);
});

// ============================================================
// HISTORY ROUTES
// ============================================================

// User's own history
app.get("/api/history", async (req, res) => {
  try {
    const query = req.user ? { userId: req.user._id } : {};
    const predictions = await PredictionHistory.find(query)
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();
    res.json({ count: predictions.length, predictions });
  } catch (err) {
    res.json({ count: 0, predictions: [] });
  }
});

// All history (admin)
app.get("/api/admin/history", requireAuth, requireAdmin, async (req, res) => {
  try {
    const predictions = await PredictionHistory.find()
      .populate("userId", "name email")
      .sort({ createdAt: -1 })
      .limit(100)
      .lean();
    res.json({ count: predictions.length, predictions });
  } catch {
    res.json({ count: 0, predictions: [] });
  }
});

// ============================================================
// ADMIN: DISEASE MANAGEMENT
// ============================================================

app.get("/api/admin/diseases", async (req, res) => {
  try {
    const diseases = await Disease.find().sort({ createdAt: -1 }).lean();
    res.json(diseases);
  } catch {
    res.json(INITIAL_DISEASES);
  }
});

app.post("/api/admin/diseases", async (req, res) => {
  try {
    const { name, symptoms, precautions, medicines, severity, _id } = req.body;
    if (!name || !symptoms) {
      return res.status(400).json({ error: "Name and symptoms are required" });
    }

    const symptomsArr = Array.isArray(symptoms)
      ? symptoms.map((s) => s.trim().toLowerCase()).filter(Boolean)
      : symptoms.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);

    const precautionsArr = Array.isArray(precautions)
      ? precautions
      : (precautions || "").split(",").map((p) => p.trim()).filter(Boolean);

    const medicinesArr = Array.isArray(medicines)
      ? medicines
      : (medicines || "").split(",").map((m) => m.trim()).filter(Boolean);

    if (_id) {
      // Update existing
      const updated = await Disease.findByIdAndUpdate(
        _id,
        { name, symptoms: symptomsArr, precautions: precautionsArr, medicines: medicinesArr, severity: severity || "moderate" },
        { new: true }
      );
      if (!updated) return res.status(404).json({ error: "Disease not found" });
      return res.json({ success: true, disease: updated });
    }

    // Create new
    const disease = await Disease.create({
      name,
      symptoms: symptomsArr,
      precautions: precautionsArr,
      medicines: medicinesArr,
      severity: severity || "moderate",
    });

    res.status(201).json({ success: true, disease });
  } catch (err) {
    if (err.code === 11000) {
      return res.status(400).json({ error: "Disease with this name already exists" });
    }
    res.status(500).json({ error: "Failed to save disease" });
  }
});

app.delete("/api/admin/diseases/:id", async (req, res) => {
  try {
    await Disease.findByIdAndDelete(req.params.id);
    res.json({ success: true });
  } catch {
    res.status(500).json({ error: "Failed to delete disease" });
  }
});

// ============================================================
// ADMIN: USER MANAGEMENT
// ============================================================

app.get("/api/admin/users", requireAuth, requireAdmin, async (req, res) => {
  try {
    const users = await User.find().select("-password").sort({ createdAt: -1 }).lean();
    res.json(users);
  } catch {
    res.json([]);
  }
});

app.put("/api/admin/users/:id/role", requireAuth, requireAdmin, async (req, res) => {
  try {
    const { role } = req.body;
    if (!["user", "admin"].includes(role)) {
      return res.status(400).json({ error: "Invalid role" });
    }
    const user = await User.findByIdAndUpdate(req.params.id, { role }, { new: true });
    res.json(user.toJSON());
  } catch {
    res.status(500).json({ error: "Failed to update user role" });
  }
});

app.delete("/api/admin/users/:id", requireAuth, requireAdmin, async (req, res) => {
  try {
    await User.findByIdAndDelete(req.params.id);
    res.json({ success: true });
  } catch {
    res.status(500).json({ error: "Failed to delete user" });
  }
});

// ============================================================
// ADMIN: STATS
// ============================================================

app.get("/api/admin/stats", async (req, res) => {
  try {
    const [totalUsers, totalDiseases, totalPredictions, recentPredictions] = await Promise.all([
      User.countDocuments().catch(() => 0),
      Disease.countDocuments().catch(() => 0),
      PredictionHistory.countDocuments().catch(() => 0),
      PredictionHistory.find()
        .sort({ createdAt: -1 })
        .limit(7)
        .lean()
        .catch(() => []),
    ]);

    // Get predictions per day for last 7 days
    const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    let dailyPredictions = [];
    try {
      dailyPredictions = await PredictionHistory.aggregate([
        { $match: { createdAt: { $gte: sevenDaysAgo } } },
        {
          $group: {
            _id: { $dateToString: { format: "%Y-%m-%d", date: "$createdAt" } },
            count: { $sum: 1 },
            avgConfidence: { $avg: "$confidence" },
          },
        },
        { $sort: { _id: 1 } },
      ]);
    } catch {}

    res.json({
      totalUsers,
      totalDiseases,
      totalPredictions,
      dailyPredictions,
      mlStatus: "operational",
    });
  } catch {
    res.json({
      totalUsers: 0,
      totalDiseases: 0,
      totalPredictions: 0,
      dailyPredictions: [],
      mlStatus: "unknown",
    });
  }
});

// ============================================================
// DATASET & TRAINING
// ============================================================

app.post("/api/dataset/upload", upload.single("dataset"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });
  res.json({ success: true, filename: req.file.filename, size: req.file.size });
});

app.post("/api/model/train", (req, res) => {
  const script = path.join(__dirname, "..", "ml-service", "train_model.py");
  exec(`python3 "${script}"`, { timeout: 120000 }, (error, stdout, stderr) => {
    if (error) {
      logger.error("Training failed:", error.message);
      return res.status(500).json({ error: "Training failed", details: stderr });
    }
    logger.info("Model training completed");
    res.json({ success: true, output: stdout });
  });
});

// ============================================================
// CHATBOT (Static Knowledge Base)
// ============================================================

const HEALTH_KB = {
  greetings: [
    "Hello! 👋 I'm **MediScan AI Health Assistant**. How can I help you today?\n\nYou can ask me about:\n- 🩺 Common health conditions\n- 💊 Symptoms & medicines\n- 🥗 Diet & nutrition\n- 🏃 Exercise & fitness\n- 😴 Sleep & wellness",
    "Hi there! I'm your AI health companion. Ask me any health-related question and I'll do my best to help! 💙",
  ],
  fever: "## 🌡️ Fever\n\nFever is usually a sign that your body is fighting an infection.\n\n**Immediate Steps:**\n- Rest well and stay hydrated\n- Take **Paracetamol** (Dolo 650) if temperature exceeds 100°F\n- Use a cool, damp cloth on the forehead\n- Wear light clothing\n\n**⚠️ See a Doctor If:**\n- Fever persists beyond **3 days**\n- Temperature exceeds **103°F (39.4°C)**\n- Accompanied by severe headache, stiff neck, or rash\n\n> 💡 *Tip: Use our Prediction tool for a detailed symptom analysis!*",
  headache: "## 🤕 Headache\n\nHeadaches can be caused by stress, dehydration, eye strain, or lack of sleep.\n\n**Quick Relief:**\n- Rest in a dark, quiet room\n- Stay hydrated — drink water\n- Apply a cold or warm compress\n- Take **Ibuprofen** or **Paracetamol** if needed\n\n**Prevention:**\n- Manage stress with deep breathing\n- Maintain good posture\n- Take regular screen breaks\n- Get 7-8 hours of sleep\n\n**⚠️ Seek Immediate Help If:**\n- Sudden, severe \"worst headache ever\"\n- Accompanied by vision changes, numbness, or confusion",
  cold: "## 🤧 Common Cold\n\nMost colds resolve in **7-10 days**.\n\n**Home Remedies:**\n- Drink warm fluids (soup, herbal tea, warm water with honey)\n- Use saline nasal drops for congestion\n- Steam inhalation with eucalyptus\n- Rest and get adequate sleep\n\n**OTC Medications:**\n- **Antihistamines** (Cetirizine) for sneezing\n- **Nasal decongestants** for blocked nose\n- **Throat lozenges** for sore throat\n- **Vitamin C** to support immunity",
  cough: "## 😷 Cough\n\n**For Dry Cough:**\n- Honey with warm water or tea\n- Stay hydrated\n- Use a humidifier\n- **Dextromethorphan** cough syrup\n\n**For Productive (Wet) Cough:**\n- **Ambroxol** or **Guaifenesin** expectorants\n- Steam inhalation\n- Stay upright to help drainage\n\n**⚠️ See a Doctor If:**\n- Cough persists over **2 weeks**\n- Coughing up blood\n- Accompanied by shortness of breath or chest pain",
  stomach: "## 🤢 Stomach Issues\n\n**Immediate Relief:**\n- Follow the **BRAT diet** (Bananas, Rice, Applesauce, Toast)\n- Stay hydrated with **ORS** (Oral Rehydration Solution)\n- Avoid spicy, oily, and dairy foods\n- Ginger tea can help with nausea\n\n**Medications:**\n- **ORS sachets** for dehydration\n- **Ondansetron** for nausea/vomiting\n- **Probiotics** for gut health\n\n**⚠️ Seek Help If:**\n- Severe abdominal pain\n- Blood in stool or vomit\n- Signs of dehydration (dry mouth, dark urine)",
  sleep: "## 😴 Sleep & Insomnia\n\n**Sleep Hygiene Tips:**\n- Maintain a consistent sleep schedule (even weekends)\n- Avoid screens **1 hour** before bed\n- Keep your room cool (65-68°F) and dark\n- Limit caffeine after **2 PM**\n\n**Relaxation Techniques:**\n- **4-7-8 Breathing**: Inhale 4s, Hold 7s, Exhale 8s\n- Progressive muscle relaxation\n- Guided meditation apps\n\n**Natural Aids:**\n- Chamomile tea\n- Warm milk\n- Lavender aromatherapy\n\n> 💡 Adults need **7-9 hours** of quality sleep per night.",
  anxiety: "## 🧠 Anxiety & Stress\n\n**Immediate Coping:**\n- **4-7-8 Breathing**: Inhale 4s, Hold 7s, Exhale 8s\n- Grounding technique: Name 5 things you see, 4 you hear, 3 you touch\n- Take a walk outside\n\n**Long-term Management:**\n- Regular exercise (30 min/day)\n- Limit caffeine and alcohol\n- Maintain social connections\n- Practice mindfulness meditation\n- Keep a worry journal\n\n**⚠️ Seek Professional Help If:**\n- Anxiety interferes with daily life\n- Panic attacks occur frequently\n- You experience persistent worry for weeks",
  diet: "## 🥗 Diet & Nutrition\n\n**Balanced Plate:**\n- 🥬 **50%** Fruits & Vegetables\n- 🍚 **25%** Whole Grains\n- 🍗 **25%** Lean Protein\n- 🥑 Healthy fats (nuts, olive oil, avocado)\n\n**Daily Essentials:**\n- Drink **8 glasses of water** daily\n- Eat **5 servings** of fruits/vegetables\n- Include fiber-rich foods\n- Limit processed foods, sugar, and sodium\n\n**Superfoods to Include:**\n- Berries, spinach, nuts, yogurt, fish, turmeric, green tea",
  exercise: "## 🏃 Exercise & Fitness\n\n**Weekly Goals (WHO Recommendation):**\n- **150 minutes** moderate aerobic activity, OR\n- **75 minutes** vigorous activity\n- **2+ days** of strength training\n\n**Getting Started:**\n- Start with 15-20 min walks\n- Gradually increase intensity\n- Include stretching and warm-up\n- Find activities you enjoy!\n\n**Quick Home Workouts:**\n- Push-ups, squats, planks, lunges\n- Yoga or stretching routines\n- Dance workouts\n\n> 💡 Even **10 minutes** of exercise provides health benefits!",
  diabetes: "## 🩸 Diabetes\n\nDiabetes is a condition where blood sugar levels are chronically elevated.\n\n**Types:**\n- **Type 1**: Autoimmune, insulin-dependent\n- **Type 2**: Most common, linked to lifestyle\n- **Gestational**: During pregnancy\n\n**Warning Signs:**\n- Frequent urination, excessive thirst\n- Unexplained weight loss\n- Blurred vision, slow wound healing\n\n**Management:**\n- Monitor blood sugar regularly\n- Follow a low-sugar, high-fiber diet\n- Exercise 30 min daily\n- Take prescribed medications (Metformin, insulin)\n\n> 💡 *Upload your glucose report on our Predict page for AI analysis!*",
  blood_pressure: "## ❤️ Blood Pressure\n\n**Normal Ranges:**\n- Normal: < 120/80 mmHg\n- Elevated: 120-129 / < 80\n- High (Stage 1): 130-139 / 80-89\n- High (Stage 2): ≥ 140 / ≥ 90\n\n**To Lower BP:**\n- Reduce salt intake (< 2,300 mg/day)\n- Exercise regularly\n- Maintain healthy weight\n- Limit alcohol and quit smoking\n- Manage stress\n\n**Medications:**\n- Amlodipine, Losartan, Metoprolol\n- Always take as prescribed by your doctor",
  cholesterol: "## 🫀 Cholesterol\n\n**Understanding Levels:**\n- **Total Cholesterol**: < 200 mg/dL (desirable)\n- **LDL (bad)**: < 100 mg/dL\n- **HDL (good)**: > 60 mg/dL\n- **Triglycerides**: < 150 mg/dL\n\n**Heart-Healthy Tips:**\n- Eat more fiber, omega-3 fatty acids\n- Reduce saturated and trans fats\n- Exercise 30 min/day\n- Maintain healthy weight\n\n**Medications:**\n- Statins (Atorvastatin, Rosuvastatin) if prescribed\n\n> 💡 *Upload your lipid profile report for automated analysis!*",
  skin: "## 🧴 Skin Health\n\n**Common Conditions:**\n- **Acne**: Gentle cleansing, benzoyl peroxide, avoid touching face\n- **Eczema**: Moisturize regularly, avoid irritants, use hydrocortisone\n- **Fungal Infections**: Keep area dry, use antifungal cream\n\n**Skincare Basics:**\n- Cleanse, moisturize, apply sunscreen daily\n- Stay hydrated\n- Eat antioxidant-rich foods\n- Avoid harsh chemicals\n\n**⚠️ See a Dermatologist If:**\n- Persistent rash or skin changes\n- Moles that change shape/color\n- Severe or widespread skin issues",
  covid: "## 🦠 COVID-19\n\n**Common Symptoms:**\n- Fever, dry cough, fatigue\n- Loss of taste or smell\n- Shortness of breath, body aches\n\n**If You Suspect COVID:**\n- Isolate immediately\n- Get tested (RT-PCR or RAT)\n- Monitor oxygen levels (SpO2 > 94%)\n- Stay hydrated, rest\n\n**Medications:**\n- Paracetamol for fever\n- ORS for hydration\n- Zinc + Vitamin D supplements\n\n**⚠️ Emergency Signs:**\n- SpO2 drops below 92%\n- Difficulty breathing\n- Persistent chest pain",
  allergy: "## 🤧 Allergies\n\n**Common Types:**\n- Seasonal (pollen, dust)\n- Food allergies\n- Drug allergies\n- Skin allergies (contact dermatitis)\n\n**Management:**\n- Identify and avoid triggers\n- **Antihistamines** (Cetirizine, Loratadine)\n- Nasal sprays (Fluticasone)\n- Keep windows closed during high pollen\n\n**⚠️ Anaphylaxis Warning:**\n- Difficulty breathing, swelling of face/throat\n- Rapid heartbeat, dizziness\n- **Call emergency services immediately!**",
  vitamin: "## 💊 Vitamins & Supplements\n\n**Essential Vitamins:**\n- **Vitamin D**: Sunlight, fortified foods (bone health)\n- **Vitamin B12**: Meat, eggs, dairy (energy, nerves)\n- **Vitamin C**: Citrus fruits (immunity)\n- **Iron**: Leafy greens, meat (blood health)\n- **Omega-3**: Fish, walnuts (heart & brain)\n\n**When to Supplement:**\n- Vegetarian/vegan diets (B12, Iron)\n- Limited sun exposure (Vitamin D)\n- During pregnancy (Folic Acid, Iron)\n\n> Always consult your doctor before starting supplements.",
  default: "I'm not sure about that specific topic, but I'd love to help! 🤔\n\n**Here's what you can try:**\n- Rephrase your question with specific keywords\n- Ask about: fever, cold, headache, diabetes, diet, exercise, sleep, anxiety\n- Try our **Disease Prediction** tool for symptom-based analysis\n- Upload a medical report on the **Predict** page for AI analysis\n\n> ⚠️ For accurate medical advice, always consult a qualified healthcare professional.",
};

// Shared chatbot handler — matches message against knowledge base
function handleChatMessage(message) {
  const lower = message.toLowerCase();
  let response = HEALTH_KB.default;

  if (lower.match(/\b(hi|hello|hey|greetings|good morning|good evening)\b/)) {
    response = HEALTH_KB.greetings[Math.floor(Math.random() * HEALTH_KB.greetings.length)];
  } else if (lower.match(/\b(diabetes|diabetic|blood sugar|glucose|insulin|hba1c)\b/)) {
    response = HEALTH_KB.diabetes;
  } else if (lower.match(/\b(blood pressure|bp|hypertension|systolic|diastolic)\b/)) {
    response = HEALTH_KB.blood_pressure;
  } else if (lower.match(/\b(cholesterol|ldl|hdl|triglyceride|lipid)\b/)) {
    response = HEALTH_KB.cholesterol;
  } else if (lower.match(/\b(covid|corona|coronavirus|pandemic|omicron)\b/)) {
    response = HEALTH_KB.covid;
  } else if (lower.match(/\b(skin|acne|pimple|rash|eczema|dermatitis|fungal)\b/)) {
    response = HEALTH_KB.skin;
  } else if (lower.match(/\b(allergy|allergic|pollen|histamine|anaphylaxis)\b/)) {
    response = HEALTH_KB.allergy;
  } else if (lower.match(/\b(vitamin|supplement|b12|vitamin d|omega|iron deficiency)\b/)) {
    response = HEALTH_KB.vitamin;
  } else if (lower.match(/\b(fever|temperature|hot)\b/)) {
    response = HEALTH_KB.fever;
  } else if (lower.match(/\b(headache|head pain|migraine)\b/)) {
    response = HEALTH_KB.headache;
  } else if (lower.match(/\b(cold|runny nose|sneezing)\b/)) {
    response = HEALTH_KB.cold;
  } else if (lower.match(/\b(cough|coughing|throat)\b/)) {
    response = HEALTH_KB.cough;
  } else if (lower.match(/\b(stomach|digestion|diarrhea|nausea|vomiting)\b/)) {
    response = HEALTH_KB.stomach;
  } else if (lower.match(/\b(sleep|insomnia|tired)\b/)) {
    response = HEALTH_KB.sleep;
  } else if (lower.match(/\b(anxiety|anxious|stress|worried|panic|mental health|depression)\b/)) {
    response = HEALTH_KB.anxiety;
  } else if (lower.match(/\b(diet|nutrition|food|eat|weight loss|calories)\b/)) {
    response = HEALTH_KB.diet;
  } else if (lower.match(/\b(exercise|workout|fitness|gym|yoga|running)\b/)) {
    response = HEALTH_KB.exercise;
  }

  return { response, timestamp: new Date().toISOString() };
}

app.post("/api/chatbot", (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required" });
    res.json(handleChatMessage(message));
  } catch (err) {
    logger.error("Chatbot error:", err.message);
    res.status(500).json({ error: "Chatbot processing failed", response: HEALTH_KB.default, timestamp: new Date().toISOString() });
  }
});

// Alias: /api/chat → same handler
app.post("/api/chat", (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required" });
    res.json(handleChatMessage(message));
  } catch (err) {
    logger.error("Chat error:", err.message);
    res.status(500).json({ error: "Chat processing failed", response: HEALTH_KB.default, timestamp: new Date().toISOString() });
  }
});

// ============================================================
// START SERVER with Socket.io
// ============================================================
connectDB().then(() => {
  const server = http.createServer(app);
  const io = new SocketServer(server, {
    cors: { origin: "*", methods: ["GET", "POST"] },
  });
  global.io = io;

  io.on("connection", (socket) => {
    logger.info(`🔌 WebSocket client connected: ${socket.id}`);
    socket.on("disconnect", () => {
      logger.info(`🔌 WebSocket client disconnected: ${socket.id}`);
    });
  });

  server.listen(PORT, () => {
    logger.info(`🏥 MediScan AI Backend on port ${PORT}`);
    logger.info(`📡 WebSocket server ready on port ${PORT}`);
  });
});
