// ============================================================
// predictService.js — Disease Prediction Engine
// Contains 3 core functions:
//   1. predictFromSymptoms(symptoms)
//   2. predictFromReport(text)
//   3. mergePredictions(symptomResult, reportResult)
// ============================================================

// ----------------------------------------------------------
// Symptom → Disease mapping rules
// ----------------------------------------------------------
const SYMPTOM_RULES = [
  {
    symptoms: ["fever", "cough"],
    disease: "Flu",
    confidence: 0.85,
  },
  {
    symptoms: ["fever", "cough", "sore throat"],
    disease: "Flu",
    confidence: 0.9,
  },
  {
    symptoms: ["headache"],
    disease: "Migraine",
    confidence: 0.7,
  },
  {
    symptoms: ["headache", "nausea"],
    disease: "Migraine",
    confidence: 0.8,
  },
  {
    symptoms: ["fever", "body ache", "fatigue"],
    disease: "Viral Infection",
    confidence: 0.75,
  },
  {
    symptoms: ["sneezing", "runny nose"],
    disease: "Common Cold",
    confidence: 0.8,
  },
  {
    symptoms: ["nausea", "vomiting", "diarrhea"],
    disease: "Gastroenteritis",
    confidence: 0.8,
  },
  {
    symptoms: ["chest pain", "shortness of breath"],
    disease: "Cardiac Concern",
    confidence: 0.75,
  },
];

// ----------------------------------------------------------
// Report keyword → Disease mapping rules
// ----------------------------------------------------------
const REPORT_RULES = [
  {
    keywords: ["glucose", "sugar", "hba1c", "blood sugar", "diabetic"],
    disease: "Diabetes",
    confidence: 0.9,
  },
  {
    keywords: ["bp", "blood pressure", "hypertension", "systolic", "diastolic"],
    disease: "Hypertension",
    confidence: 0.85,
  },
  {
    keywords: ["hemoglobin", "hb", "anemia", "iron", "ferritin"],
    disease: "Anemia",
    confidence: 0.8,
  },
  {
    keywords: ["cholesterol", "ldl", "hdl", "triglyceride", "lipid"],
    disease: "High Cholesterol",
    confidence: 0.8,
  },
  {
    keywords: ["thyroid", "tsh", "t3", "t4"],
    disease: "Thyroid Disorder",
    confidence: 0.78,
  },
  {
    keywords: ["creatinine", "kidney", "renal", "gfr", "urea"],
    disease: "Kidney Concern",
    confidence: 0.82,
  },
];

// ============================================================
// 1. predictFromSymptoms(symptoms)
// ============================================================
/**
 * Predict a disease from an array of symptom strings.
 *
 * @param {string[]} symptoms — e.g. ["fever", "cough"]
 * @returns {{ disease: string, confidence: number }}
 */
function predictFromSymptoms(symptoms) {
  if (!symptoms || symptoms.length === 0) {
    console.log("[Predict] ⚠️  No symptoms provided");
    return { disease: "Unknown", confidence: 0 };
  }

  // Normalize input
  const normalized = symptoms.map((s) => s.toLowerCase().trim());
  console.log(`[Predict] 🩺 Analyzing symptoms: [${normalized.join(", ")}]`);

  let bestMatch = null;
  let bestScore = 0;

  for (const rule of SYMPTOM_RULES) {
    // Count how many of the rule's symptoms are present in the user input
    const matchCount = rule.symptoms.filter((s) => normalized.includes(s)).length;

    if (matchCount === 0) continue;

    // Score = (matched / total rule symptoms) * rule confidence
    const score = (matchCount / rule.symptoms.length) * rule.confidence;

    if (score > bestScore) {
      bestScore = score;
      bestMatch = {
        disease: rule.disease,
        confidence: parseFloat(score.toFixed(2)),
      };
    }
  }

  if (bestMatch) {
    console.log(`[Predict] ✅ Symptom result: ${bestMatch.disease} (${bestMatch.confidence})`);
    return bestMatch;
  }

  console.log("[Predict] ❓ No symptom match found");
  return { disease: "Unknown", confidence: 0 };
}

// ============================================================
// 2. predictFromReport(text)
// ============================================================
/**
 * Predict a disease from OCR-extracted report text using keyword detection.
 *
 * @param {string} text — Extracted text from a medical report (already lowercase)
 * @returns {{ disease: string, confidence: number }}
 */
function predictFromReport(text) {
  if (!text || text.trim().length === 0) {
    console.log("[Predict] ⚠️  No report text provided");
    return { disease: "Unknown", confidence: 0 };
  }

  const lower = text.toLowerCase();
  console.log(`[Predict] 📄 Analyzing report text (${lower.length} chars)...`);

  let bestMatch = null;
  let bestKeywordCount = 0;

  for (const rule of REPORT_RULES) {
    // Count how many keywords from the rule appear in the text
    const matchedKeywords = rule.keywords.filter((kw) => lower.includes(kw));

    if (matchedKeywords.length === 0) continue;

    // Prefer rules with more keyword matches; among ties, use rule confidence
    if (
      matchedKeywords.length > bestKeywordCount ||
      (matchedKeywords.length === bestKeywordCount &&
        (!bestMatch || rule.confidence > bestMatch.confidence))
    ) {
      bestKeywordCount = matchedKeywords.length;
      bestMatch = {
        disease: rule.disease,
        confidence: rule.confidence,
        matchedKeywords,
      };
    }
  }

  if (bestMatch) {
    console.log(
      `[Predict] ✅ Report result: ${bestMatch.disease} (${bestMatch.confidence}) — matched: [${bestMatch.matchedKeywords.join(", ")}]`
    );
    return {
      disease: bestMatch.disease,
      confidence: bestMatch.confidence,
    };
  }

  console.log("[Predict] ❓ No report keyword match found");
  return { disease: "Unknown", confidence: 0 };
}

// ============================================================
// 3. mergePredictions(symptomResult, reportResult)
// ============================================================
/**
 * Merge predictions from symptoms and report analysis.
 * Priority: report result wins if it's not "Unknown", else symptom result.
 * If both agree, confidence is boosted.
 *
 * @param {{ disease: string, confidence: number }} symptomResult
 * @param {{ disease: string, confidence: number }} reportResult
 * @returns {{ disease: string, confidence: number, source: string }}
 */
function mergePredictions(symptomResult, reportResult) {
  const hasReport = reportResult && reportResult.disease !== "Unknown";
  const hasSymptom = symptomResult && symptomResult.disease !== "Unknown";

  console.log("[Predict] 🔀 Merging predictions...");
  console.log(`  ├─ Symptom: ${symptomResult?.disease || "None"} (${symptomResult?.confidence || 0})`);
  console.log(`  └─ Report:  ${reportResult?.disease || "None"} (${reportResult?.confidence || 0})`);

  // Both sources agree → boost confidence
  if (hasReport && hasSymptom && reportResult.disease === symptomResult.disease) {
    const boosted = Math.min(
      parseFloat(((reportResult.confidence + symptomResult.confidence) / 2 + 0.05).toFixed(2)),
      0.99
    );
    console.log(`[Predict] 🎯 Both agree: ${reportResult.disease} — boosted to ${boosted}`);
    return {
      disease: reportResult.disease,
      confidence: boosted,
      source: "merged (symptoms + report)",
    };
  }

  // Report takes priority when available
  if (hasReport) {
    console.log(`[Predict] 📄 Using report result: ${reportResult.disease}`);
    return {
      disease: reportResult.disease,
      confidence: reportResult.confidence,
      source: "report",
    };
  }

  // Fall back to symptom-based result
  if (hasSymptom) {
    console.log(`[Predict] 🩺 Using symptom result: ${symptomResult.disease}`);
    return {
      disease: symptomResult.disease,
      confidence: symptomResult.confidence,
      source: "symptoms",
    };
  }

  // Neither source produced a result
  console.log("[Predict] ⚠️  No prediction from either source");
  return {
    disease: "Unknown",
    confidence: 0,
    source: "none",
  };
}

module.exports = {
  predictFromSymptoms,
  predictFromReport,
  mergePredictions,
};
