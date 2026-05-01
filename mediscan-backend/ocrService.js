// ============================================================
// ocrService.js — OCR Engine (Tesseract.js)
// Extracts text from uploaded medical images/reports
// ============================================================

const Tesseract = require("tesseract.js");

/**
 * Extract text from an image file using Tesseract OCR.
 *
 * @param {string} filePath — Absolute path to the image file
 * @returns {Promise<{ text: string, confidence: number }>}
 */
async function extractTextFromImage(filePath) {
  try {
    console.log(`[OCR] 🔍 Starting OCR on: ${filePath}`);

    const { data } = await Tesseract.recognize(filePath, "eng", {
      // Log progress updates to the console
      logger: (info) => {
        if (info.status === "recognizing text") {
          console.log(`[OCR] ⏳ Progress: ${Math.round(info.progress * 100)}%`);
        }
      },
    });

    // Normalize: lowercase + trim whitespace
    const extractedText = data.text.toLowerCase().trim();

    console.log(`[OCR] ✅ Done — extracted ${extractedText.length} chars (confidence: ${data.confidence}%)`);

    return {
      text: extractedText,
      confidence: data.confidence,
    };
  } catch (error) {
    console.error(`[OCR] ❌ Failed:`, error.message);
    return {
      text: "",
      confidence: 0,
    };
  }
}

module.exports = { extractTextFromImage };
