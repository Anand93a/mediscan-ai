/**
 * fileParserService.js — Unified text extraction service for MediScan AI.
 *
 * Supports:
 *   - PDF   → pdf-parse
 *   - Images (PNG, JPG, JPEG, BMP, GIF, WebP) → tesseract.js (OCR)
 *   - DOC/DOCX → mammoth
 *   - TXT   → fs.readFileSync
 *   - CSV   → custom parser (returns structured text)
 *
 * Usage:
 *   const { extractText } = require("./fileParserService");
 *   const result = await extractText(filePath, mimeType);
 *   // result = { text: "...", confidence: 0-100, method: "pdf|ocr|docx|txt|csv" }
 */

const fs = require("fs");
const path = require("path");
const Tesseract = require("tesseract.js");

// ============================================================
// PDF Extraction
// ============================================================
async function extractFromPDF(filePath) {
  try {
    const pdfParse = require("pdf-parse");
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return {
      text: data.text || "",
      confidence: data.text && data.text.trim().length > 10 ? 85 : 30,
      method: "pdf",
      pages: data.numpages || 0,
    };
  } catch (err) {
    console.error("[FileParser] PDF extraction failed:", err.message);
    return { text: "", confidence: 0, method: "pdf", error: err.message };
  }
}

// ============================================================
// Image OCR Extraction (Tesseract.js)
// ============================================================
async function extractFromImage(filePath) {
  try {
    const { data } = await Tesseract.recognize(filePath, "eng", {
      logger: (m) => {
        if (m.status === "recognizing text" && m.progress) {
          // Progress logging (optional)
        }
      },
    });
    return {
      text: data.text || "",
      confidence: data.confidence || 0,
      method: "ocr",
    };
  } catch (err) {
    console.error("[FileParser] OCR extraction failed:", err.message);
    return { text: "", confidence: 0, method: "ocr", error: err.message };
  }
}

// ============================================================
// DOC/DOCX Extraction (Mammoth)
// ============================================================
async function extractFromDOCX(filePath) {
  try {
    const mammoth = require("mammoth");
    const result = await mammoth.extractRawText({ path: filePath });
    const text = result.value || "";
    return {
      text,
      confidence: text.trim().length > 10 ? 90 : 30,
      method: "docx",
      warnings: result.messages || [],
    };
  } catch (err) {
    console.error("[FileParser] DOCX extraction failed:", err.message);
    return { text: "", confidence: 0, method: "docx", error: err.message };
  }
}

// ============================================================
// TXT Extraction
// ============================================================
function extractFromTXT(filePath) {
  try {
    const text = fs.readFileSync(filePath, "utf-8");
    return {
      text,
      confidence: text.trim().length > 0 ? 100 : 0,
      method: "txt",
    };
  } catch (err) {
    console.error("[FileParser] TXT extraction failed:", err.message);
    return { text: "", confidence: 0, method: "txt", error: err.message };
  }
}

// ============================================================
// CSV Extraction — Converts CSV content to readable text
// ============================================================
function extractFromCSV(filePath) {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);

    if (lines.length === 0) {
      return { text: "", confidence: 0, method: "csv" };
    }

    // Parse header and rows
    const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
    const rows = lines.slice(1).map((line) =>
      line.split(",").map((cell) => cell.trim().replace(/^"|"$/g, ""))
    );

    // Convert to readable text format for analysis
    let textParts = [];
    textParts.push(`CSV Report with ${rows.length} data rows.`);
    textParts.push(`Columns: ${header.join(", ")}`);
    textParts.push(""); // blank line

    // Add each row as "Column: Value" pairs
    rows.forEach((row, i) => {
      const pairs = header.map((col, j) => `${col}: ${row[j] || "N/A"}`);
      textParts.push(`Row ${i + 1}: ${pairs.join(", ")}`);
    });

    const text = textParts.join("\n");
    return {
      text,
      confidence: 95,
      method: "csv",
      rowCount: rows.length,
      columns: header,
    };
  } catch (err) {
    console.error("[FileParser] CSV extraction failed:", err.message);
    return { text: "", confidence: 0, method: "csv", error: err.message };
  }
}

// ============================================================
// Unified Extract Function — Routes to the correct parser
// ============================================================
/**
 * Extract text from a file based on its MIME type or extension.
 * @param {string} filePath - Absolute path to the file
 * @param {string} mimeType - MIME type of the file (e.g., "application/pdf")
 * @param {string} [originalName] - Original filename (used for extension fallback)
 * @returns {Promise<{text: string, confidence: number, method: string}>}
 */
async function extractText(filePath, mimeType, originalName = "") {
  // Determine file type from MIME or extension
  const ext = path.extname(originalName || filePath).toLowerCase();
  const mime = (mimeType || "").toLowerCase();

  // PDF
  if (mime === "application/pdf" || ext === ".pdf") {
    return await extractFromPDF(filePath);
  }

  // Images → OCR
  if (
    mime.startsWith("image/") ||
    [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"].includes(ext)
  ) {
    return await extractFromImage(filePath);
  }

  // DOCX
  if (
    mime === "application/vnd.openxmlformats-officedocument.wordprocessingml.document" ||
    ext === ".docx"
  ) {
    return await extractFromDOCX(filePath);
  }

  // DOC (legacy .doc — mammoth only supports .docx, so we try and fall back)
  if (mime === "application/msword" || ext === ".doc") {
    // mammoth can sometimes handle .doc files
    const result = await extractFromDOCX(filePath);
    if (result.text.trim().length > 0) return result;
    // Fallback: try reading as text
    return extractFromTXT(filePath);
  }

  // CSV
  if (mime === "text/csv" || ext === ".csv") {
    return extractFromCSV(filePath);
  }

  // TXT (or any other text-based file)
  if (
    mime === "text/plain" ||
    ext === ".txt" ||
    mime.startsWith("text/")
  ) {
    return extractFromTXT(filePath);
  }

  // Fallback: try reading as text
  console.warn(`[FileParser] Unknown file type: ${mime} / ${ext}, attempting text read`);
  return extractFromTXT(filePath);
}

module.exports = {
  extractText,
  extractFromPDF,
  extractFromImage,
  extractFromDOCX,
  extractFromTXT,
  extractFromCSV,
};
