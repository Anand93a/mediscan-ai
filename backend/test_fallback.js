const fs = require('fs');
function symptomsMatch(userSym, diseaseSym) {
    const u = userSym.toLowerCase().trim();
    const d = diseaseSym.toLowerCase().trim();
    if (u === d) return true;
    const uNorm = u.replace(/[_\-]/g, " ");
    const dNorm = d.replace(/[_\-]/g, " ");
    if (uNorm === dNorm) return true;
    if (uNorm.length >= 4 && dNorm.length >= 4) {
      if (dNorm.includes(uNorm) && uNorm.length >= dNorm.length * 0.5) return true;
      if (uNorm.includes(dNorm) && dNorm.length >= uNorm.length * 0.5) return true;
    }
    return false;
}

const INITIAL_DISEASES = [
  { name: "Common Flu (Influenza)", symptoms: ["fever", "cough", "sore throat", "body ache", "fatigue", "chills", "headache", "runny nose"] },
  { name: "Common Cold", symptoms: ["runny nose", "sneezing", "sore throat", "mild cough", "congestion", "watery eyes"] },
  { name: "COVID-19 (Suspected)", symptoms: ["fever", "dry cough", "loss of taste", "loss of smell", "shortness of breath", "fatigue", "body ache"] },
  { name: "Gastroenteritis (Stomach Flu)", symptoms: ["nausea", "vomiting", "diarrhea", "stomach pain", "fever", "dehydration", "cramps"] },
];

const diseases = INITIAL_DISEASES;

function fallbackPredict(userSymptoms) {
  const normalized = [...new Set(userSymptoms.map((s) => s.toLowerCase().trim().replace(/[_\-]/g, " ")).filter(Boolean))];
  const symptomFrequency = {};
  diseases.forEach((d) => {
    (d.symptoms || []).forEach((s) => {
      const key = s.toLowerCase().trim();
      symptomFrequency[key] = (symptomFrequency[key] || 0) + 1;
    });
  });
  const totalDiseases = diseases.length || 1;

  const scored = diseases.map((entry) => {
    const diseaseSymptoms = (entry.symptoms || []).map((s) => s.toLowerCase().trim());
    let matchedUserSymptoms = 0;
    let matchedDiseaseSymptoms = 0;
    let rarityBonus = 0;
    normalized.forEach((userSym) => {
      const found = diseaseSymptoms.find((ds) => symptomsMatch(userSym, ds));
      if (found) {
        matchedUserSymptoms++;
        const freq = symptomFrequency[found] || 1;
        rarityBonus += (1 - freq / totalDiseases);
      }
    });

    diseaseSymptoms.forEach((ds) => {
      if (normalized.some((userSym) => symptomsMatch(userSym, ds))) matchedDiseaseSymptoms++;
    });

    if (matchedUserSymptoms === 0) return null;
    const precision = matchedUserSymptoms / normalized.length;
    const recall = matchedDiseaseSymptoms / (diseaseSymptoms.length || 1);
    const f1 = precision * 0.4 + recall * 0.4 + (rarityBonus / normalized.length) * 0.2;
    return { name: entry.name, score: f1 };
  }).filter(Boolean);

  scored.sort((a, b) => b.score - a.score);
  return scored[0];
}

console.log("fever:", fallbackPredict(["fever"]));
console.log("nausea:", fallbackPredict(["nausea"]));
console.log("random:", fallbackPredict(["xyzdsfsdf"]));
console.log("cough:", fallbackPredict(["cough"]));
