// Inputs per pathway stage. Ranges mirror backend/schemas.py.
export const STAGE_FIELDS = {
  stage1: {
    title: "Stage 1 · Cognitive screening",
    help: "Demographics, screening tests and baseline comorbidities. Required for every assessment.",
    fields: [
      { key: "age", label: "Age", help: "Years, 40–110", min: 40, max: 110, step: 1 },
      { key: "education", label: "Education", help: "Years, 0–30", min: 0, max: 30, step: 1 },
      { key: "mmse", label: "MMSE", help: "Score, 0–30", min: 0, max: 30, step: 1 },
      { key: "adas", label: "ADAS-Cog 11", help: "Score, 0–70", min: 0, max: 70, step: "any" },
      { key: "faq", label: "FAQ", help: "Score, 0–30", min: 0, max: 30, step: 1 },
      { key: "gds", label: "Depression (GDS)", help: "Score, 0–15", min: 0, max: 15, step: 1 },
      { key: "bmi", label: "BMI", help: "kg/m²", min: 12, max: 60, step: "any" },
      { key: "sbp", label: "Systolic BP", help: "mmHg", min: 60, max: 260, step: 1 },
      { key: "hx_cardiovascular", label: "Cardiovascular disease", help: "History", options: [["", "Not recorded"], ["1", "Yes"], ["0", "No"]] },
      { key: "hx_endocrine", label: "Diabetes / endocrine", help: "History", options: [["", "Not recorded"], ["1", "Yes"], ["0", "No"]] },
      { key: "hx_psychiatric", label: "Psychiatric", help: "History", options: [["", "Not recorded"], ["1", "Yes"], ["0", "No"]] },
      { key: "hx_neurologic", label: "Neurologic (non-AD)", help: "History", options: [["", "Not recorded"], ["1", "Yes"], ["0", "No"]] },
      { key: "hx_smoking", label: "Smoking", help: "History", options: [["", "Not recorded"], ["1", "Yes"], ["0", "No"]] },
    ],
  },
  stage2: {
    title: "Stage 2 · Blood-based biomarkers",
    help: "Plasma results refine the score for patients escalated after screening.",
    fields: [
      { key: "ptau217", label: "p-tau217", help: "pg/mL", min: 0, step: "any" },
      { key: "ab42_ab40", label: "Aβ42/40 ratio", help: "0–1", min: 0, max: 1, step: "any" },
      { key: "nfl", label: "NfL", help: "pg/mL", min: 0, step: "any" },
      { key: "gfap", label: "GFAP", help: "pg/mL", min: 0, step: "any" },
      { key: "ptau181", label: "p-tau181", help: "pg/mL", min: 0, step: "any" },
      { key: "nfl_legacy", label: "NfL (older assay)", help: "pg/mL, Blennow lab", min: 0, step: "any" },
      { key: "apoe4", label: "APOE ε4 alleles", help: "Genotype", options: [["", "Not tested"], ["0", "0 (e.g. ε3/ε3)"], ["1", "1 (e.g. ε3/ε4)"], ["2", "2 (ε4/ε4)"]] },
    ],
  },
  stage3: {
    title: "Stage 3 · MRI evaluation",
    help: "MRI-derived volumetrics after imaging.",
    fields: [
      { key: "eTIV", label: "eTIV", help: "cm³", min: 0, step: "any" },
      { key: "nWBV", label: "nWBV", help: "0–1", min: 0, max: 1, step: "any" },
      { key: "ASF", label: "ASF", help: "Atlas scaling factor", min: 0, step: "any" },
      { key: "hippocampus_icv", label: "Hippocampus / ICV", help: "e.g. 0.0049", min: 0, max: 0.02, step: "any" },
    ],
  },
  stage4: {
    title: "Stage 4 · PET imaging",
    help: "Amyloid PET and FDG-PET for narrowed-down candidates.",
    fields: [
      { key: "amyloid_centiloid", label: "Amyloid Centiloid", help: "−100 to 300", min: -100, max: 300, step: "any" },
      { key: "fdg", label: "FDG-PET MetaROI", help: "e.g. 1.24", min: 0.5, max: 2, step: "any" },
    ],
  },
};

export const STAGES = ["stage1", "stage2", "stage3", "stage4"];

export const STAGE_NAMES = {
  stage1: "Cognitive screening",
  stage2: "Blood biomarkers",
  stage3: "MRI",
  stage4: "PET",
};

export function toNumberOrNull(value) {
  if (value === "" || value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function pct(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function priorityClass(priority) {
  return String(priority || "LOW").toLowerCase();
}
