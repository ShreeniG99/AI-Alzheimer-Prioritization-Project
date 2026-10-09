// Inputs per pathway stage. Ranges mirror backend/schemas.py.
export const STAGE_FIELDS = {
  stage1: {
    title: "Stage 1 · Cognitive screening",
    help: "Demographics and screening tests. Required for every assessment.",
    fields: [
      { key: "education", label: "Education", help: "Years, 0–30", min: 0, max: 30, step: 1 },
      { key: "mmse", label: "MMSE", help: "Score, 0–30", min: 0, max: 30, step: 1 },
      { key: "adas", label: "ADAS-Cog 11", help: "Score, 0–70", min: 0, max: 70, step: "any" },
      { key: "faq", label: "FAQ", help: "Score, 0–30", min: 0, max: 30, step: 1 },
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
    ],
  },
  stage3: {
    title: "Stage 3 · MRI evaluation",
    help: "MRI-derived volumetrics after imaging.",
    fields: [
      { key: "eTIV", label: "eTIV", help: "cm³", min: 0, step: "any" },
      { key: "nWBV", label: "nWBV", help: "0–1", min: 0, max: 1, step: "any" },
      { key: "ASF", label: "ASF", help: "Atlas scaling factor", min: 0, step: "any" },
    ],
  },
  stage4: {
    title: "Stage 4 · PET imaging",
    help: "Amyloid and tau PET for narrowed-down candidates.",
    fields: [
      { key: "amyloid_centiloid", label: "Amyloid Centiloid", help: "−100 to 300", min: -100, max: 300, step: "any" },
      { key: "amyloid_suvr", label: "Amyloid SUVR", help: "Summary SUVR", min: 0, step: "any" },
      { key: "tau_suvr", label: "Tau SUVR", help: "Meta-temporal", min: 0, step: "any" },
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
