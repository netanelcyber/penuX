/**
 * Ported from PenuX-AP-Severity/api/main.py — the same research-only
 * prediction endpoints, faithfully translated to JS so they can run on
 * this single Node/Express Render service instead of a separate Python
 * deployment. Every formula below is a 1:1 port of the Python source;
 * see main.py for the original with full docstrings/citations.
 *
 * RESEARCH USE ONLY. Not validated for clinical use. Not for patient-care
 * decisions.
 */
import express from 'express';
import { readFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const router = express.Router();

const RESEARCH_WARNING =
  'This is a research prototype only. It is not validated for clinical use and must not be used for patient-care decisions.';

// ─────────────────────────────────────────────────────────────────────────
// /predict — plain JSON heuristic (logistic BISAP/Ranson-weighted fallback,
// used when no trained ML model is configured — same fallback path as the
// Python API's _heuristic_score()).
// ─────────────────────────────────────────────────────────────────────────
const HEURISTIC_WEIGHTS = {
  wbc: 0.08, crp: 0.004, creatinine: 0.30, bun: 0.015,
  glucose: 0.002, ldh: 0.001, hematocrit: 0.03, ast: 0.002,
  albumin: -0.30, calcium: -0.40, bilirubin_total: 0.05,
};
const HEURISTIC_THRESHOLDS = {
  wbc: 12.0, crp: 150.0, creatinine: 1.5, bun: 25.0,
  glucose: 200.0, ldh: 250.0, hematocrit: 44.0, ast: 250.0,
  albumin: 3.5, calcium: 8.0, bilirubin_total: 3.0,
};
const RISK_THRESHOLDS = { low: 0.3, intermediate: 0.6 };

router.post('/predict', (req, res) => {
  const d = req.body || {};
  let logit = -1.8 + 0.015 * Math.max(0, (d.age ?? 55) - 55);
  if (String(d.sex ?? '').toUpperCase().match(/^(M|MALE|זכר)$/)) logit += 0.15;
  for (const [feat, w] of Object.entries(HEURISTIC_WEIGHTS)) {
    const v = d[feat];
    if (v == null) continue;
    const t = HEURISTIC_THRESHOLDS[feat];
    const deviation = (feat === 'albumin' || feat === 'calcium') ? Math.max(0, t - v) : Math.max(0, v - t);
    logit += w * deviation;
  }
  const proba = Math.round((1 / (1 + Math.exp(-logit))) * 10000) / 10000;
  const risk_group = proba < RISK_THRESHOLDS.low ? 'low' : proba < RISK_THRESHOLDS.intermediate ? 'intermediate' : 'high';
  res.json({ severe_ap_probability: proba, threshold_used: 0.5, risk_group, model_version: '0.1.0', warning: RESEARCH_WARNING, error: null });
});

// ─────────────────────────────────────────────────────────────────────────
// /predict/sepsis — SIRS + qSOFA + organ dysfunction
// ─────────────────────────────────────────────────────────────────────────
router.post('/predict/sepsis', (req, res) => {
  const d = req.body || {};
  const criteria = [];
  let sirs = 0, qsofa = 0;

  if (d.temperature_c != null) {
    if (d.temperature_c > 38.3) { sirs++; criteria.push(`Fever (${d.temperature_c}°C > 38.3)`); }
    else if (d.temperature_c < 36.0) { sirs++; criteria.push(`Hypothermia (${d.temperature_c}°C < 36.0)`); }
  }
  if (d.heart_rate != null && d.heart_rate > 90) { sirs++; criteria.push(`Tachycardia (HR ${d.heart_rate} > 90)`); }
  if (d.respiratory_rate != null) {
    if (d.respiratory_rate > 20) { sirs++; criteria.push(`Tachypnea (RR ${d.respiratory_rate} > 20)`); }
    if (d.respiratory_rate >= 22) qsofa++;
  }
  if (d.wbc != null) {
    if (d.wbc > 12.0) { sirs++; criteria.push(`Leukocytosis (WBC ${d.wbc} > 12.0)`); }
    else if (d.wbc < 4.0) { sirs++; criteria.push(`Leukopenia (WBC ${d.wbc} < 4.0)`); }
  }
  if (d.systolic_bp != null && d.systolic_bp <= 100) { qsofa++; criteria.push(`Hypotension (SBP ${d.systolic_bp} ≤ 100)`); }

  let organScore = 0.0;
  if (d.creatinine != null && d.creatinine > 2.0) { organScore += 0.15; criteria.push(`Renal dysfunction (Creatinine ${d.creatinine} > 2.0)`); }
  if (d.bilirubin != null && d.bilirubin > 2.0) { organScore += 0.10; criteria.push(`Hepatic dysfunction (Bilirubin ${d.bilirubin} > 2.0)`); }
  if (d.platelets != null && d.platelets < 100) { organScore += 0.15; criteria.push(`Thrombocytopenia (Platelets ${d.platelets} < 100)`); }
  if (d.lactate != null && d.lactate > 2.0) {
    organScore += 0.20; criteria.push(`Elevated lactate (${d.lactate} > 2.0 mmol/L)`);
    if (d.lactate > 4.0) { organScore += 0.10; criteria.push(`Critical lactate (${d.lactate} > 4.0 — septic shock)`); }
  }
  if (d.map_mmhg != null && d.map_mmhg < 65) { organScore += 0.20; criteria.push(`Low MAP (${d.map_mmhg} < 65 mmHg — vasopressor territory)`); }
  if (d.spo2 != null && d.spo2 < 94) { organScore += 0.08; criteria.push(`Hypoxia (SpO2 ${d.spo2}% < 94%)`); }

  const ageAdj = (d.age != null && d.age > 65) ? 0.05 : 0.0;
  const logit = -2.9 + 0.55 * sirs + 0.65 * qsofa + organScore * 4.0 + ageAdj;
  const proba = Math.round((1 / (1 + Math.exp(-logit))) * 10000) / 10000;

  let risk;
  if (proba < 0.15) risk = 'low';
  else if (proba < 0.40) risk = 'moderate';
  else if (proba < 0.70) risk = 'high';
  else risk = 'critical';

  res.json({
    sepsis_risk_probability: proba, risk_group: risk,
    sirs_score: Math.min(sirs, 4), qsofa_score: qsofa,
    criteria_met: criteria, warning: RESEARCH_WARNING,
  });
});

// ─────────────────────────────────────────────────────────────────────────
// /predict/deterioration — NEWS2
// ─────────────────────────────────────────────────────────────────────────
router.post('/predict/deterioration', (req, res) => {
  const d = req.body || {};
  const scores = {};

  if (d.respiratory_rate != null) {
    const rr = d.respiratory_rate;
    scores.respiratory_rate = rr <= 8 ? 3 : rr <= 11 ? 1 : rr <= 20 ? 0 : rr <= 24 ? 2 : 3;
  }
  if (d.spo2 != null) {
    const s = d.spo2;
    scores.spo2 = s <= 91 ? 3 : s <= 93 ? 2 : s <= 95 ? 1 : 0;
  }
  if (d.on_supplemental_oxygen != null) scores.supplemental_oxygen = d.on_supplemental_oxygen ? 2 : 0;
  if (d.systolic_bp != null) {
    const bp = d.systolic_bp;
    scores.systolic_bp = bp <= 90 ? 3 : bp <= 100 ? 2 : bp <= 110 ? 1 : bp <= 219 ? 0 : 3;
  }
  if (d.heart_rate != null) {
    const hr = d.heart_rate;
    scores.heart_rate = hr <= 40 ? 3 : hr <= 50 ? 1 : hr <= 90 ? 0 : hr <= 110 ? 1 : hr <= 130 ? 2 : 3;
  }
  if (d.consciousness_altered != null) scores.consciousness = d.consciousness_altered ? 3 : 0;
  if (d.temperature_c != null) {
    const t = d.temperature_c;
    scores.temperature = t <= 35.0 ? 3 : t <= 36.0 ? 1 : t <= 38.0 ? 0 : t <= 39.0 ? 1 : 2;
  }

  const total = Object.values(scores).reduce((a, b) => a + b, 0);
  const hasSingle3 = Object.values(scores).some(v => v === 3);

  let risk;
  if (total >= 7) risk = 'high';
  else if (hasSingle3 || total >= 5) risk = 'medium';
  else if (total >= 1) risk = 'low-medium';
  else risk = 'low';

  const escalationMap = {
    low: 'Routine monitoring — continue per ward protocol.',
    'low-medium': 'Increase monitoring frequency; nurse-led review.',
    medium: 'Urgent review by ward-based clinician; consider critical care outreach.',
    high: 'Emergency assessment — critical care outreach team review, consider transfer to higher-acuity setting.',
  };

  res.json({ news2_score: total, risk_group: risk, component_scores: scores, escalation: escalationMap[risk], warning: RESEARCH_WARNING });
});

// ─────────────────────────────────────────────────────────────────────────
// /predict/mortality — 30-day in-hospital mortality heuristic
// ─────────────────────────────────────────────────────────────────────────
router.post('/predict/mortality', (req, res) => {
  const d = req.body || {};
  const factors = [];
  let logit = -4.2;

  if (d.age != null && d.age > 65) {
    logit += 0.03 * (d.age - 65);
    if (d.age > 75) factors.push(`Advanced age (${Math.round(d.age)})`);
  }
  if (d.comorbidity_count != null && d.comorbidity_count > 0) {
    logit += 0.35 * d.comorbidity_count;
    factors.push(`Comorbidity burden (${d.comorbidity_count} major comorbidities)`);
  }
  if (d.systolic_bp != null && d.systolic_bp < 90) { logit += 0.9; factors.push(`Hypotension (SBP ${d.systolic_bp} < 90)`); }
  if (d.heart_rate != null && d.heart_rate > 120) { logit += 0.4; factors.push(`Severe tachycardia (HR ${d.heart_rate} > 120)`); }
  if (d.respiratory_rate != null && d.respiratory_rate > 24) { logit += 0.4; factors.push(`Tachypnea (RR ${d.respiratory_rate} > 24)`); }
  if (d.temperature_c != null && (d.temperature_c < 35.0 || d.temperature_c > 39.5)) { logit += 0.5; factors.push(`Temperature dysregulation (${d.temperature_c}°C)`); }
  if (d.spo2 != null && d.spo2 < 90) { logit += 0.6; factors.push(`Hypoxia (SpO2 ${d.spo2}% < 90%)`); }
  if (d.consciousness_altered) { logit += 0.8; factors.push('Altered consciousness'); }
  if (d.creatinine != null && d.creatinine > 2.0) { logit += 0.5; factors.push(`Renal dysfunction (Creatinine ${d.creatinine} > 2.0)`); }
  if (d.bun != null && d.bun > 40) { logit += 0.3; factors.push(`Elevated BUN (${d.bun} > 40)`); }
  if (d.bilirubin_total != null && d.bilirubin_total > 3.0) { logit += 0.4; factors.push(`Hepatic dysfunction (Bilirubin ${d.bilirubin_total} > 3.0)`); }
  if (d.albumin != null && d.albumin < 2.5) { logit += 0.5; factors.push(`Hypoalbuminemia (Albumin ${d.albumin} < 2.5)`); }
  if (d.platelets != null && d.platelets < 100) { logit += 0.4; factors.push(`Thrombocytopenia (Platelets ${d.platelets} < 100)`); }
  if (d.lactate != null && d.lactate > 2.0) { logit += 0.5 * Math.min(d.lactate / 2.0, 3.0); factors.push(`Elevated lactate (${d.lactate} mmol/L)`); }
  if (d.wbc != null && (d.wbc > 15.0 || d.wbc < 3.0)) { logit += 0.3; factors.push(`Leukocyte abnormality (WBC ${d.wbc})`); }

  const proba = Math.round((1 / (1 + Math.exp(-logit))) * 10000) / 10000;
  let risk;
  if (proba < 0.05) risk = 'low';
  else if (proba < 0.20) risk = 'moderate';
  else if (proba < 0.50) risk = 'high';
  else risk = 'critical';

  res.json({ mortality_risk_probability: proba, risk_group: risk, contributing_factors: factors, warning: RESEARCH_WARNING });
});

// ─────────────────────────────────────────────────────────────────────────
// /predict/saps2 — SAPS II (Le Gall et al., 1993)
// ─────────────────────────────────────────────────────────────────────────
function band(value, table) {
  for (const [lo, hi, pts] of table) {
    if ((lo == null || value >= lo) && (hi == null || value < hi)) return pts;
  }
  return undefined;
}

router.post('/predict/saps2', (req, res) => {
  const d = req.body || {};
  const points = {};
  const missing = [];

  if (d.age != null) points.age = band(d.age, [[null, 40, 0], [40, 60, 7], [60, 70, 12], [70, 75, 15], [75, 80, 16], [80, null, 18]]);
  else missing.push('age');

  if (d.heart_rate != null) points.heart_rate = band(d.heart_rate, [[null, 40, 11], [40, 70, 2], [70, 120, 0], [120, 160, 4], [160, null, 7]]);
  else missing.push('heart_rate');

  if (d.systolic_bp != null) points.systolic_bp = band(d.systolic_bp, [[null, 70, 13], [70, 100, 5], [100, 200, 0], [200, null, 2]]);
  else missing.push('systolic_bp');

  if (d.temperature_c != null) points.temperature = d.temperature_c >= 39.0 ? 3 : 0;
  else missing.push('temperature_c');

  if (d.ventilated_or_cpap && d.pao2_fio2 != null) {
    points.pao2_fio2 = band(d.pao2_fio2, [[null, 100, 11], [100, 200, 9], [200, null, 6]]);
  } else if (d.ventilated_or_cpap) {
    missing.push('pao2_fio2 (ventilated but ratio not supplied)');
  }

  if (d.urine_output_l_24h != null) points.urine_output = band(d.urine_output_l_24h, [[null, 0.5, 11], [0.5, 1.0, 4], [1.0, null, 0]]);
  else missing.push('urine_output_l_24h');

  if (d.bun_mg_dl != null) points.bun = band(d.bun_mg_dl, [[null, 28, 0], [28, 84, 6], [84, null, 10]]);
  else missing.push('bun_mg_dl');

  if (d.wbc != null) points.wbc = band(d.wbc, [[null, 1, 12], [1, 20, 0], [20, null, 3]]);
  else missing.push('wbc');

  if (d.potassium != null) points.potassium = band(d.potassium, [[null, 3, 3], [3, 5, 0], [5, null, 3]]);
  else missing.push('potassium');

  if (d.sodium != null) points.sodium = band(d.sodium, [[null, 125, 5], [125, 145, 0], [145, null, 1]]);
  else missing.push('sodium');

  if (d.bicarbonate != null) points.bicarbonate = band(d.bicarbonate, [[null, 15, 6], [15, 20, 3], [20, null, 0]]);
  else missing.push('bicarbonate');

  if (d.bilirubin_total != null) points.bilirubin = band(d.bilirubin_total, [[null, 4.0, 0], [4.0, 6.0, 4], [6.0, null, 9]]);
  else missing.push('bilirubin_total');

  if (d.gcs != null) points.gcs = band(d.gcs, [[null, 6, 26], [6, 9, 13], [9, 11, 7], [11, 14, 5], [14, null, 0]]);
  else missing.push('gcs');

  const admissionPoints = { scheduled_surgical: 0, medical: 6, unscheduled_surgical: 8 };
  if (d.admission_type in admissionPoints) points.admission_type = admissionPoints[d.admission_type];
  else if (d.admission_type != null) missing.push('admission_type (unrecognized value)');

  const chronicPoints = { metastatic_cancer: 9, hematologic_malignancy: 10, aids: 17 };
  if (d.chronic_disease in chronicPoints) points.chronic_disease = chronicPoints[d.chronic_disease];

  const total = Object.values(points).reduce((a, b) => a + b, 0);
  const logit = -7.7631 + 0.0737 * total + 0.9971 * Math.log(total + 1);
  const proba = Math.round((1 / (1 + Math.exp(-logit))) * 10000) / 10000;

  res.json({ saps2_score: total, predicted_mortality_probability: proba, point_breakdown: points, missing_variables: missing, warning: RESEARCH_WARNING });
});

// ─────────────────────────────────────────────────────────────────────────
// /predict/polynomial-logit — ln(sigmoid(f(x))), f(x) a degree-2 polynomial
// ─────────────────────────────────────────────────────────────────────────
const POLY_VARS = {
  age: [55.0, 15.0, true], wbc: [10.0, 4.0, true], crp: [80.0, 60.0, true],
  creatinine: [1.0, 0.4, true], glucose: [110.0, 40.0, true], ldh: [200.0, 80.0, true],
  ast: [35.0, 30.0, true], hematocrit: [42.0, 6.0, true],
  calcium: [9.2, 0.8, false], albumin: [4.0, 0.6, false],
};
const POLY_LINEAR_WEIGHT = 0.10, POLY_QUADRATIC_WEIGHT = 0.02, POLY_INTERACTION_WEIGHT = 0.02, POLY_INTERCEPT = -2.5;

router.post('/predict/polynomial-logit', (req, res) => {
  const d = req.body || {};
  const z = {};
  const termsUsed = [];
  let fx = POLY_INTERCEPT;

  for (const [name, [center, scale, higherIsWorse]] of Object.entries(POLY_VARS)) {
    const value = d[name];
    if (value == null) continue;
    const rawZ = (value - center) / scale;
    z[name] = higherIsWorse ? rawZ : -rawZ;
    fx += POLY_LINEAR_WEIGHT * z[name] + POLY_QUADRATIC_WEIGHT * (z[name] ** 2);
    termsUsed.push(name);
  }
  if ('wbc' in z && 'crp' in z) {
    fx += POLY_INTERACTION_WEIGHT * z.wbc * z.crp;
    termsUsed.push('wbc×crp interaction');
  }

  const riskProbability = 1 / (1 + Math.exp(-fx));
  const logRiskIndex = Math.log(Math.max(riskProbability, 1e-12));
  const riskGroup = riskProbability < 0.2 ? 'low' : riskProbability < 0.5 ? 'intermediate' : 'high';

  res.json({
    polynomial_score: Math.round(fx * 10000) / 10000,
    risk_probability: Math.round(riskProbability * 1e6) / 1e6,
    log_risk_index: Math.round(logRiskIndex * 1e6) / 1e6,
    risk_group: riskGroup,
    terms_used: termsUsed,
    warning: RESEARCH_WARNING,
  });
});

// ─────────────────────────────────────────────────────────────────────────
// /health — lightweight endpoint for uptime/keep-alive pings (see
// .github/workflows/render-keepalive.yml). Deliberately does nothing but
// return 200 — no DB query, no file read — so pings are cheap.
// ─────────────────────────────────────────────────────────────────────────
router.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'penux-tasks-api', timestamp: new Date().toISOString() });
});

// ─────────────────────────────────────────────────────────────────────────
// /models/sweep — serves the pre-computed 294-model sweep results
// (a bundled snapshot copy — see tasks-api/data/model_sweep_271_results.json)
// ─────────────────────────────────────────────────────────────────────────
const SWEEP_FILE = join(__dirname, 'data', 'model_sweep_271_results.json');
const SWEEP_CAVEATS = [
  "This sweep was run on data/public_sanitized/ap_multiml_sanitized.csv (Guilin Medical University, 2016-2024, n=1289) — NOT the primary n=722 Atlanta-2012-labeled cohort used by /predict and the manuscript. Results here are exploratory/supplementary and not directly comparable to the primary model's reported performance.",
  "This dataset's own SOURCES.md documents label=1 as the minority SAP class (204/15.8%), but the actual CSV has label=1 on the majority (1085 rows) — a real discrepancy between the source's documentation and its data. AUROC/AUPRC are mathematically symmetric to which class is called positive, so the discrimination scores are valid, but which class means 'severe' is unverified.",
];

router.get('/models/sweep', (req, res) => {
  if (!existsSync(SWEEP_FILE)) {
    return res.status(404).json({ error: `Sweep results not found at ${SWEEP_FILE}. Run PenuX-AP-Severity/scripts/model_sweep_271.py first.` });
  }
  const data = JSON.parse(readFileSync(SWEEP_FILE, 'utf-8'));
  let topN = parseInt(req.query.top_n, 10);
  if (!Number.isFinite(topN)) topN = 30;
  topN = Math.max(1, Math.min(topN, data.results_ranked_by_auroc.length));

  res.json({
    dataset: data.dataset,
    n_samples: data.n_samples,
    n_features: data.n_features,
    positive_rate: data.positive_rate,
    cv_folds: data.cv_folds,
    n_configs_attempted: data.n_configs_attempted,
    n_configs_succeeded: data.n_configs_succeeded,
    n_configs_failed: data.n_configs_failed,
    total_runtime_seconds: data.total_runtime_seconds,
    results: data.results_ranked_by_auroc.slice(0, topN),
    caveats: SWEEP_CAVEATS,
  });
});

export default router;
