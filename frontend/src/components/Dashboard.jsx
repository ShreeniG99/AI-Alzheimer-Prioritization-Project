import React from "react";
import { STAGE_NAMES, pct } from "../fields";

export default function Dashboard({ modelInfo, onNavigate }) {
  const stage1 = modelInfo?.stages?.find((s) => s.stage === "stage1");
  const cohort = stage1?.cohort || {};
  const fmt = (v) => (v === undefined || v === null ? "—" : Number(v).toLocaleString());

  return (
    <section className="card dashboard-card">
      <div className="eyebrow">CLINICAL DECISION-SUPPORT RESEARCH PROTOTYPE</div>
      <h1>
        AI-Driven Alzheimer's
        <br />
        Diagnostic Prioritization
      </h1>
      <p className="hero-description">
        Calibrated risk estimation, SHAP explanations and capacity-aware
        prioritization along a progressive diagnostic pathway: cognitive
        screening → blood biomarkers → MRI → PET.
      </p>

      <div className="dashboard-stats">
        <div className="stat-card">
          <span>RESEARCH COHORT</span>
          <strong>{fmt(cohort.participants)}</strong>
          <small>Stage-1 participants</small>
        </div>
        <div className="stat-card">
          <span>TARGET CASES</span>
          <strong>{fmt(cohort.positives)}</strong>
          <small>24-month progression to dementia</small>
        </div>
        <div className="stat-card">
          <span>TARGET RATE</span>
          <strong>{cohort.positive_rate !== undefined ? pct(cohort.positive_rate, 2) : "—"}</strong>
          <small>Retrospective cohort rate</small>
        </div>
        <div className="stat-card">
          <span>COMPLETE INPUTS</span>
          <strong>
            {cohort.complete_input_fraction !== undefined
              ? pct(cohort.complete_input_fraction, 2)
              : "—"}
          </strong>
          <small>Records with all Stage-1 inputs</small>
        </div>
      </div>

      <div className="pathway-overview">
        {(modelInfo?.stages || []).map((stage, index) => (
          <div
            key={stage.stage}
            className={stage.available ? "pathway-step available" : "pathway-step"}
          >
            <span className="step-number">{index + 1}</span>
            <strong>{STAGE_NAMES[stage.stage]}</strong>
            <small>{stage.available ? stage.selected_model?.replace("_", " ") : "model not trained"}</small>
          </div>
        ))}
      </div>

      <div className="dashboard-actions">
        <button type="button" className="primary-button" onClick={() => onNavigate("assessment")}>
          Start new assessment
        </button>
        <button type="button" className="secondary-button" onClick={() => onNavigate("queue")}>
          Open prioritization queue
        </button>
      </div>

      <div className="research-notice">
        <strong>Research prototype:</strong> This dashboard supports research
        prioritization. It does not provide a medical diagnosis or treatment
        recommendation.
      </div>
    </section>
  );
}
