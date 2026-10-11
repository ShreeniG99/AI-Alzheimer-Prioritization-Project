import React, { useState } from "react";
import { pct, priorityClass } from "../fields";

export function PriorityBadge({ priority }) {
  return (
    <span className={`table-priority ${priorityClass(priority)}`}>{priority}</span>
  );
}

export function PathwayTrace({ trace, nextStep }) {
  return (
    <div>
      <div className="pathway-trace">
        {trace.map((step) => (
          <div key={step.stage} className={step.available ? "trace-step" : "trace-step unavailable"}>
            <span className="metric-label">{step.name}</span>
            {step.available ? (
              <>
                <strong>{pct(step.risk_score)}</strong>
                <PriorityBadge priority={step.priority} />
              </>
            ) : (
              <small>{step.note}</small>
            )}
          </div>
        ))}
      </div>
      {nextStep && (
        <div className={`next-step ${nextStep.next_stage ? "escalate" : ""}`}>
          <span className="metric-label">RECOMMENDED WORKFLOW STEP</span>
          <strong>{nextStep.action}</strong>
          <p>{nextStep.detail}</p>
          {nextStep.next_stage && !nextStep.next_stage_model_available && (
            <small>No model is deployed yet for the next stage.</small>
          )}
        </div>
      )}
    </div>
  );
}

function displayValue(c) {
  if (c.missing) return "missing (imputed)";
  if (c.feature === "sex") return { 1: "Male", 2: "Female" }[c.value] || c.value;
  if (c.feature.startsWith("hx_")) return { 1: "Yes", 0: "No" }[c.value] ?? c.value;
  return c.value;
}

const SHAP_TOP_N = 10;

export function ShapChart({ explanation }) {
  const [showAll, setShowAll] = useState(false);
  if (!explanation) return null;
  const all = explanation.contributions || [];
  const items = showAll ? all : all.slice(0, SHAP_TOP_N);
  const max = Math.max(...items.map((c) => Math.abs(c.shap_value)), 1e-9);

  return (
    <div>
      <p className="section-description">
        {explanation.method} on the {explanation.stage} {explanation.model?.replace("_", " ")} model
        (values in {explanation.output_space}). Bars to the right raised this
        patient's score; bars to the left lowered it.
      </p>
      <div className="shap-chart">
        {items.map((c) => {
          const width = (Math.abs(c.shap_value) / max) * 50;
          const positive = c.shap_value > 0;
          return (
            <div key={c.feature} className="shap-row">
              <span className="shap-label">
                {c.label}
                <small>{displayValue(c)}</small>
              </span>
              <div className="shap-track">
                <div className="shap-axis" />
                <div
                  className={positive ? "shap-bar up" : "shap-bar down"}
                  style={positive ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
                />
              </div>
              <span className={positive ? "direction-up" : c.shap_value < 0 ? "direction-down" : "direction-neutral"}>
                {c.shap_value > 0 ? "+" : ""}
                {c.shap_value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
      {all.length > SHAP_TOP_N && (
        <button type="button" className="link-button" onClick={() => setShowAll((v) => !v)}>
          {showAll ? `Show top ${SHAP_TOP_N} only` : `Show all ${all.length} inputs`}
        </button>
      )}
      <div className="explanation-note">{explanation.interpretation}</div>
    </div>
  );
}

export function QualityPanel({ quality }) {
  if (!quality) return null;
  return (
    <div className="quality-panel">
      <span className="metric-label">DATA QUALITY</span>
      <div className={`quality-badge ${quality.level.toLowerCase()}`}>{quality.level}</div>
      <p>
        <strong>{quality.available}</strong> of <strong>{quality.total}</strong> model inputs available
      </p>
      <div className="progress-track">
        <div className="progress-bar" style={{ width: `${quality.completeness * 100}%` }} />
      </div>
      {quality.warnings?.length > 0 && (
        <div className="warning-list">
          {quality.warnings.map((warning, index) => (
            <p key={index}>{warning}</p>
          ))}
        </div>
      )}
    </div>
  );
}

export function UncertaintyPanel({ uncertainty }) {
  if (!uncertainty) return null;
  return (
    <>
      <div className="uncertainty-grid">
        {uncertainty.production_calibrated_risk !== undefined && (
          <div className="uncertainty-box">
            <span>Production calibrated risk ({uncertainty.stage})</span>
            <strong>{pct(uncertainty.production_calibrated_risk, 2)}</strong>
          </div>
        )}
        {uncertainty.available && (
          <div className="uncertainty-box">
            <span>Bootstrap raw model probability</span>
            <strong>{pct(uncertainty.bootstrap_raw_probability_mean, 2)}</strong>
            <small>Mean across {uncertainty.bootstrap_replicates} uncalibrated refits.</small>
          </div>
        )}
      </div>
      {uncertainty.available ? (
        <div className="interval-panel">
          <h3>Bootstrap raw-probability interval</h3>
          <div className="interval-value">
            {pct(uncertainty.bootstrap_raw_probability_lower_95, 2)} –{" "}
            {pct(uncertainty.bootstrap_raw_probability_upper_95, 2)}
          </div>
          <p>
            Empirical uncertainty of raw model estimates across retrospective
            bootstrap refits. It is <strong>not</strong> a clinical confidence interval.
          </p>
        </div>
      ) : (
        <div className="research-notice">
          <strong>Patient-specific bootstrap interval unavailable.</strong>
          <p>{uncertainty.reason}</p>
        </div>
      )}
    </>
  );
}
