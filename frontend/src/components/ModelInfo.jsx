import React from "react";
import { pct } from "../fields";

const METRICS = [
  ["roc_auc", "ROC-AUC"],
  ["pr_auc", "PR-AUC"],
  ["recall", "Recall"],
  ["top_k_recall", "Top-20% recall"],
  ["brier", "Brier"],
  ["ece", "ECE"],
];

export default function ModelInfo({ modelInfo }) {
  if (!modelInfo) return <section className="card"><p>Loading model information...</p></section>;

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <span className="section-number">06</span>
          <h2>Model Information</h2>
        </div>
      </div>
      <p className="section-description">
        One calibrated model per pathway stage, selected from logistic
        regression, random forest and XGBoost by PR-AUC, top-k recall,
        calibration and interpretability. Metrics are on an untouched test split.
      </p>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Stage</th>
              <th>Model</th>
              <th>Cohort</th>
              {METRICS.map(([, label]) => <th key={label}>{label}</th>)}
              <th>High / Medium threshold</th>
            </tr>
          </thead>
          <tbody>
            {modelInfo.stages.map((s) => (
              <tr key={s.stage}>
                <td><strong>{s.name}</strong><small className="cell-sub">{s.stage}</small></td>
                {s.available ? (
                  <>
                    <td>{s.selected_model}<small className="cell-sub">{s.model_version}</small></td>
                    <td>{s.cohort?.participants ?? "—"}</td>
                    {METRICS.map(([key]) => (
                      <td key={key}>
                        {s.test_metrics?.[key] !== undefined ? Number(s.test_metrics[key]).toFixed(3) : "—"}
                      </td>
                    ))}
                    <td>
                      {pct(s.priority_thresholds.high_threshold, 1)} / {pct(s.priority_thresholds.medium_threshold, 1)}
                    </td>
                  </>
                ) : (
                  <td colSpan={METRICS.length + 3} className="muted">
                    Not trained yet — run scripts/02_build_dataset.py and scripts/03_train.py with --stage {s.stage}.
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3>TEMPO: expected decline (annual CDR-SB change)</h3>
      <p className="section-description">
        Ridge regression, random forest and XGBoost compared per stage; metrics on
        an untouched test split. "Fast captured" = share of truly fast decliners
        (≥ 1 point/year) in the predicted top 20%.
      </p>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Stage</th><th>Model</th><th>Cohort</th><th>MAE</th><th>R²</th>
              <th>Spearman</th><th>Fast captured</th><th>Simple-sum fast captured</th><th>80% interval</th>
            </tr>
          </thead>
          <tbody>
            {modelInfo.stages.map((s) => {
              const t = s.tempo;
              if (!t) {
                return (
                  <tr key={s.stage}>
                    <td><strong>{s.name}</strong></td>
                    <td colSpan={8} className="muted">Not trained — run scripts/11_train_tempo.py --stage {s.stage}.</td>
                  </tr>
                );
              }
              const m = t.test_metrics || {};
              const fmt = (v) => (v === undefined || v === null ? "—" : Number(v).toFixed(3));
              return (
                <tr key={s.stage}>
                  <td><strong>{s.name}</strong><small className="cell-sub">{s.stage}</small></td>
                  <td>{t.selected_model}</td>
                  <td>{t.cohort?.participants}</td>
                  <td>{fmt(m.mae)}</td>
                  <td>{fmt(m.r2)}</td>
                  <td>{fmt(m.spearman)}</td>
                  <td>{m.fast_capture !== undefined ? pct(m.fast_capture, 0) : "—"}</td>
                  <td>{t.baselines?.impairment_sum_ranking ? pct(t.baselines.impairment_sum_ranking.fast_capture, 0) : "—"}</td>
                  <td>±{Number(t.interval_halfwidth).toFixed(2)}/yr</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {modelInfo.stages.filter((s) => s.available).map((s) => (
        <div key={s.stage} className="feature-block">
          <h3>{s.name} features</h3>
          <div className="feature-tags">
            {s.features.map((f) => <span key={f}>{f}</span>)}
          </div>
        </div>
      ))}

      <div className="research-notice">
        <strong>Important:</strong> thresholds are capacity-oriented research
        cut-points (top 20% = HIGH, next 50% = MEDIUM on development data),
        not validated clinical cutoffs.
      </div>
    </section>
  );
}
