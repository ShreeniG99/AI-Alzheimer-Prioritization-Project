import React, { useEffect, useState } from "react";
import { api } from "../api";
import { STAGES, STAGE_NAMES, pct, toNumberOrNull } from "../fields";
import StageInputs from "./StageInputs";
import { PathwayTrace, PriorityBadge, ShapChart } from "./ResultViews";

function NewDataForm({ patient, capacity, onDone, onCancel }) {
  const nextIndex = Math.min(STAGES.indexOf(patient.current_stage) + 1, STAGES.length - 1);
  const [stage, setStage] = useState(STAGES[nextIndex]);
  const [values, setValues] = useState({});
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    const numeric = Object.fromEntries(
      Object.entries(values)
        .map(([k, v]) => [k, toNumberOrNull(v)])
        .filter(([, v]) => v !== null)
    );
    if (Object.keys(numeric).length === 0) {
      setError("Enter at least one value.");
      return;
    }
    setSaving(true);
    try {
      const data = await api(`/patients/${patient.id}`, {
        method: "PATCH",
        body: { values: numeric, capacity: toNumberOrNull(capacity) },
      });
      onDone(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="new-data-form" onSubmit={submit}>
      <label>
        Results to add
        <select value={stage} onChange={(e) => { setStage(e.target.value); setValues({}); }}>
          {STAGES.map((s) => <option key={s} value={s}>{STAGE_NAMES[s]}</option>)}
        </select>
      </label>
      <StageInputs stage={stage} values={values} onChange={(k, v) => setValues((p) => ({ ...p, [k]: v }))} />
      {error && <div className="alert error">{error}</div>}
      <div className="form-actions">
        <button type="submit" className="primary-button" disabled={saving}>
          {saving ? "Re-scoring..." : "Save & re-rank"}
        </button>
        <button type="button" className="secondary-button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

export default function Queue() {
  const [ranking, setRanking] = useState(null);
  const [capacity, setCapacity] = useState("");
  const [stageFilter, setStageFilter] = useState("");
  const [expanded, setExpanded] = useState(null);
  const [detail, setDetail] = useState(null);
  const [editing, setEditing] = useState(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const triage = async () => {
    setError("");
    setLoading(true);
    try {
      const data = await api("/triage", {
        method: "POST",
        body: { capacity: toNumberOrNull(capacity), stage: stageFilter || null },
      });
      setRanking(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    triage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggle = async (patient) => {
    setEditing(null);
    if (expanded === patient.id) {
      setExpanded(null);
      return;
    }
    setExpanded(patient.id);
    setDetail(null);
    try {
      setDetail(await api(`/patients/${patient.id}`));
    } catch (err) {
      setError(err.message);
    }
  };

  const remove = async (patient) => {
    if (!window.confirm(`Remove ${patient.assessment_id} from the queue?`)) return;
    try {
      await api(`/patients/${patient.id}`, { method: "DELETE" });
      setMessage(`Removed ${patient.assessment_id}.`);
      setExpanded(null);
      triage();
    } catch (err) {
      setError(err.message);
    }
  };

  const onNewData = (data) => {
    const before = data.previous;
    const after = data.patient;
    setMessage(
      `${after.assessment_id}: ${STAGE_NAMES[before.stage]} ${before.priority} (${pct(before.risk_score)}) → ` +
      `${STAGE_NAMES[after.current_stage]} ${after.priority} (${pct(after.risk_score)}). Queue re-ranked.`
    );
    setEditing(null);
    setDetail(after);
    if (stageFilter) {
      triage();
    } else {
      setRanking({ ...data.queue, task: "rerank" });
    }
  };

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <span className="section-number">05</span>
          <h2>Prioritization Queue</h2>
        </div>
        <span className="stage-tag">TRIAGE</span>
      </div>
      <p className="section-description">
        Saved assessments ranked by calibrated risk. Enter the number of
        available diagnostic slots (e.g. blood tests, MRI or PET scans) to
        simulate capacity. Adding new results to a patient re-scores them and
        re-ranks the whole queue.
      </p>

      <div className="ranking-controls">
        <label>
          Available capacity
          <span className="input-help">Optional number of slots</span>
          <input type="number" min="1" value={capacity}
            onChange={(e) => setCapacity(e.target.value)} placeholder="e.g. 5" />
        </label>
        <label>
          Current stage
          <span className="input-help">Filter the queue</span>
          <select value={stageFilter} onChange={(e) => setStageFilter(e.target.value)}>
            <option value="">All stages</option>
            {STAGES.map((s) => <option key={s} value={s}>{STAGE_NAMES[s]}</option>)}
          </select>
        </label>
        <button type="button" className="primary-button" onClick={triage} disabled={loading}>
          {loading ? "Ranking..." : "Run triage"}
        </button>
      </div>

      {message && <div className="alert success">{message}</div>}
      {error && <div className="alert error"><strong>Error:</strong> {error}</div>}

      {ranking && ranking.total_patients === 0 && (
        <div className="empty-state">
          No patients in the queue yet. New assessments are added automatically.
        </div>
      )}

      {ranking && ranking.total_patients > 0 && (
        <div className="ranking-results">
          <div className="ranking-summary">
            <div><span>Patients</span><strong>{ranking.total_patients}</strong></div>
            <div><span>Selected for capacity</span><strong>{ranking.selected_count}</strong></div>
            <div><span>High / Medium / Low</span>
              <strong>
                {ranking.cohort_summary.high_priority_count} / {ranking.cohort_summary.medium_priority_count} /{" "}
                {ranking.cohort_summary.low_priority_count}
              </strong>
            </div>
            <div><span>Mean risk</span><strong>{pct(ranking.cohort_summary.mean_risk)}</strong></div>
          </div>

          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Patient</th>
                  <th>Stage</th>
                  <th>Risk</th>
                  <th>Priority</th>
                  <th>Next step</th>
                  <th>Capacity</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {ranking.patients.map((patient) => (
                  <React.Fragment key={patient.id}>
                    <tr className={patient.selected_for_capacity ? "selected-row" : ""}>
                      <td><strong>#{patient.rank}</strong></td>
                      <td>
                        <strong>{patient.patient_ref}</strong>
                        <small className="cell-sub">{patient.assessment_id}</small>
                      </td>
                      <td>{STAGE_NAMES[patient.current_stage]}</td>
                      <td>{pct(patient.risk_score, 2)}</td>
                      <td><PriorityBadge priority={patient.priority} /></td>
                      <td className="cell-small">{patient.next_step?.action || "—"}</td>
                      <td>{patient.selected_for_capacity ? "SELECTED" : "—"}</td>
                      <td className="row-actions">
                        <button type="button" className="link-button" onClick={() => toggle(patient)}>
                          {expanded === patient.id ? "Close" : "Details"}
                        </button>
                        <button type="button" className="link-button"
                          onClick={() => { setExpanded(patient.id); setEditing(patient.id); }}>
                          Add results
                        </button>
                        <button type="button" className="remove-button" onClick={() => remove(patient)}>
                          Remove
                        </button>
                      </td>
                    </tr>
                    {expanded === patient.id && (
                      <tr className="detail-row">
                        <td colSpan={8}>
                          {editing === patient.id ? (
                            <NewDataForm
                              patient={patient}
                              capacity={capacity}
                              onDone={onNewData}
                              onCancel={() => setEditing(null)}
                            />
                          ) : detail && detail.id === patient.id ? (
                            <div className="patient-detail">
                              <PathwayTrace trace={detail.pathway?.trace || []} nextStep={detail.next_step} />
                              <h3>SHAP explanation</h3>
                              <ShapChart explanation={detail.explanation} />
                            </div>
                          ) : (
                            <p>Loading...</p>
                          )}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>

          <div className="research-notice">
            <strong>Research resource-allocation simulation:</strong> rankings
            and capacity selection support clinician-led workflows. They do not
            autonomously determine who receives care or diagnostic testing.
          </div>
        </div>
      )}
    </section>
  );
}
