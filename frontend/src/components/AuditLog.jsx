import React, { useEffect, useState } from "react";
import { api } from "../api";

export default function AuditLog() {
  const [entries, setEntries] = useState([]);
  const [error, setError] = useState("");

  const load = () => api("/audit?limit=300").then(setEntries).catch((err) => setError(err.message));

  useEffect(() => {
    load();
  }, []);

  return (
    <section className="card">
      <div className="section-heading">
        <div>
          <span className="section-number">07</span>
          <h2>Audit Log</h2>
        </div>
        <button type="button" className="secondary-button" onClick={load}>Refresh</button>
      </div>
      <p className="section-description">
        Every sign-in, assessment, view, update, triage and deletion is recorded.
      </p>
      {error && <div className="alert error">{error}</div>}
      <div className="table-wrapper">
        <table>
          <thead>
            <tr><th>Time (UTC)</th><th>User</th><th>Action</th><th>Resource</th><th>Details</th></tr>
          </thead>
          <tbody>
            {entries.map((e, i) => (
              <tr key={i}>
                <td className="cell-small">{e.timestamp.replace("T", " ").slice(0, 19)}</td>
                <td>{e.username}</td>
                <td><code>{e.action}</code></td>
                <td className="cell-small">{e.resource}</td>
                <td className="cell-small">{e.details ? JSON.stringify(e.details) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
