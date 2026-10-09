import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

import { api, clearSession, getSession, setUnauthorizedHandler } from "./api";
import Login from "./components/Login";
import Dashboard from "./components/Dashboard";
import Assessment from "./components/Assessment";
import Queue from "./components/Queue";
import ModelInfo from "./components/ModelInfo";
import AuditLog from "./components/AuditLog";

const PAGES = [
  { key: "dashboard", label: "Dashboard" },
  { key: "assessment", label: "New assessment" },
  { key: "queue", label: "Prioritization queue" },
  { key: "model", label: "Models" },
  { key: "audit", label: "Audit log", admin: true },
];

function App() {
  const [session, setSession] = useState(getSession());
  const [page, setPage] = useState("dashboard");
  const [modelInfo, setModelInfo] = useState(null);
  const [error, setError] = useState("");

  const logout = useCallback(() => {
    clearSession();
    setSession(null);
    setModelInfo(null);
    setPage("dashboard");
  }, []);

  useEffect(() => {
    setUnauthorizedHandler(logout);
  }, [logout]);

  useEffect(() => {
    if (!session) return;
    api("/model-info")
      .then(setModelInfo)
      .catch((err) => setError(err.message));
  }, [session]);

  if (!session) {
    return <Login onLogin={() => setSession(getSession())} />;
  }

  const isAdmin = session.user.role === "admin";

  return (
    <div className="app">
      <nav className="topnav">
        <div className="topnav-inner">
          <div className="brand">
            <strong>Alzheimer's Diagnostic Prioritization</strong>
            <span>Research prototype</span>
          </div>
          <div className="nav-links">
            {PAGES.filter((p) => !p.admin || isAdmin).map((p) => (
              <button
                key={p.key}
                type="button"
                className={page === p.key ? "nav-link active" : "nav-link"}
                onClick={() => setPage(p.key)}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="nav-user">
            <span>
              {session.user.full_name || session.user.username}
              <small>{session.user.role}</small>
            </span>
            <button type="button" className="secondary-button" onClick={logout}>
              Sign out
            </button>
          </div>
        </div>
      </nav>

      <main>
        {error && (
          <section className="alert error">
            <strong>Request error:</strong> {error}
          </section>
        )}

        {page === "dashboard" && (
          <Dashboard modelInfo={modelInfo} onNavigate={setPage} />
        )}
        {page === "assessment" && <Assessment modelInfo={modelInfo} />}
        {page === "queue" && <Queue />}
        {page === "model" && <ModelInfo modelInfo={modelInfo} />}
        {page === "audit" && isAdmin && <AuditLog />}

        <section className="card final-notice">
          <h2>Research Prototype Notice</h2>
          <p>
            This clinical decision-support research prototype estimates
            progression-related research risk from retrospective ADNI data to
            support prioritization for further diagnostic evaluation.
          </p>
          <p>
            It does not provide a medical diagnosis, does not recommend
            treatment, and does not replace clinician judgment, local protocols,
            or established diagnostic evaluation.
          </p>
        </section>
      </main>

      <footer>
        AI-Driven Prioritization for Early Alzheimer's Diagnostic Pathways ·
        Research Prototype
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
