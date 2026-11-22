import React, { useState } from "react";
import { login } from "../api";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(username, password);
      onLogin();
    } catch (err) {
      setError(err.message || "Sign-in failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-shell">
      <div className="login-visual">
        <img src="/images/patient-bulb.webp" alt="" />
        <div className="login-brand">
          <img className="login-logo" src="/images/idea-logo.png" alt="IDEA logo" />
          <h2>IDEA</h2>
        </div>
        <p>Intelligent Detection of Early Alzheimer's</p>
      </div>
      <form className="card login-card" onSubmit={submit}>
        <div className="eyebrow">CLINICAL DECISION-SUPPORT RESEARCH PROTOTYPE</div>
        <h1>Welcome back</h1>
        <p className="section-description">
          Sign in with your clinician account. Sessions use signed JSON Web
          Tokens and every action is audit-logged.
        </p>

        {error && <div className="alert error">{error}</div>}

        <label>
          Username
          <input
            autoComplete="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </label>
        <label>
          Password
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </label>
        <button type="submit" className="primary-button" disabled={loading}>
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>
      </div>
    </div>
  );
}
