import React from "react";
import { STAGE_FIELDS } from "../fields";

export default function StageInputs({ stage, values, onChange, available = true }) {
  const spec = STAGE_FIELDS[stage];
  return (
    <fieldset className="stage-fieldset">
      <legend>
        {spec.title}
        {!available && <span className="stage-tag muted-tag">model not trained</span>}
      </legend>
      <p className="input-help">{spec.help}</p>
      <div className="assessment-grid">
        {spec.fields.map((field) => (
          <label key={field.key}>
            {field.label}
            <span className="input-help">{field.help}</span>
            {field.options ? (
              <select
                value={values[field.key] ?? ""}
                onChange={(e) => onChange(field.key, e.target.value)}
              >
                {field.options.map(([value, label]) => (
                  <option key={value} value={value}>{label}</option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                min={field.min}
                max={field.max}
                step={field.step}
                value={values[field.key] ?? ""}
                onChange={(e) => onChange(field.key, e.target.value)}
                placeholder="Optional"
              />
            )}
          </label>
        ))}
      </div>
    </fieldset>
  );
}
