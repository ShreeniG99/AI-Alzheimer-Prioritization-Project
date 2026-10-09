from pathlib import Path
import os
import re
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Saved model bundles reference ml.model_utils (e.g. PlattCalibrator), so the
# project root must be importable when any script unpickles them.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _data_dir(key, env_var):
    """Directory from config.yaml, overridable by an environment variable.

    The overrides let tests and the synthetic demo run the pipeline in an
    isolated location without touching real ADNI data or artifacts.
    """
    override = os.getenv(env_var)
    if override:
        return Path(override)
    return ROOT / load_config()["data"][key]

def raw_dir():
    return _data_dir("raw_dir", "ALZ_RAW_DIR")

def processed_dir():
    return _data_dir("processed_dir", "ALZ_PROCESSED_DIR")

def artifacts_dir():
    return _data_dir("artifacts_dir", "ALZ_ARTIFACTS_DIR")

def reports_dir():
    return _data_dir("reports_dir", "ALZ_REPORTS_DIR")

def newest_csv(prefix):
    candidates = sorted(raw_dir().glob(f"{prefix}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {prefix}*.csv in {raw_dir()}. "
            "Unzip the ADNI download and copy the CSV into data/raw."
        )
    return candidates[0]

def read_prefix(prefix):
    path = newest_csv(prefix)
    print(f"Loading {path.name}")
    return pd.read_csv(path, low_memory=False)

def normalize_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())

def find_column(df, aliases, required=True):
    norm = {normalize_col(c): c for c in df.columns}
    for alias in aliases:
        key = normalize_col(alias)
        if key in norm:
            return norm[key]
    if required:
        raise KeyError(f"None of these columns were found: {aliases}\nAvailable columns: {list(df.columns)[:80]}")
    return None

def find_id_column(df):
    return find_column(df, ["RID", "PTID", "Subject ID", "SubjectID"])

def ensure_rid(df):
    out = df.copy()
    rid = find_column(out, ["RID"], required=False)
    if rid is not None:
        out["RID"] = pd.to_numeric(out[rid], errors="coerce").astype("Int64")
        return out
    ptid = find_column(out, ["PTID", "Subject ID", "SubjectID"], required=False)
    if ptid is None:
        raise KeyError("No RID/PTID subject identifier found.")
    extracted = out[ptid].astype(str).str.extract(r"(\d+)$")[0]
    out["RID"] = pd.to_numeric(extracted, errors="coerce").astype("Int64")
    return out

def baseline_only(df):
    """Prefer BL/baseline rows; otherwise take earliest dated row per RID."""
    df = ensure_rid(df)
    vis = find_column(df, ["VISCODE2", "VISCODE"], required=False)
    if vis:
        s = df[vis].astype(str).str.lower()
        mask = s.isin(["bl", "baseline", "sc"])
        if mask.any():
            return df.loc[mask].copy()

    date_col = find_column(df, ["EXAMDATE", "EXAMDATE2", "SCANDATE", "USERDATE"], required=False)
    if date_col:
        tmp = df.copy()
        tmp["_date"] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.sort_values(["RID", "_date"])
        return tmp.drop_duplicates("RID", keep="first").drop(columns=["_date"])
    return df.drop_duplicates("RID", keep="first")

def clean_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def map_binary_target(series, high_values):
    x = series.astype(str).str.strip().str.lower()
    high = {str(v).lower() for v in high_values}
    y = x.map(lambda v: 1 if v in high else (0 if v not in {"", "nan", "none"} else pd.NA))
    return y.astype("Int64")
