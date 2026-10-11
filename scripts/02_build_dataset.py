import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from common import (
    load_config,
    processed_dir,
    read_prefix,
    ensure_rid,
    baseline_only,
    find_column,
)


# ============================================================
# VERIFIED ADNI VARIABLE NAMES
# ============================================================

ALIASES = {
    # ADNI PTDEMOG has no AGE column; age at baseline is derived from
    # date of birth (PTDOB "MM/YYYY", or PTDOBYY) and the baseline date.
    "age": ["AGE", "PTAGE", "Age"],

    "dob": ["PTDOB"],

    "dob_year": ["PTDOBYY"],

    "sex": [
        "PTGENDER",
        "SEX",
        "GENDER",
        "Sex",
    ],

    "education": [
        "PTEDUCAT",
        "EDUCATION",
        "YEARS_EDUCATION",
        "Education",
    ],

    # SES is not confirmed in the downloaded PTDEMOG file.
    "ses": [
        "PTSES",
        "SES",
        "SOCIOECONOMIC_STATUS",
        "Socioeconomic Status",
    ],

    # Verified ADNI fields
    "mmse": [
        "MMSCORE",
        "MMSE",
        "MMSE_TOTAL",
        "MMSE Score",
    ],

    "moca": [
        "MOCA",
        "MOCATOTS",
        "MOCA_TOTAL",
        "MoCA",
    ],

    # IMPORTANT:
    # TOTSCORE = ADAS 11 total (0-70)
    # TOTAL13 = ADAS 13 total (0-85)
    "adas": [
        "TOTSCORE",
        "ADAS11",
        "ADAS_TOTAL",
        "ADAS_COG",
        "ADAS",
    ],

    "faq": [
        "FAQTOTAL",
        "FAQ_TOTAL",
        "FAQ",
        "FAQTOTALSCORE",
    ],

    # Stage 3 MRI variables.
    # These are aliases only; we will verify whether they
    # actually exist in the downloaded MRI table.
    "eTIV": [
        "eTIV",
        "ETIV",
        "EstimatedTotalIntracranialVolume",
    ],

    "nWBV": [
        "nWBV",
        "NWBV",
        "NormalizedWholeBrainVolume",
    ],

    "ASF": [
        "ASF",
        "AtlasScalingFactor",
    ],

    "hippocampus_icv": [
        "HIPPOCAMPUS_ICV",
        "hippocampus_icv",
    ],

    # Stage 2 plasma biomarkers (ADNI UPenn Fujirebio/Quanterix table).
    "ptau217": [
        "pT217_F",
        "PTAU217",
        "PT217",
        "pTau217",
    ],

    "ab42_ab40": [
        "AB42_AB40_F",
        "AB42AB40",
        "AB4240",
        "ABETA42_40",
    ],

    # UPENN Quanterix NfL. The older Blennow-lab assay (PLASMA_NFL) is on a
    # different scale and is kept separately as nfl_legacy.
    "nfl": [
        "NfL_Q",
        "NFL",
    ],

    "ptau181": [
        "PLASMAPTAU181",
        "PTAU181",
    ],

    "nfl_legacy": [
        "PLASMA_NFL",
    ],

    "gds": [
        "GDTOTAL",
        "GDS",
    ],

    "gfap": [
        "GFAP_Q",
        "GFAP",
        "PLASMA_GFAP",
    ],

    # Stage 4 PET (ADNI UC Berkeley amyloid/tau summary tables).
    "amyloid_centiloid": [
        "CENTILOIDS",
        "CENTILOID",
        "AMYLOID_CENTILOID",
    ],

    "amyloid_suvr": [
        "SUMMARY_SUVR",
        "SUVR",
        "AMYLOID_SUVR",
    ],

    "tau_suvr": [
        "META_TEMPORAL_SUVR",
        "TAU_SUVR",
        "METATEMPORAL_SUVR",
    ],

    "dx": [
        "DX",
        "DXCURREN",
        "CURRENT_DIAGNOSIS",
        "Diagnosis",
        "DIAGNOSIS",
    ],
}


# ============================================================
# COLUMN HELPERS
# ============================================================

def pick(df, key, required=False):
    # Keys produced only by table-specific transforms have no aliases.
    if key not in ALIASES:
        return None
    return find_column(
        df,
        ALIASES[key],
        required=required
    )


def get_feature(df, key):
    c = pick(df, key, required=False)

    if c is None:
        return None

    return df[c]


# ADNI encodes "not done / not applicable" with negative sentinel codes
# (e.g. -1, -4). They must never reach the model as real values.
ADNI_MISSING_CODES = [-1, -4, -5, -9]

# Columns where negative values are genuine measurements, not missing codes.
SENTINEL_EXEMPT = {"amyloid_centiloid"}

# Plausible ranges; values outside are treated as missing.
VALID_RANGES = {
    "age": (40, 110),
    "education": (0, 30),
    "mmse": (0, 30),
    "moca": (0, 30),
    "adas": (0, 70),
    "faq": (0, 30),
    "eTIV": (900, 2300),
    "nWBV": (0.5, 0.95),
    "ASF": (0.7, 2.0),
    "hippocampus_icv": (0.001, 0.01),
    "bmi": (12, 60),
    "sbp": (76, 250),
    "gds": (0, 15),
    "ptau217": (0, 10),
    "ab42_ab40": (0.01, 0.3),
    "nfl": (0, 300),
    "gfap": (0, 3000),
    "ptau181": (0, 200),
    "nfl_legacy": (0, 500),
    "apoe4": (0, 2),
    "amyloid_centiloid": (-100, 300),
    "fdg": (0.5, 2.0),
}


def clean_clinical_values(data):
    """Replace ADNI sentinel codes and implausible values with NaN."""
    for column in data.columns:
        if column in ("RID", "target"):
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        sentinel = numeric.isin(ADNI_MISSING_CODES) & (column not in SENTINEL_EXEMPT)
        if sentinel.any():
            print(f"{column}: {int(sentinel.sum())} ADNI missing-code value(s) set to NaN")
        data.loc[sentinel, column] = np.nan
        if column in VALID_RANGES:
            low, high = VALID_RANGES[column]
            numeric = pd.to_numeric(data[column], errors="coerce")
            outside = numeric.notna() & ~numeric.between(low, high)
            if outside.any():
                print(f"{column}: {int(outside.sum())} implausible value(s) set to NaN")
            data.loc[outside, column] = np.nan
    # ASF is derived from eTIV; drop it whenever eTIV was rejected.
    if "eTIV" in data.columns and "ASF" in data.columns:
        data.loc[data["eTIV"].isna(), "ASF"] = np.nan
    return data


def parse_dob(df):
    """Date of birth from PTDOB ("MM/YYYY") or PTDOBYY (year or date)."""
    dob = pd.Series(pd.NaT, index=df.index)
    c = pick(df, "dob")
    if c is not None:
        dob = pd.to_datetime(df[c].astype(str), format="%m/%Y", errors="coerce")
    c = pick(df, "dob_year")
    if c is not None:
        raw = df[c].astype(str).str.strip()
        year_only = pd.to_datetime(raw, format="%Y", errors="coerce")
        full = pd.to_datetime(raw, errors="coerce")
        # mid-year when only the year is known
        dob = dob.fillna(full).fillna(year_only + pd.Timedelta(days=182))
    return dob


# ============================================================
# BASELINE DEMOGRAPHICS
# ============================================================

def prepare_static_demographics():

    df = baseline_only(
        read_prefix("PTDEMOG_")
    )

    df = ensure_rid(df)

    out = pd.DataFrame({
        "RID": df["RID"]
    })

    # Only variables actually found in the downloaded file
    for key in ["age", "sex", "education", "ses"]:

        s = get_feature(df, key)

        if s is not None:
            out[key] = s.values
        else:
            out[key] = np.nan

    out["_dob"] = parse_dob(df).values

    return out.drop_duplicates("RID")


# ============================================================
# COGNITIVE ASSESSMENTS
# ============================================================

def prepare_cognitive():

    out = None

    files = [
        ("MMSE_", "mmse"),
        ("MOCA_", "moca"),
        ("ADAS_", "adas"),
        ("FAQ_", "faq"),
    ]

    for prefix, key in files:

        try:

            df = baseline_only(
                read_prefix(prefix)
            )

            df = ensure_rid(df)

            c = pick(
                df,
                key,
                required=False
            )

            if c is None:

                print(
                    f"{prefix}: "
                    f"variable for {key} not found; skipping."
                )

                continue

            part = df[
                ["RID", c]
            ].copy()

            part[key] = pd.to_numeric(
                part[c],
                errors="coerce"
            )

            part = part[
                ["RID", key]
            ].drop_duplicates("RID")

            if out is None:

                out = part

            else:

                out = out.merge(
                    part,
                    on="RID",
                    how="outer"
                )

        except FileNotFoundError:

            print(
                f"{prefix}: "
                f"file not found; skipping."
            )

    if out is None:

        raise RuntimeError(
            "No cognitive assessment "
            "file was available."
        )

    return out


# ============================================================
# LATER-STAGE MODALITIES — STAGES 2, 3 AND 4
# ============================================================

# Each pathway stage adds one modality. A modality may be spread over
# several ADNI tables (e.g. amyloid and tau PET); all matching tables are
# merged on RID.
# ------------------------------------------------------------
# Table-specific transforms. Each returns (features, dates) where
# features has RID + feature columns and dates is the measurement date
# per row (None for time-invariant data such as genotype).
# ------------------------------------------------------------

def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def transform_apoe(df):
    """APOE genotype ("3/4") -> number of e4 alleles (0, 1, 2). Time-invariant."""
    genotype = df["GENOTYPE"].astype(str)
    valid = genotype.str.match(r"^[234]/[234]$")
    out = pd.DataFrame({"RID": df["RID"], "apoe4": genotype.str.count("4").where(valid)})
    return out.dropna(subset=["apoe4"]), None


def transform_fdg(df):
    """UC Berkeley FDG (long format): MetaROI mean = cortical glucose metabolism."""
    df = df[df["ROINAME"].astype(str).str.lower() == "metaroi"]
    return pd.DataFrame({"RID": df["RID"], "fdg": _num(df, "MEAN")}), df["EXAMDATE"]


def transform_amyloid(df):
    """UC Berkeley amyloid PET. Centiloid is harmonised across tracers
    (florbetapir, florbetaben, NAV4694); tracer-specific SUVR is not."""
    if "qc_flag" in df.columns:
        qc = pd.to_numeric(df["qc_flag"], errors="coerce")
        df = df[~qc.isin([0, -2])]  # 0 = fail, -2 = cannot be assessed
    out = pd.DataFrame({"RID": df["RID"]})
    for key in ["amyloid_centiloid", "amyloid_suvr"]:
        c = pick(df, key)
        if c is not None:
            out[key] = pd.to_numeric(df[c], errors="coerce")
    return out, df[find_column(df, ["SCANDATE", "EXAMDATE"])]


def transform_vitals(df):
    """BMI (units: weight 1=lb 2=kg; height 1=in 2=cm) and systolic BP."""
    weight = _num(df, "VSWEIGHT")
    weight = weight.where(_num(df, "VSWTUNIT") != 1, weight * 0.45359237)
    height = _num(df, "VSHEIGHT")
    unit = _num(df, "VSHTUNIT")
    # Height unit is often unrecorded after screening; infer it from the value.
    inches = (unit == 1) | (~unit.isin([1, 2]) & height.between(48, 84))
    height_cm = height.where(~inches, height * 2.54)
    height_cm = height_cm.where(height_cm.between(120, 215))
    bmi = weight / (height_cm / 100) ** 2
    return pd.DataFrame({"RID": df["RID"], "bmi": bmi, "sbp": _num(df, "VSBPSYS")}), df["VISDATE"]


MEDHIST_FLAGS = {
    "MH4CARD": "hx_cardiovascular",
    "MH9ENDO": "hx_endocrine",
    "MHPSYCH": "hx_psychiatric",
    "MH2NEURL": "hx_neurologic",
    "MH16SMOK": "hx_smoking",
}


def transform_medhist(df):
    """Baseline medical history (comorbidities), 1 = yes, 0 = no."""
    out = pd.DataFrame({"RID": df["RID"]})
    for column, name in MEDHIST_FLAGS.items():
        out[name] = _num(df, column)
    return out, df["VISDATE"]


TRANSFORMS = {
    "APOERES_": transform_apoe,
    "UCBERKELEYFDG_": transform_fdg,
    "UCBERKELEY_AMY_6MM_": transform_amyloid,
    "VITALS_": transform_vitals,
    "MEDHIST_": transform_medhist,
}


# Each pathway stage adds one modality. A modality may be spread over
# several ADNI tables; all matching tables are merged on RID (the first
# table providing a feature wins).
MODALITIES = {
    "clinical": {
        "prefixes": ["VITALS_", "GDSCALE_", "MEDHIST_"],
        "keys": ["bmi", "sbp", "gds", *MEDHIST_FLAGS.values()],
    },
    "plasma": {
        "prefixes": [
            "UPENN_PLASMA_FUJIREBIO_QUANTERIX_",
            "UGOTPTAU181_",
            "ADNI_BLENNOWPLASMANFLLONG_",
            "PLASMA_",
            "APOERES_",
        ],
        "keys": ["ptau217", "ab42_ab40", "nfl", "gfap", "ptau181", "nfl_legacy", "apoe4"],
        # A participant reaches Stage 2 only with an actual blood biomarker;
        # genotype alone (available for nearly everyone) does not count.
        "inclusion_keys": ["ptau217", "ab42_ab40", "nfl", "gfap", "ptau181", "nfl_legacy"],
    },
    "mri": {
        "prefixes": ["All_Subjects_Key_MRI_", "UCSFFSX7_"],
        "keys": ["eTIV", "nWBV", "ASF", "hippocampus_icv"],
    },
    "pet": {
        "prefixes": ["UCBERKELEY_AMY_6MM_", "UCBERKELEYFDG_"],
        "keys": ["amyloid_centiloid", "amyloid_suvr", "fdg"],
        "inclusion_keys": ["amyloid_centiloid", "fdg"],
    },
}

# Modality that a participant must have to be included at each stage.
# Earlier optional modalities are left-joined (missing values are handled
# by the model's imputation + missingness indicators).
STAGE_MODALITIES = {
    "stage1": {"required": None, "optional": ["clinical"]},
    "stage2": {"required": "plasma", "optional": ["clinical"]},
    "stage3": {"required": "mri", "optional": ["clinical", "plasma"]},
    "stage4": {"required": "pet", "optional": ["clinical", "plasma", "mri"]},
}


# FreeSurfer (UCSFFSX*) codes from the ADNI data dictionary.
FS_ICV = "ST10CV"
FS_HIPPOCAMPUS = ["ST29SV", "ST88SV"]
# FreeSurfer 7 summary volumes: total gray matter (cortical + subcortical +
# cerebellar GM), cerebral white matter, cerebellar WM L/R, brainstem.
FS7_BRAIN = ["ST154SV", "ST152SV", "ST18SV", "ST77SV", "ST1SV"]
# Older FreeSurfer tables: sum of the individual structures.
FS_BRAIN = [
    "ST19SV", "ST78SV",   # cerebral cortex L/R (cortical gray matter)
    "ST20SV", "ST79SV",   # cerebral white matter L/R
    "ST17SV", "ST76SV",   # cerebellum cortex L/R
    "ST18SV", "ST77SV",   # cerebellum white matter L/R
    "ST1SV",              # brainstem
    "ST61SV", "ST120SV",  # thalamus
    "ST16SV", "ST75SV",   # caudate
    "ST53SV", "ST112SV",  # putamen
    "ST42SV", "ST101SV",  # pallidum
    "ST29SV", "ST88SV",   # hippocampus
    "ST12SV", "ST71SV",   # amygdala
    "ST11SV", "ST70SV",   # accumbens
    "ST65SV", "ST124SV",  # ventral DC
]


def derive_freesurfer(df):
    """OASIS-style MRI measures from ADNI FreeSurfer volumes (mm^3).

    eTIV  = intracranial volume (cm^3)
    nWBV  = whole-brain tissue volume / intracranial volume
    ASF   = 1755 / eTIV  (atlas scaling factor, Buckner et al. 2004)
    hippocampus_icv = (left + right hippocampus) / intracranial volume
    """
    if FS_ICV not in df.columns:
        return None
    if "OVERALLQC" in df.columns:
        df = df[df["OVERALLQC"].astype(str).str.lower() != "fail"]
    icv = pd.to_numeric(df[FS_ICV], errors="coerce")
    out = pd.DataFrame({"RID": df["RID"], "eTIV": icv / 1000.0})
    out["ASF"] = 1755.0 / out["eTIV"]
    for components in (FS7_BRAIN, FS_BRAIN):
        if all(c in df.columns for c in components):
            brain = df[components].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(components))
            out["nWBV"] = brain / icv
            break
    if all(c in df.columns for c in FS_HIPPOCAMPUS):
        hippocampus = df[FS_HIPPOCAMPUS].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=2)
        out["hippocampus_icv"] = hippocampus / icv
    return out


def nearest_to_baseline(part, dates, baseline_dates, window_days):
    """Keep, per participant, the measurement closest to the baseline date
    within +/- window_days (avoids using scans from years later)."""
    part = part.assign(_date=pd.to_datetime(dates, errors="coerce").values)
    part = part.merge(baseline_dates, on="RID", how="inner").dropna(subset=["_date"])
    part["_delta"] = (part["_date"] - part["baseline_date"]).dt.days.abs()
    part = part[part["_delta"] <= window_days]
    part = part.sort_values(["RID", "_delta"]).drop_duplicates("RID")
    return part.drop(columns=["_date", "_delta", "baseline_date"])


def prepare_modality(name, baseline_dates=None, window_days=180, required=False):

    spec = MODALITIES[name]

    result = None

    for prefix in spec["prefixes"]:

        try:
            raw = ensure_rid(read_prefix(prefix))
        except FileNotFoundError:
            continue

        transform = next((f for p, f in TRANSFORMS.items() if prefix.startswith(p)), None)
        if transform is not None:
            part, dates = transform(raw)
            if dates is None:  # time-invariant (e.g. genotype)
                part = part.drop_duplicates("RID")
            elif baseline_dates is not None:
                part = nearest_to_baseline(part, dates.loc[part.index], baseline_dates, window_days)
            else:
                part = part.loc[baseline_only(raw.loc[part.index]).index].drop_duplicates("RID")
            print(f"{prefix}: {part['RID'].nunique():,} participants")
            result = part if result is None else result.merge(
                part.drop(columns=[c for c in part.columns if c in result.columns and c != "RID"]),
                on="RID", how="outer")
            continue

        date_col = find_column(raw, ["EXAMDATE", "SCANDATE", "VISDATE", "EXAMDATE2", "USERDATE"], required=False)
        aligned = baseline_dates is not None and date_col is not None
        df = raw if aligned else baseline_only(raw)

        part = pd.DataFrame({"RID": df["RID"]})

        found = 0

        for key in spec["keys"]:

            c = pick(df, key, required=False)

            if c is not None:
                part[key] = pd.to_numeric(df[c], errors="coerce")
                found += 1

        if not found and name == "mri":
            derived = derive_freesurfer(df)
            if derived is not None:
                print(f"{prefix}: deriving eTIV/nWBV/ASF/hippocampus_icv from FreeSurfer volumes")
                part = derived
                df = df.loc[derived.index]
                found = len(derived.columns) - 1

        if not found:
            continue

        if aligned:
            part = nearest_to_baseline(part, df[date_col], baseline_dates, window_days)
            print(f"{prefix}: {part['RID'].nunique():,} participants with a measurement "
                  f"within {window_days} days of baseline")
        else:
            part = part.drop_duplicates("RID")

        if result is None:
            result = part
        else:
            overlap = [c for c in part.columns if c in result.columns and c != "RID"]
            result = result.merge(part.drop(columns=overlap), on="RID", how="outer")

    if result is None:
        raise RuntimeError(
            f"No {name} table with any of {spec['keys']} was found "
            f"(looked for prefixes {spec['prefixes']})."
        )

    # When this modality defines stage membership, keep only participants
    # with an actual measurement (e.g. a plasma biomarker, not genotype alone).
    inclusion = [k for k in spec.get("inclusion_keys", []) if k in result.columns]
    if required and inclusion:
        result = result[result[inclusion].notna().any(axis=1)]

    return result


def prepare_mri(baseline_dates=None, window_days=180):
    return prepare_modality("mri", baseline_dates, window_days)


# ============================================================
# ADNI DIAGNOSIS MAPPING
# ============================================================

def diagnosis_binary_numeric(series):

    """
    ADNI DXSUM DIAGNOSIS mapping confirmed
    from the downloaded DXSUM file:

        1 = normal / cognitively normal
        2 = MCI
        3 = Alzheimer's dementia
        10 = other / unclear

    For progression:

        1 -> 0
        2 -> 0
        3 -> 1
        10 -> missing
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="Int64"
    )

    result.loc[numeric == 1] = 0
    result.loc[numeric == 2] = 0
    result.loc[numeric == 3] = 1

    return result


# ============================================================
# TARGET
# ============================================================

def prepare_target():

    cfg = load_config()

    mode = cfg["target"].get(
        "mode",
        "progression_24m"
    )

    raw = ensure_rid(
        read_prefix("DXSUM_")
    )

    dxcol = pick(
        raw,
        "dx",
        required=True
    )

    # --------------------------------------------------------
    # BASELINE DIAGNOSIS MODE
    # --------------------------------------------------------

    if mode == "baseline_diagnosis":

        df = baseline_only(raw)

        numeric_dx = pd.to_numeric(
            df[dxcol],
            errors="coerce"
        )

        target = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int64"
        )

        # Normal/CN and MCI -> 0
        target.loc[
            numeric_dx.isin([1, 2])
        ] = 0

        # Alzheimer's dementia -> 1
        target.loc[
            numeric_dx == 3
        ] = 1

        datecol = find_column(df, ["EXAMDATE", "EXAMDATE2", "USERDATE"], required=False)

        return pd.DataFrame({
            "RID": df["RID"],
            "target": target,
            "baseline_date": pd.to_datetime(df[datecol], errors="coerce") if datecol else pd.NaT,
        }).drop_duplicates("RID")


    # --------------------------------------------------------
    # PROSPECTIVE 24-MONTH CONVERSION MODE
    # --------------------------------------------------------

    if mode == "progression_24m":

        datecol = find_column(
            raw,
            [
                "EXAMDATE",
                "EXAMDATE2",
                "USERDATE"
            ],
            required=False
        )

        if datecol is None:

            raise RuntimeError(
                "progression_24m requires "
                "EXAMDATE."
            )

        # Convert examination date
        raw["_date"] = pd.to_datetime(
            raw[datecol],
            errors="coerce"
        )

        raw = raw.dropna(
            subset=["_date"]
        ).copy()

        raw = raw.sort_values(
            ["RID", "_date"]
        )

        # Numeric ADNI diagnosis
        raw["_diagnosis"] = pd.to_numeric(
            raw[dxcol],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Determine each participant's baseline
        # ----------------------------------------------------

        baseline = (
            raw.sort_values(
                ["RID", "_date"]
            )
            .groupby("RID", as_index=False)
            .first()
        )

        baseline = baseline[
            [
                "RID",
                "_date",
                "_diagnosis"
            ]
        ].rename(
            columns={
                "_date": "baseline_date",
                "_diagnosis": "baseline_diagnosis",
            }
        )

        # Only baseline CN or MCI
        baseline = baseline[
            baseline["baseline_diagnosis"].isin(
                [1, 2]
            )
        ].copy()

        # ----------------------------------------------------
        # Merge baseline information back onto
        # every longitudinal diagnosis row
        # ----------------------------------------------------

        merged = raw.merge(
            baseline,
            on="RID",
            how="inner"
        )

        merged["months_after"] = (
            merged["_date"]
            - merged["baseline_date"]
        ).dt.days / 30.4375

        # ----------------------------------------------------
        # Require >= 24 months of actual follow-up
        # ----------------------------------------------------

        last_date = (
            raw.groupby("RID")["_date"]
            .max()
            .rename("last_date")
        )

        baseline = baseline.merge(
            last_date,
            on="RID",
            how="left"
        )

        baseline["followup_months"] = (
            baseline["last_date"]
            - baseline["baseline_date"]
        ).dt.days / 30.4375

        baseline = baseline[
            baseline["followup_months"] >= 24
        ].copy()

        # ----------------------------------------------------
        # Identify dementia conversion WITHIN first 24 months
        # ----------------------------------------------------

        events = merged[
            (merged["months_after"] > 0)
            &
            (merged["months_after"] <= 24)
            &
            (merged["_diagnosis"] == 3)
        ]

        dementia_rids = set(
            events["RID"].unique()
        )

        baseline["target"] = (
            baseline["RID"]
            .isin(dementia_rids)
            .astype(int)
        )

        # ----------------------------------------------------
        # Final target table
        # ----------------------------------------------------

        out = baseline[
            [
                "RID",
                "target",
                "baseline_date",
            ]
        ].copy()

        return out.drop_duplicates(
            "RID"
        )


    raise ValueError(
        f"Unsupported target mode: {mode}"
    )


# ============================================================
# MAIN DATASET CONSTRUCTION
# ============================================================

def main(stage):

    cfg = load_config()

    processed_dir().mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n====================================")
    print("BUILDING DATASET")
    print("====================================")

    print(f"Stage: {stage}")

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    print("\nLoading demographics...")

    demo = prepare_static_demographics()

    print(
        f"Demographic participants: "
        f"{demo['RID'].nunique():,}"
    )

    # --------------------------------------------------------
    # Cognitive assessments
    # --------------------------------------------------------

    print("\nLoading cognitive assessments...")

    cog = prepare_cognitive()

    print(
        f"Cognitive participants: "
        f"{cog['RID'].nunique():,}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    print("\nBuilding target...")

    target = prepare_target()

    print(
        f"Target participants: "
        f"{target['RID'].nunique():,}"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    data = (
        demo
        .merge(
            cog,
            on="RID",
            how="inner"
        )
        .merge(
            target,
            on="RID",
            how="inner"
        )
    )

    # --------------------------------------------------------
    # Stage 2-4 modalities
    # --------------------------------------------------------

    stage_spec = STAGE_MODALITIES[stage]

    baseline_dates = data[["RID", "baseline_date"]].dropna()
    window_days = int(cfg["data"].get("modality_window_days", 180))

    for name in stage_spec["optional"]:

        try:
            part = prepare_modality(name, baseline_dates, window_days)
        except RuntimeError as exc:
            print(f"\nOptional {name} data unavailable: {exc}")
            continue

        print(f"\nOptional {name} participants: {part['RID'].nunique():,}")

        data = data.merge(part, on="RID", how="left")

    if stage_spec["required"]:

        name = stage_spec["required"]

        print(f"\nLoading {name} variables...")

        part = prepare_modality(name, baseline_dates, window_days, required=True)

        print(f"{name} participants: {part['RID'].nunique():,}")

        data = data.merge(part, on="RID", how="inner")

    # --------------------------------------------------------
    # Clean target
    # --------------------------------------------------------

    data = data.dropna(
        subset=["target"]
    ).copy()

    data["target"] = (
        data["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Age at baseline (derived when no AGE column exists)
    # --------------------------------------------------------

    derived_age = (data["baseline_date"] - data["_dob"]).dt.days / 365.25
    data["age"] = pd.to_numeric(data["age"], errors="coerce").fillna(derived_age).round(1)
    data = data.drop(columns=["_dob", "baseline_date"])

    # --------------------------------------------------------
    # ADNI missing codes / implausible values -> NaN
    # --------------------------------------------------------

    data = clean_clinical_values(data)

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        c for c in data.columns if c not in ("RID", "target", "sex", "ses")
    ]

    for c in numeric_columns:

        if c in data.columns:

            data[c] = pd.to_numeric(
                data[c],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Categorical conversion
    # --------------------------------------------------------

    for c in [
        "sex",
        "ses"
    ]:

        if c in data.columns:

            data[c] = (
                data[c]
                .astype("string")
                .str.strip()
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    out_path = (
        processed_dir()
        / f"{stage}.csv"
    )

    data.to_csv(
        out_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nSaved dataset:"
        f"\n{out_path}"
    )

    print(
        f"\nRows: "
        f"{len(data):,}"
    )

    print(
        f"Participants: "
        f"{data['RID'].nunique():,}"
    )

    print(
        "\nColumns:"
    )

    print(
        list(data.columns)
    )

    print(
        "\nTarget distribution:"
    )

    print(
        data["target"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTarget percentage:"
    )

    print(
        (
            data["target"]
            .value_counts(
                normalize=True
            )
            * 100
        )
        .sort_index()
        .round(2)
    )

    print(
        "\nMissingness (%):"
    )

    missingness = (
        data
        .isna()
        .mean()
        * 100
    )

    print(
        missingness
        .sort_values(
            ascending=False
        )
        .round(2)
    )

    print(
        "\n===================================="
    )

    print(
        "DATASET BUILD COMPLETE"
    )

    print(
        "====================================\n"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stage",
        choices=list(STAGE_MODALITIES),
        required=True
    )

    args = ap.parse_args()

    main(args.stage)