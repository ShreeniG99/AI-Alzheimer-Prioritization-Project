"""
Generate SYNTHETIC ADNI-shaped CSV files.

ADNI subject-level data is access-controlled and must never be committed to
the repository. This script writes small, entirely artificial tables with the
same file prefixes and column names that 02_build_dataset.py expects, so that:

- the full pipeline (build -> train -> evaluate -> API -> dashboard) can be
  demonstrated and tested without ADNI credentials;
- continuous integration can exercise every stage of the progressive pathway.

Every number produced here is simulated from a simple latent-severity model.
Results obtained on this data say NOTHING about real patients and must never
be reported as research findings.

Usage:
    python scripts/00_make_synthetic_data.py --out data/raw_synthetic --subjects 2000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VISIT_MONTHS = [0, 6, 12, 24, 36]
VISIT_CODES = {0: "bl", 6: "m06", 12: "m12", 24: "m24", 36: "m36"}
BASE_DATE = pd.Timestamp("2010-01-01")
SUFFIX = "SYNTHETIC"


def simulate(n_subjects: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rid = np.arange(1, n_subjects + 1)

    # Baseline diagnosis: 1 = CN, 2 = MCI, 3 = AD dementia (ADNI DXSUM coding).
    baseline_dx = rng.choice([1, 2, 3], size=n_subjects, p=[0.38, 0.47, 0.15])

    # Latent neurodegeneration severity drives every simulated measurement.
    severity = (
        np.select([baseline_dx == 1, baseline_dx == 2], [-1.0, 0.3], 1.6)
        + rng.normal(0, 0.7, n_subjects)
    )

    # Underlying amyloid/tau pathology: only partly visible in cognitive
    # scores, measured more directly by plasma biomarkers and PET. This makes
    # later pathway stages genuinely more informative, as intended clinically.
    pathology = severity + rng.normal(0, 0.9, n_subjects)

    sex = rng.choice([1, 2], size=n_subjects)
    education = np.clip(np.round(rng.normal(15.5, 2.8, n_subjects)), 6, 20)

    mmse = np.clip(np.round(28.5 - 2.4 * np.maximum(severity + 0.8, 0)
                            + 0.12 * (education - 15) + rng.normal(0, 1.0, n_subjects)), 0, 30)
    adas = np.clip(6 + 5.0 * (severity + 1.2) + rng.normal(0, 2.5, n_subjects), 0, 70).round(2)
    faq = np.clip(np.round(np.maximum(severity, 0) ** 1.6 * 5 + rng.normal(0, 1.2, n_subjects)), 0, 30)
    moca = np.clip(np.round(26 - 3.0 * np.maximum(severity + 0.7, 0) + rng.normal(0, 1.4, n_subjects)), 0, 30)

    # Stage 2: plasma biomarkers.
    ptau217 = np.exp(rng.normal(-1.2 + 0.55 * pathology, 0.35, n_subjects)).round(4)
    ab42_ab40 = np.clip(0.105 - 0.009 * pathology + rng.normal(0, 0.006, n_subjects), 0.04, 0.16).round(4)
    nfl = np.exp(rng.normal(2.9 + 0.22 * severity, 0.3, n_subjects)).round(2)
    gfap = np.exp(rng.normal(4.8 + 0.25 * severity, 0.3, n_subjects)).round(2)

    # Stage 3: MRI-derived volumes (OASIS-style names).
    etiv = rng.normal(1500, 160, n_subjects).round(0)
    asf = (1755.0 / etiv).round(3)
    nwbv = np.clip(0.75 - 0.018 * severity + rng.normal(0, 0.012, n_subjects), 0.62, 0.85).round(3)

    ptau181 = np.exp(rng.normal(2.6 + 0.35 * pathology, 0.35, n_subjects)).round(2)
    nfl_legacy = np.exp(rng.normal(3.6 + 0.2 * severity, 0.3, n_subjects)).round(1)
    # APOE e4 carriers have more pathology on average.
    apoe4 = np.clip(np.round(rng.normal(0.45 + 0.35 * pathology, 0.5, n_subjects)), 0, 2).astype(int)

    # Stage 4: amyloid PET (Centiloid) and FDG-PET metabolism.
    centiloid = (20 + 32 * pathology + rng.normal(0, 14, n_subjects)).round(1)
    amyloid_suvr = (1.0 + 0.0059 * centiloid + rng.normal(0, 0.02, n_subjects)).round(3)
    fdg = np.clip(1.30 - 0.05 * severity + rng.normal(0, 0.05, n_subjects), 0.8, 1.6).round(4)

    # Stage 1 comorbidities (weak, realistic associations).
    sbp = np.clip(rng.normal(135, 17, n_subjects), 90, 210).round(0)
    weight_lb = np.clip(rng.normal(170, 30, n_subjects), 95, 320).round(1)
    height_in = np.clip(rng.normal(66, 4, n_subjects), 56, 78).round(1)
    gds = np.clip(np.round(rng.normal(1.3 + 0.4 * np.maximum(severity, 0), 1.3, n_subjects)), 0, 15)
    hx = {name: (rng.random(n_subjects) < p).astype(int) for name, p in
          [("MH4CARD", 0.6), ("MH9ENDO", 0.4), ("MHPSYCH", 0.3), ("MH2NEURL", 0.15), ("MH16SMOK", 0.4)]}

    # Conversion to AD dementia over follow-up for CN/MCI participants.
    logit = -3.6 + 0.9 * severity + 1.3 * pathology + 0.6 * (baseline_dx == 2)
    p_convert_24m = 1 / (1 + np.exp(-logit))
    converts = (rng.random(n_subjects) < p_convert_24m) & (baseline_dx != 3)
    conversion_month = np.where(converts, rng.choice([6, 12, 24], size=n_subjects), 999)
    # Some participants drop out before 24 months.
    last_month = rng.choice([12, 24, 36], size=n_subjects, p=[0.10, 0.15, 0.75])

    subjects = pd.DataFrame({
        "RID": rid, "baseline_dx": baseline_dx, "sex": sex, "education": education,
        "mmse": mmse, "adas": adas, "faq": faq, "moca": moca,
        "ptau217": ptau217, "ab42_ab40": ab42_ab40, "nfl": nfl, "gfap": gfap,
        "etiv": etiv, "asf": asf, "nwbv": nwbv,
        "centiloid": centiloid, "amyloid_suvr": amyloid_suvr, "fdg": fdg,
        "conversion_month": conversion_month, "last_month": last_month,
    })

    def drop_random(series, fraction):
        mask = rng.random(len(series)) < fraction
        return series.mask(mask)

    bl = pd.DataFrame({"RID": rid, "VISCODE": "bl", "VISCODE2": "bl",
                       "EXAMDATE": BASE_DATE.strftime("%Y-%m-%d")})

    tables = {}
    age = np.clip(rng.normal(73, 7, n_subjects), 55, 92)
    birth = [BASE_DATE - pd.Timedelta(days=int(a * 365.25)) for a in age]
    ptdemog = bl.assign(
        PTGENDER=sex,
        PTEDUCAT=education,
        PTDOB=[b.strftime("%m/%Y") for b in birth],
        PTDOBYY=[b.strftime("%Y-01-01") for b in birth],
    )
    # ADNI uses negative sentinel codes for missing values; include a few.
    ptdemog.loc[rng.random(n_subjects) < 0.01, "PTEDUCAT"] = -4
    tables["PTDEMOG"] = ptdemog
    tables["MMSE"] = bl.assign(MMSCORE=drop_random(pd.Series(mmse), 0.01))
    tables["ADAS"] = bl.assign(TOTSCORE=drop_random(pd.Series(adas), 0.03))
    faq_series = drop_random(pd.Series(faq), 0.03)
    faq_series[rng.random(n_subjects) < 0.005] = -1
    tables["FAQ"] = bl.assign(FAQTOTAL=faq_series)
    tables["MOCA"] = bl.assign(MOCA=drop_random(pd.Series(moca), 0.65))

    visit = bl.assign(VISDATE=bl["EXAMDATE"])
    vitals = visit.assign(VSWEIGHT=weight_lb, VSWTUNIT=1, VSHEIGHT=height_in, VSHTUNIT=1, VSBPSYS=sbp)
    vitals.loc[rng.random(n_subjects) < 0.3, "VSHTUNIT"] = -4  # unit often unrecorded in ADNI
    tables["VITALS"] = vitals
    tables["GDSCALE"] = visit.assign(GDTOTAL=gds)
    medhist = visit.assign(**hx)
    tables["MEDHIST"] = medhist[rng.random(n_subjects) < 0.7]
    genotype = np.select([apoe4 == 0, apoe4 == 1], ["3/3", "3/4"], "4/4")
    tables["APOERES"] = bl[["RID"]].assign(VISCODE="sc", GENOTYPE=genotype, USERDATE="2010-02-01")

    # Not every participant reaches every pathway stage.
    def subset(fraction):
        return rng.random(n_subjects) < fraction

    plasma = subset(0.70)
    upenn = plasma & subset(0.6)
    tables["UPENN_PLASMA_FUJIREBIO_QUANTERIX"] = bl[upenn].assign(
        pT217_F=ptau217[upenn], AB42_AB40_F=ab42_ab40[upenn],
        NfL_Q=nfl[upenn], GFAP_Q=gfap[upenn],
    )
    legacy = plasma & ~upenn
    tables["UGOTPTAU181"] = bl[legacy].assign(PLASMAPTAU181=ptau181[legacy])
    tables["ADNI_BLENNOWPLASMANFLLONG"] = bl[legacy].assign(PLASMA_NFL=nfl_legacy[legacy])
    mri = subset(0.85)
    tables["All_Subjects_Key_MRI"] = bl[mri].assign(eTIV=etiv[mri], nWBV=nwbv[mri], ASF=asf[mri])
    pet = subset(0.60)
    amyloid = pet & subset(0.8)
    tables["UCBERKELEY_AMY_6MM"] = bl[amyloid].assign(
        SCANDATE=BASE_DATE.strftime("%Y-%m-%d"), qc_flag=2,
        CENTILOIDS=centiloid[amyloid], SUMMARY_SUVR=amyloid_suvr[amyloid],
    )
    fdg_rows = bl[pet][["RID", "VISCODE", "EXAMDATE"]]
    tables["UCBERKELEYFDG_8mm"] = pd.concat([
        fdg_rows.assign(ROINAME="MetaROI", MEAN=fdg[pet]),
        fdg_rows.assign(ROINAME="Top50PonsVermis", MEAN=1.0),
    ])

    # Longitudinal diagnosis summary.
    rows = []
    for s in subjects.itertuples(index=False):
        for month in VISIT_MONTHS:
            if month > s.last_month:
                break
            if s.baseline_dx == 3 or month >= s.conversion_month:
                dx = 3
            else:
                dx = s.baseline_dx
            date = BASE_DATE + pd.DateOffset(months=month)
            rows.append({"RID": s.RID, "VISCODE": VISIT_CODES[month], "VISCODE2": VISIT_CODES[month],
                         "EXAMDATE": date.strftime("%Y-%m-%d"), "DIAGNOSIS": dx})
    tables["DXSUM"] = pd.DataFrame(rows)

    # Longitudinal CDR Sum of Boxes (TEMPO target): baseline level and
    # yearly rate both rise with severity/pathology; converters speed up.
    cdr_rows = []
    cdr_base = np.clip(np.maximum(severity + 0.6, 0) * 1.2 + rng.normal(0, 0.4, n_subjects), 0, 18)
    cdr_rate = np.maximum(0, 0.15 + 0.35 * severity + 0.30 * pathology + rng.normal(0, 0.3, n_subjects))
    for i, s in enumerate(subjects.itertuples(index=False)):
        for month in VISIT_MONTHS:
            if month > s.last_month:
                break
            boost = 0.8 * max(0, month - s.conversion_month) / 12
            value = cdr_base[i] + cdr_rate[i] * month / 12 + boost + rng.normal(0, 0.25)
            date = BASE_DATE + pd.DateOffset(months=month)
            cdr_rows.append({"RID": s.RID, "VISCODE2": VISIT_CODES[month], "VISDATE": date.strftime("%Y-%m-%d"),
                             "CDRSB": round(float(np.clip(value, 0, 18)) * 2) / 2})
    tables["CDR"] = pd.DataFrame(cdr_rows)

    return tables


def main(out_dir: Path, n_subjects: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = simulate(n_subjects, seed)
    for prefix, df in tables.items():
        path = out_dir / f"{prefix}_{SUFFIX}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(df):,} rows)")
    (out_dir / "README_SYNTHETIC.txt").write_text(
        "These CSV files are SYNTHETIC and were generated by "
        "scripts/00_make_synthetic_data.py.\nThey contain no real "
        "participant data and must not be used for research conclusions.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw_synthetic"))
    ap.add_argument("--subjects", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.out, args.subjects, args.seed)
