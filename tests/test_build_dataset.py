"""Unit tests for the real-ADNI handling in 02_build_dataset.py."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT

spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "scripts" / "02_build_dataset.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_adni_missing_codes_and_implausible_values_become_nan():
    data = pd.DataFrame({
        "RID": [1, 2, 3, 4],
        "target": [0, 1, 0, 0],
        "faq": [-1, 5, 3, 31],
        "education": [-4, 16, 12, 14],
        "sex": [1, 2, -4, 1],
        "eTIV": [1500, 5000, 1400, 1600],
        "ASF": [1.17, 0.35, 1.25, 1.10],
    })
    cleaned = build.clean_clinical_values(data.copy())
    assert np.isnan(cleaned.loc[0, "faq"]) and np.isnan(cleaned.loc[3, "faq"])
    assert np.isnan(cleaned.loc[0, "education"])
    assert np.isnan(cleaned.loc[2, "sex"])
    # Implausible eTIV removed, and the ASF derived from it with it.
    assert np.isnan(cleaned.loc[1, "eTIV"]) and np.isnan(cleaned.loc[1, "ASF"])
    assert cleaned.loc[1, "faq"] == 5 and cleaned["target"].tolist() == [0, 1, 0, 0]


def test_dob_parsing_prefers_month_year():
    df = pd.DataFrame({"PTDOB": ["04/1931", None], "PTDOBYY": ["1931-01-01", "1944-01-01"]})
    dob = build.parse_dob(df)
    assert dob[0] == pd.Timestamp("1931-04-01")
    assert dob[1] == pd.Timestamp("1944-01-01")


def test_freesurfer7_derivation():
    volumes = {
        "RID": [1, 2],
        "OVERALLQC": ["Pass", "Fail"],
        "ST10CV": [1_500_000.0, 1_500_000.0],
        "ST154SV": [560_000.0, 560_000.0],
        "ST152SV": [430_000.0, 430_000.0],
        "ST18SV": [14_000.0, 14_000.0],
        "ST77SV": [14_000.0, 14_000.0],
        "ST1SV": [22_000.0, 22_000.0],
        "ST29SV": [3_500.0, 3_500.0],
        "ST88SV": [3_600.0, 3_600.0],
    }
    out = build.derive_freesurfer(pd.DataFrame(volumes))
    assert out["RID"].tolist() == [1]  # failed QC excluded
    row = out.iloc[0]
    assert row["eTIV"] == pytest.approx(1500.0)
    assert row["ASF"] == pytest.approx(1755 / 1500)
    assert row["nWBV"] == pytest.approx(1_040_000 / 1_500_000)
    assert row["hippocampus_icv"] == pytest.approx(7_100 / 1_500_000)


def test_nearest_to_baseline_uses_closest_scan_within_window():
    part = pd.DataFrame({"RID": [1, 1, 1, 2], "eTIV": [1400.0, 1500.0, 1600.0, 1450.0]})
    dates = pd.Series(["2010-01-01", "2010-02-15", "2016-01-01", "2014-01-01"])
    baseline = pd.DataFrame({
        "RID": [1, 2],
        "baseline_date": pd.to_datetime(["2010-02-01", "2010-01-01"]),
    })
    out = build.nearest_to_baseline(part, dates, baseline, window_days=180)
    assert out["RID"].tolist() == [1]  # participant 2's only scan is 4 years later
    assert out["eTIV"].iloc[0] == 1500.0


def test_apoe_genotype_to_e4_count():
    df = pd.DataFrame({"RID": [1, 2, 3, 4], "GENOTYPE": ["3/3", "3/4", "4/4", "unknown"]})
    out, dates = build.transform_apoe(df)
    assert dates is None
    assert dict(zip(out["RID"], out["apoe4"])) == {1: 0, 2: 1, 3: 2}


def test_vitals_bmi_units_and_inferred_height_unit():
    df = pd.DataFrame({
        "RID": [1, 2, 3],
        "VISDATE": ["2010-01-01"] * 3,
        "VSWEIGHT": [154.0, 70.0, 70.0],
        "VSWTUNIT": [1, 2, 2],      # pounds, kg, kg
        "VSHEIGHT": [70.0, 175.0, 69.0],
        "VSHTUNIT": [1, 2, -4],     # inches, cm, unrecorded (69 -> inches)
        "VSBPSYS": [130, 140, 120],
    })
    out, _ = build.transform_vitals(df)
    expected = [154 * 0.45359237 / (1.778 ** 2), 70 / 1.75 ** 2, 70 / (69 * 0.0254) ** 2]
    assert out["bmi"].tolist() == pytest.approx(expected, rel=1e-6)
    assert out["sbp"].tolist() == [130, 140, 120]


def test_fdg_uses_meta_roi_and_amyloid_drops_failed_qc():
    fdg = pd.DataFrame({
        "RID": [1, 1], "EXAMDATE": ["2010-01-01"] * 2,
        "ROINAME": ["MetaROI", "Top50PonsVermis"], "MEAN": [1.21, 1.0],
    })
    out, _ = build.transform_fdg(fdg)
    assert out["fdg"].tolist() == [1.21]

    amy = pd.DataFrame({
        "RID": [1, 2, 3], "SCANDATE": ["2010-01-01"] * 3,
        "qc_flag": [2, 0, -1], "CENTILOIDS": [-4.0, 50.0, 10.0], "SUMMARY_SUVR": [1.0, 1.3, 1.1],
    })
    out, _ = build.transform_amyloid(amy)
    assert out["RID"].tolist() == [1, 3]  # qc 0 = fail removed; -1 = not assessed kept
    # A Centiloid of -4 is a real value, not an ADNI missing code.
    cleaned = build.clean_clinical_values(out.assign(target=0))
    assert cleaned.loc[cleaned["RID"] == 1, "amyloid_centiloid"].item() == -4.0
