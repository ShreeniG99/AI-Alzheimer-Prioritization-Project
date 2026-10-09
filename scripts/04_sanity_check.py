import argparse
import json
import pandas as pd
from pathlib import Path
from common import ROOT, load_config, processed_dir, artifacts_dir

def main(stage):
    data = pd.read_csv(processed_dir() / f"{stage}.csv")
    artifact = artifacts_dir() / f"{stage}_model.joblib"
    metadata = artifacts_dir() / f"{stage}_metadata.json"

    assert len(data) > 0, "Processed dataset is empty."
    assert "RID" in data.columns, "RID missing."
    assert "target" in data.columns, "target missing."
    assert data["target"].isin([0,1]).all(), "Target is not binary."
    assert artifact.exists(), "Model artifact missing."
    assert metadata.exists(), "Model metadata missing."

    meta = json.loads(metadata.read_text(encoding="utf-8"))
    print("Stage:", stage)
    print("Rows:", len(data))
    print("Features:", meta["features"])
    print("Selected model:", meta["selected_model"])
    print("Test metrics:", meta["untouched_test_metrics"])
    print("SANITY CHECK PASSED")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(load_config()["features"]), required=True)
    main(ap.parse_args().stage)
