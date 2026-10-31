import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from db import get_engine, recent_predictions, save_prediction  # noqa: E402
from features import clean_education, clean_role, clean_years  # noqa: E402
from prepare_data import prepare  # noqa: E402
from train import encode_row, predict_exported  # noqa: E402


def test_years_parsing():
    assert clean_years("Less than 1 year") == 0.5
    assert clean_years("More than 50 years") == 50.0
    assert clean_years("7") == 7.0
    assert clean_years(float("nan")) is None


def test_role_priority():
    assert clean_role("Developer, back-end;Engineering manager") == "Engineering manager or executive"
    assert clean_role("Developer, front-end;Developer, full-stack") == "Full-stack developer"
    assert clean_role("Engineer, data;Developer, back-end") == "Data engineer"
    assert clean_role(None) is None


def test_education_buckets():
    assert clean_education("Master’s degree (M.A., M.S., M.Eng., MBA, etc.)") == "Master's"
    assert clean_education("Something else") is None


def test_prepare_filters_rows():
    raw = pd.DataFrame(
        {
            "MainBranch": ["I am a developer by profession"] * 3 + ["I am a student"],
            "Employment": ["Employed, full-time", "Employed, full-time", "Employed, part-time", "Employed, full-time"],
            "Country": ["India"] * 4,
            "EdLevel": ["Bachelor’s degree (B.A., B.S., B.Eng., etc.)"] * 4,
            "YearsCodePro": ["3"] * 4,
            "DevType": ["Developer, back-end"] * 4,
            "RemoteWork": ["Hybrid (some remote, some in-person)"] * 4,
            "ConvertedCompYearly": [30000, 1_000_000, 30000, 30000],
        }
    )
    out = prepare(raw)
    assert len(out) == 1  # salary outlier, part-time, and student rows removed
    assert out.loc[0, "country"] == "Other"  # India has fewer than 200 rows here


@pytest.mark.skipif(not (ROOT / "models" / "salary_model.joblib").exists(), reason="run src/train.py first")
def test_browser_model_matches_sklearn():
    import joblib

    bundle = joblib.load(ROOT / "models" / "salary_model.joblib")
    web = json.loads((ROOT / "docs" / "model.json").read_text())
    rows = [
        {"country": "Germany", "education": "Master's", "years_pro": 4.0, "role": "Data engineer", "remote": "Hybrid"},
        {"country": "Other", "education": "Bachelor's", "years_pro": 12.0, "role": "Mobile developer", "remote": "In-person"},
    ]
    for key in ("median", "p25", "p75"):
        sk = bundle["models"][key].predict(pd.DataFrame(rows))
        js = [predict_exported(web["models"][key], encode_row(r, web["countries"])) for r in rows]
        assert np.allclose(sk, js, atol=1e-4)


def test_prediction_history_roundtrip(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'p.db'}")
    inputs = {"country": "Canada", "education": "Bachelor's", "years_pro": 2.0, "role": "Back-end developer", "remote": "Fully remote"}
    save_prediction(engine, inputs, 70000.4, 55000, 90000)
    rows = recent_predictions(engine)
    assert rows[0]["predicted_usd"] == 70000 and rows[0]["country"] == "Canada"
