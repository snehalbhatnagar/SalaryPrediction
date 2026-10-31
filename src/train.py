"""Train the salary model, evaluate it, and export it for the apps.

Usage:
    python src/train.py

Reads data/processed/salaries_2022.csv.gz and writes:
    models/salary_model.joblib   scikit-learn pipelines for the Streamlit app
    models/metrics.json          held-out test metrics
    docs/model.json              the same trees as plain JSON for the browser demo
    docs/explore.json            summary tables for the explore charts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

sys.path.insert(0, str(Path(__file__).parent))
from features import EDUCATION_LEVELS, FEATURES, REMOTE, ROLES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "salaries_2022.csv.gz"
SEED = 42

GBM = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=20, random_state=SEED)


def make_pipeline(countries: list[str], loss: str = "squared_error", alpha: float = 0.5) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(categories=[countries, ROLES, REMOTE], handle_unknown="ignore", sparse_output=False), ["country", "role", "remote"]),
            ("edu", OrdinalEncoder(categories=[EDUCATION_LEVELS]), ["education"]),
            ("num", "passthrough", ["years_pro"]),
        ]
    )
    return Pipeline([("pre", pre), ("gbm", GradientBoostingRegressor(loss=loss, alpha=alpha, **GBM))])


def export_trees(pipe: Pipeline) -> dict:
    """Flatten a fitted GradientBoostingRegressor into arrays a browser can walk."""
    gbm = pipe.named_steps["gbm"]
    trees = []
    for est in gbm.estimators_[:, 0]:
        t = est.tree_
        trees.append(
            {
                "f": t.feature.tolist(),
                "t": np.round(t.threshold, 6).tolist(),
                "l": t.children_left.tolist(),
                "r": t.children_right.tolist(),
                "v": np.round(t.value[:, 0, 0], 6).tolist(),
            }
        )
    init = float(gbm.init_.constant_[0][0]) if hasattr(gbm.init_, "constant_") else float(gbm._raw_predict_init(np.zeros((1, gbm.n_features_in_)))[0, 0])
    return {"init": init, "lr": gbm.learning_rate, "trees": trees}


def encode_row(row: dict, countries: list[str]) -> list[float]:
    """The feature vector the exported trees expect (same order as the pipeline)."""
    vec = [1.0 if row["country"] == c else 0.0 for c in countries]
    vec += [1.0 if row["role"] == r else 0.0 for r in ROLES]
    vec += [1.0 if row["remote"] == r else 0.0 for r in REMOTE]
    vec.append(float(EDUCATION_LEVELS.index(row["education"])))
    vec.append(float(row["years_pro"]))
    return vec


def predict_exported(model: dict, x: list[float]) -> float:
    total = model["init"]
    for t in model["trees"]:
        n = 0
        while t["l"][n] != -1:
            n = t["l"][n] if x[t["f"][n]] <= t["t"][n] else t["r"][n]
        total += model["lr"] * t["v"][n]
    return total


def main() -> None:
    df = pd.read_csv(DATA)
    countries = sorted(c for c in df["country"].unique() if c != "Other") + ["Other"]
    X, y = df[FEATURES], np.log(df["salary_usd"].astype(float))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "median": make_pipeline(countries).fit(X_tr, y_tr),
        "p25": make_pipeline(countries, "quantile", 0.25).fit(X_tr, y_tr),
        "p75": make_pipeline(countries, "quantile", 0.75).fit(X_tr, y_tr),
    }

    pred = np.exp(models["median"].predict(X_te))
    actual = np.exp(y_te)
    lo, hi = np.exp(models["p25"].predict(X_te)), np.exp(models["p75"].predict(X_te))
    baseline = np.exp(y_tr.groupby(X_tr["country"]).median()).reindex(X_te["country"]).to_numpy()
    metrics = {
        "rows_train": int(len(X_tr)),
        "rows_test": int(len(X_te)),
        "mae_usd": round(float(mean_absolute_error(actual, pred))),
        "median_abs_error_usd": round(float(median_absolute_error(actual, pred))),
        "r2_log_salary": round(float(r2_score(y_te, np.log(pred))), 3),
        "baseline_country_median_mae_usd": round(float(mean_absolute_error(actual, baseline))),
        "share_inside_p25_p75": round(float(np.mean((actual >= lo) & (actual <= hi))), 3),
    }

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"models": models, "countries": countries, "metrics": metrics}, ROOT / "models" / "salary_model.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    web = {
        "countries": countries,
        "roles": ROLES,
        "remote": REMOTE,
        "education": EDUCATION_LEVELS,
        "metrics": metrics,
        "models": {k: export_trees(m) for k, m in models.items()},
    }
    # Parity check: the exported trees must reproduce scikit-learn's predictions.
    sample = X_te.head(200).to_dict("records")
    sk = models["median"].predict(pd.DataFrame(sample))
    js = [predict_exported(web["models"]["median"], encode_row(r, countries)) for r in sample]
    assert np.allclose(sk, js, atol=1e-4), "exported trees disagree with scikit-learn"

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "model.json").write_text(json.dumps(web, separators=(",", ":")))

    by = lambda col: (  # noqa: E731
        df.groupby(col)["salary_usd"].agg(median="median", n="size").reset_index().sort_values("median", ascending=False).to_dict("records")
    )
    bands = pd.cut(df["years_pro"], [0, 2, 5, 10, 15, 20, 41], right=False, labels=["0-1", "2-4", "5-9", "10-14", "15-19", "20+"])
    explore = {
        "rows": int(len(df)),
        "by_country": by("country"),
        "by_role": by("role"),
        "by_experience": df.groupby(bands, observed=True)["salary_usd"].agg(median="median", n="size").reset_index().rename(columns={"years_pro": "band"}).to_dict("records"),
    }
    (ROOT / "docs" / "explore.json").write_text(json.dumps(explore, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
