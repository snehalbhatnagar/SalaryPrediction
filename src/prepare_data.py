"""Turn the raw 2022 Stack Overflow survey CSV into a small, clean training table.

Usage:
    python src/prepare_data.py path/to/survey_results_public.csv

Writes data/processed/salaries_2022.csv.gz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import (  # noqa: E402
    MIN_COUNTRY_RESPONSES,
    clean_country,
    clean_education,
    clean_remote,
    clean_role,
    clean_years,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "salaries_2022.csv.gz"
RAW_COLS = ["MainBranch", "Employment", "Country", "EdLevel", "YearsCodePro", "DevType", "RemoteWork", "ConvertedCompYearly"]

# Annual USD pay outside this band is almost always a typo or a monthly figure.
MIN_SALARY, MAX_SALARY = 5_000, 400_000


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[
        (raw["MainBranch"] == "I am a developer by profession")
        & (raw["Employment"] == "Employed, full-time")
        & raw["ConvertedCompYearly"].between(MIN_SALARY, MAX_SALARY)
    ].copy()

    out = pd.DataFrame(
        {
            "country": df["Country"].map(clean_country),
            "education": df["EdLevel"].map(clean_education),
            "years_pro": df["YearsCodePro"].map(clean_years),
            "role": df["DevType"].map(clean_role),
            "remote": df["RemoteWork"].map(clean_remote),
            "salary_usd": df["ConvertedCompYearly"].round().astype("Int64"),
        }
    ).dropna()

    counts = out["country"].value_counts()
    out["country"] = out["country"].where(out["country"].map(counts) >= MIN_COUNTRY_RESPONSES, "Other")
    out["years_pro"] = out["years_pro"].clip(upper=40)
    return out.reset_index(drop=True)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = pd.read_csv(sys.argv[1], usecols=RAW_COLS, low_memory=False)
    clean = prepare(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT, index=False, compression="gzip")
    print(f"{len(raw):,} responses -> {len(clean):,} rows, {clean['country'].nunique()} country groups -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
