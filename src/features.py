"""Cleaning rules shared by data preparation, training, and the apps.

Each function maps one raw Stack Overflow 2022 survey answer onto the small
set of categories the model uses, so the training data and live inputs are
encoded the same way.
"""
from __future__ import annotations

import math

MIN_COUNTRY_RESPONSES = 200

EDUCATION_LEVELS = ["Less than a bachelor's", "Bachelor's", "Master's", "Doctorate or professional"]

ROLES = [
    "Back-end developer",
    "Front-end developer",
    "Full-stack developer",
    "Mobile developer",
    "Embedded or desktop developer",
    "Data engineer",
    "Data scientist or ML engineer",
    "DevOps, SRE, or cloud",
    "Engineering manager or executive",
    "Other",
]

REMOTE = ["Fully remote", "Hybrid", "In-person"]

FEATURES = ["country", "education", "years_pro", "role", "remote"]


def clean_education(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    if raw.startswith("Bachelor"):
        return "Bachelor's"
    if raw.startswith("Master"):
        return "Master's"
    if raw.startswith(("Other doctoral", "Professional degree")):
        return "Doctorate or professional"
    if raw.startswith(("Some college", "Secondary school", "Associate", "Primary")):
        return "Less than a bachelor's"
    return None  # "Something else" is too vague to encode


def clean_years(raw) -> float | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    if raw == "Less than 1 year":
        return 0.5
    if raw == "More than 50 years":
        return 50.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def clean_role(raw: str | None) -> str | None:
    """The survey allows several roles; pick the most specific one.

    Order matters: someone who ticks both "Engineering manager" and
    "Developer, back-end" is counted as a manager.
    """
    if not isinstance(raw, str):
        return None
    roles = raw.split(";")
    has = lambda *keys: any(any(k in r for k in keys) for r in roles)  # noqa: E731
    if has("Engineering manager", "Senior Executive"):
        return "Engineering manager or executive"
    if has("Data scientist"):
        return "Data scientist or ML engineer"
    if has("Engineer, data"):
        return "Data engineer"
    if has("site reliability", "DevOps", "Cloud infrastructure"):
        return "DevOps, SRE, or cloud"
    if has("Developer, mobile"):
        return "Mobile developer"
    if has("embedded", "desktop or enterprise"):
        return "Embedded or desktop developer"
    if has("full-stack"):
        return "Full-stack developer"
    if has("back-end"):
        return "Back-end developer"
    if has("front-end"):
        return "Front-end developer"
    return "Other"


def clean_remote(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    if raw.startswith("Fully remote"):
        return "Fully remote"
    if raw.startswith("Hybrid"):
        return "Hybrid"
    if raw.startswith("Full in-person"):
        return "In-person"
    return None


SHORT_COUNTRY = {
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Russian Federation": "Russia",
    "Iran, Islamic Republic of...": "Iran",
}


def clean_country(raw: str | None) -> str | None:
    if not isinstance(raw, str):
        return None
    return SHORT_COUNTRY.get(raw, raw)
