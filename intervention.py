"""
Intervention logic: turn P(Dropout) + top risk factors into a concrete,
plain-English recommendation an academic advisor can act on.

This module is intentionally simple and rule-based (not another model) --
the "AI" already happened in model.py/explain.py; this layer's job is to
translate a probability into an operational decision, transparently, so
staff can see and override the rule if needed.
"""
from pathlib import Path

import pandas as pd

from src import config
from src.explain import explain_student
from src.preprocessing import get_feature_columns, load_raw_csv
from src.utils import section

RECOMMENDED_ACTIONS = {
    "High": (
        "Immediate counselor referral: assign a dedicated mentor and schedule "
        "an in-person check-in within the next week. Flag to the program "
        "coordinator for a financial-aid/academic-support review."
    ),
    "Medium": (
        "Enroll in a remedial/tutoring session for the weakest subject area "
        "and place on a structured attendance-monitoring plan for the next "
        "grading period."
    ),
    "Low": (
        "No immediate action needed. Continue routine academic tracking and "
        "re-screen at the next grading period."
    ),
}

# Binary features get a plain yes/no label instead of a raw 0/1.
_BINARY_LABELS = {
    "Debtor": ("has outstanding debt", "no outstanding debt"),
    "Tuition fees up to date": ("tuition fees are NOT up to date", "tuition fees are up to date"),
    "Scholarship holder": ("holds a scholarship", "does not hold a scholarship"),
    "Displaced": ("is a displaced student", "is not a displaced student"),
    "Educational special needs": ("has registered special educational needs", "no registered special educational needs"),
    "International": ("is an international student", "is a domestic student"),
    "Gender": ("male", "female"),
    "Daytime/evening attendance": ("attends daytime classes", "attends evening classes"),
}

_CODED_CATEGORICAL = {
    "Course", "Application mode", "Marital status", "Nacionality",
    "Previous qualification", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
}


def risk_band(p_dropout: float) -> str:
    for band, (low, high) in config.RISK_BAND_THRESHOLDS.items():
        if low <= p_dropout < high:
            return band
    return "High"


def _factor_clause(feature: str, raw_value) -> str:
    """Plain-English description of a (feature, value) pair, no direction
    suffix -- callers append direction as text (CLI/CSV) or as an icon (UI)."""
    if feature in _BINARY_LABELS:
        true_label, false_label = _BINARY_LABELS[feature]
        return f"Student {true_label if raw_value == 1 else false_label}"

    if feature in _CODED_CATEGORICAL:
        return f"{feature} = code {raw_value}"

    if "grade" in feature.lower():
        return f"{feature} = {float(raw_value):.1f}"

    if "approved" in feature.lower() or "enrolled" in feature.lower() or "evaluations" in feature.lower():
        return f"{feature} = {int(raw_value)}"

    if feature == "Age at enrollment":
        return f"Age at enrollment = {int(raw_value)}"

    return f"{feature} = {raw_value}"


def humanize_factor(feature: str, raw_value, direction: str) -> str:
    """Turn a (feature, value, direction) triple into a plain-English
    sentence fragment an advisor can read without knowing the encoding."""
    verb = "raises" if direction == "increases" else "lowers" if direction == "decreases" else "influences"
    return f"{_factor_clause(feature, raw_value)} ({verb} dropout risk)"


def build_student_report(raw_row: pd.DataFrame, student_label: str = "Student") -> dict:
    """Full explainable report for one student: prediction, risk band,
    top human-readable factors, and a recommended intervention."""
    explanation = explain_student(raw_row)
    p_dropout = explanation["p_dropout"]
    band = risk_band(p_dropout)

    readable_factors = [
        humanize_factor(f["feature"], f["raw_value"], f["direction"])
        for f in explanation["top_factors"]
    ]
    factors_detail = [
        {
            "text": _factor_clause(f["feature"], f["raw_value"]),
            "direction": f["direction"],
        }
        for f in explanation["top_factors"]
    ]

    narrative = (
        f"{student_label}: predicted outcome = {explanation['predicted_class']} "
        f"(P(Dropout) = {p_dropout:.1%}, risk band = {band}). "
        f"Top factors: {'; '.join(readable_factors[:3])}. "
        f"Recommended action: {RECOMMENDED_ACTIONS[band]}"
    )

    return {
        "student_label": student_label,
        "predicted_class": explanation["predicted_class"],
        "p_dropout": p_dropout,
        "probabilities": explanation["probabilities"],
        "risk_band": band,
        "top_factors": readable_factors,
        "factors_detail": factors_detail,
        "recommended_action": RECOMMENDED_ACTIONS[band],
        "narrative": narrative,
    }


def score_batch(df: pd.DataFrame, id_col: str = None) -> pd.DataFrame:
    """Score every row in df, return a DataFrame ranked by P(Dropout) desc
    so staff can triage the highest-risk students first."""
    feature_cols = get_feature_columns()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded CSV is missing required columns: {missing}. "
            f"Expected all of: {feature_cols}"
        )

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        label = str(row[id_col]) if id_col and id_col in df.columns else f"Student #{i+1}"
        single = pd.DataFrame([row[feature_cols]])
        report = build_student_report(single, student_label=label)
        rows.append({
            "student": label,
            "predicted_outcome": report["predicted_class"],
            "p_dropout": report["p_dropout"],
            "risk_band": report["risk_band"],
            "top_factors": "; ".join(report["top_factors"][:3]),
            "recommended_action": report["recommended_action"],
        })

    result = pd.DataFrame(rows).sort_values("p_dropout", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def run_batch_score(csv_path: str) -> None:
    section("BATCH SCORING")
    path = Path(csv_path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        return
    df = load_raw_csv(path)
    ranked = score_batch(df)

    out_path = path.parent / f"{path.stem}_scored.csv"
    ranked.to_csv(out_path, index=False)

    print(f"Scored {len(ranked)} students.")
    print(ranked.head(10).to_string(index=False))
    print(f"\nFull ranked list saved to: {out_path}")

    band_counts = ranked["risk_band"].value_counts()
    section("SUMMARY")
    print(f"Risk band distribution: {dict(band_counts)}")
    print("Recommended: triage 'High' band students first.")
