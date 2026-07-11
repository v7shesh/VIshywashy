"""
Scenario tests / guardrails for the Student Dropout Early Intervention System.

These are not unit tests of individual functions -- they are end-to-end
sanity checks that the full pipeline (preprocessor -> model -> explain ->
intervention) produces sensible, stable risk bands for representative
student profiles. Run with: pytest tests/test_scenarios.py -v

Requires a trained model (run `python -m src.main train` first).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.intervention import build_student_report, risk_band
from src.preprocessing import get_feature_columns

FEATURE_COLS = get_feature_columns()

pytestmark = pytest.mark.skipif(
    not (config.MODEL_PATH.exists() and config.PREPROCESSOR_PATH.exists()),
    reason="No trained model found -- run `python -m src.main train` first.",
)


def _make_row(**overrides) -> pd.DataFrame:
    """A neutral baseline student profile, overridable per scenario."""
    base = {
        "Marital status": 1, "Application mode": 1, "Course": 10,
        "Previous qualification": 1, "Nacionality": 1,
        "Mother's qualification": 1, "Father's qualification": 1,
        "Mother's occupation": 5, "Father's occupation": 5,
        "Daytime/evening attendance": 1, "Displaced": 0,
        "Educational special needs": 0, "Debtor": 0,
        "Tuition fees up to date": 1, "Gender": 0, "Scholarship holder": 0,
        "International": 0, "Application order": 1, "Age at enrollment": 19,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 14.0,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 14.0,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 11.1, "Inflation rate": 1.2, "GDP": 0.3,
    }
    base.update(overrides)
    return pd.DataFrame([base])[FEATURE_COLS]


class TestRiskBandThresholds:
    """Unit-level check of the band-mapping rule itself."""

    def test_band_boundaries(self):
        assert risk_band(0.0) == "Low"
        assert risk_band(0.32) == "Low"
        assert risk_band(0.33) == "Medium"
        assert risk_band(0.65) == "Medium"
        assert risk_band(0.66) == "High"
        assert risk_band(0.99) == "High"


class TestScenarios:
    """End-to-end scenario checks against the trained model."""

    def test_clearly_at_risk_profile_is_high(self):
        """Zero units approved in both semesters, zero grades, tuition
        overdue, in debt: an unambiguous at-risk case."""
        row = _make_row(**{
            "Curricular units 1st sem (approved)": 0,
            "Curricular units 1st sem (grade)": 0.0,
            "Curricular units 2nd sem (approved)": 0,
            "Curricular units 2nd sem (grade)": 0.0,
            "Tuition fees up to date": 0,
            "Debtor": 1,
            "Scholarship holder": 0,
            "Age at enrollment": 27,
        })
        report = build_student_report(row, "At-risk scenario")
        assert report["risk_band"] == "High", (
            f"Expected High, got {report['risk_band']} (P(Dropout)={report['p_dropout']:.3f})"
        )
        assert report["p_dropout"] >= 0.66

    def test_strong_profile_is_low(self):
        """Full units approved, strong grades, tuition current, no debt:
        an unambiguous low-risk case."""
        row = _make_row()  # baseline is already a strong, on-track profile
        report = build_student_report(row, "Strong scenario")
        assert report["risk_band"] == "Low", (
            f"Expected Low, got {report['risk_band']} (P(Dropout)={report['p_dropout']:.3f})"
        )
        assert report["p_dropout"] < 0.33

    def test_borderline_profile_is_medium(self):
        """Partial progress (2 of 6 units approved both semesters, grades
        around the pass/fail line): the intended Medium-band case."""
        row = _make_row(**{
            "Curricular units 1st sem (approved)": 2,
            "Curricular units 1st sem (grade)": 10.0,
            "Curricular units 2nd sem (approved)": 2,
            "Curricular units 2nd sem (grade)": 10.0,
        })
        report = build_student_report(row, "Borderline scenario")
        assert report["risk_band"] == "Medium", (
            f"Expected Medium, got {report['risk_band']} (P(Dropout)={report['p_dropout']:.3f})"
        )
        assert 0.33 <= report["p_dropout"] < 0.66

    def test_report_always_includes_top_factors_and_action(self):
        """Guardrail: every report must be explainable and actionable --
        never just a bare number."""
        for row in [_make_row(), _make_row(**{"Debtor": 1, "Tuition fees up to date": 0})]:
            report = build_student_report(row)
            assert len(report["top_factors"]) > 0
            assert report["recommended_action"]
            assert report["predicted_class"] in {"Dropout", "Enrolled", "Graduate"}

    def test_probabilities_sum_to_one(self):
        """Guardrail: predictions are a proper probability distribution
        across the 3 classes, reinforcing that outputs are probabilistic,
        not deterministic labels."""
        row = _make_row()
        report = build_student_report(row)
        total = sum(report["probabilities"].values())
        assert abs(total - 1.0) < 1e-6
