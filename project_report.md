# Project Report: Student Dropout Early Intervention System

**Team members:** _[add names here]_
**Course / capstone:** AIML Capstone
**Date:** 2026-07-11

---

## 1. Problem statement

Student attrition is typically detected retroactively — after a failed re-enrollment or
a formal withdrawal — at which point intervention options are limited and expensive.
This project reframes dropout prediction as an **early-intervention triage problem**:
given a student's admissions profile and in-progress academic performance, produce a
calibrated risk score *and* an explainable, actionable recommendation, early enough for
an advisor to intervene.

## 2. Real-world impact

A working system in this space needs to satisfy three groups simultaneously:

- **Academic advisors** need a triage list, not a spreadsheet — the system ranks
  students by P(Dropout) so the highest-risk cases are reviewed first (`src/intervention.py::score_batch`).
- **Students** need interventions that make sense for their specific situation, not a
  generic warning — every report names the top factors driving the score in plain
  language (`src/intervention.py::humanize_factor`).
- **Institutions** need this to be auditable and safe — every score is probabilistic,
  every recommendation is human-reviewed, and the rule mapping score → action is a
  simple, inspectable threshold table, not another opaque model (see §7, Responsible Use).

## 3. Dataset

**Source:** UCI Machine Learning Repository / Kaggle, *"Predict Students' Dropout and
Academic Success"* (Realinho, Vieira Martins, Machado, Baptista, 2021). 4,424 students,
one Portuguese higher-education institution, 34 features + target across four domains:
demographics, socioeconomic status, admissions history, and 1st/2nd-semester in-program
performance (units credited/enrolled/evaluated/approved, grades).

**P1 data audit findings** (`python -m src.main audit`):
- Shape: 4,424 rows × 35 columns. Target auto-detected as `Target` (`Dropout`/`Enrolled`/`Graduate`).
- No missing values, no duplicate rows.
- Class distribution: Graduate 49.9% (2,209), Dropout 32.1% (1,421), Enrolled 18.0% (794)
  — imbalance ratio 2.78x, flagged and handled via `class_weight="balanced"` and
  stratified sampling throughout.
- Data-quality quirk: the raw CSV header had a double-encoded byte-order-mark (UTF-8 BOM
  bytes re-saved as Latin-1, producing mojibake `ï»¿` in the first column name). Fixed
  transparently in `src/utils.py::load_raw_csv`, documented in `data/README.md`.

## 4. Methodology / workflow

1. **P1 — Data audit** (`src/main.py::run_audit`): load, profile, auto-detect target,
   flag imbalance. No assumptions about column names — the pipeline fails loudly with a
   clear error if expected columns are missing (`src/preprocessing.py::load_clean_data`),
   rather than silently mis-training on the wrong schema.
2. **P2 — Preprocessing** (`src/preprocessing.py`): features are split into three
   domain-informed groups — nominal categorical codes (Course, Application mode,
   parents' qualification/occupation, etc. → one-hot encoded), already-binary flags, and
   continuous numerics (both → median-imputed + standard-scaled). A single
   `ColumnTransformer` is fit once on the training split and persisted with `joblib`, so
   the exact same transform is applied at inference time — no train/serve skew.
3. **P3 — EDA** (`notebooks/exploration_or_modeling.ipynb`): visualizes outcome
   distribution, units-approved vs outcome, grades vs outcome, financial factors
   (tuition/debtor/scholarship) vs outcome, age vs outcome, previous-qualification vs
   outcome, and a full numeric correlation-with-dropout ranking. All 7 figures are saved
   to `docs/figures/` for slide use.
4. **P4 — Modeling** (`src/model.py`): three candidates — Logistic Regression baseline,
   Random Forest, Gradient Boosting — all trained with class-imbalance handling
   (`class_weight="balanced"` for LogReg/RF; `sample_weight` for GB, which has no
   native `class_weight`). Headline metrics come from 5-fold **stratified
   cross-validation** on the training split, not a single lucky split. A held-out 20%
   test set (also stratified) provides the final comparison.
5. **P5 — Explainability** (`src/explain.py`): SHAP `TreeExplainer` computes both global
   feature importance (mean |SHAP value| for the Dropout class, aggregated back from
   one-hot columns to human-readable raw feature names) and per-student signed
   contributions, with a tree-importance fallback if SHAP is unavailable.
6. **P6 — Intervention logic** (`src/intervention.py`): P(Dropout) is mapped to
   Low (<33%) / Medium (33–66%) / High (≥66%) bands, each with a concrete recommended
   action. Per-student reports combine the predicted outcome, risk band, top 3–5
   human-readable risk factors, and the recommendation into one plain-English narrative.
7. **P7 — Streamlit app** (`app/app.py`): single-student form and batch CSV upload,
   both routed through the same `intervention.py` functions used by the CLI — no logic
   duplication between the app and the pipeline.
8. **P8 — Tests / guardrails** (`tests/test_scenarios.py`): end-to-end scenario tests
   (clearly-at-risk → High, strong → Low, borderline → Medium) plus structural
   guardrails (every report has factors + an action; probabilities sum to 1).

## 5. Model selection rationale

Selection criterion, applied on the held-out test set: **pick the candidate with the
highest Dropout-class recall; if within 0.01 of the top score, break the tie by
macro-F1.** This is stated explicitly in `src/model.py::run_training` rather than
implied. The reasoning: in this domain, a false negative (an at-risk student who is
never flagged) has a materially worse real-world cost than a false positive (a routine
check-in on a student who turns out not to need one). Optimizing for raw accuracy alone
would silently favor the majority `Graduate` class.

**Result:** Random Forest was selected (test Dropout recall 0.725, macro-F1 0.719,
accuracy 0.771) — narrowly ahead of Logistic Regression and Gradient Boosting on the
prioritized metric while also having the best macro-F1 among the three. Full metrics are
in the results table below and machine-readable in `models/metrics_report.json`.

## 6. Results

### Cross-validation (5-fold stratified, training split)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Dropout Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.758 ± 0.015 | 0.719 ± 0.018 | 0.724 ± 0.018 | 0.717 ± 0.017 | 0.747 ± 0.026 |
| Random Forest | 0.775 ± 0.016 | 0.732 ± 0.020 | 0.720 ± 0.020 | 0.724 ± 0.020 | 0.742 ± 0.019 |
| Gradient Boosting | 0.778 ± 0.005 | 0.733 ± 0.007 | 0.694 ± 0.013 | 0.705 ± 0.013 | 0.761 ± 0.015 |

### Held-out test set (20%, stratified, never used in CV)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Dropout Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.693 | 0.697 | 0.685 | 0.690 |
| **Random Forest (selected)** | **0.771** | **0.723** | **0.719** | **0.719** | **0.725** |
| Gradient Boosting | 0.736 | 0.706 | 0.711 | 0.698 | 0.690 |

### Test-set confusion matrix (Random Forest)

Rows = actual, columns = predicted, order [Dropout, Enrolled, Graduate]:

```
                Predicted
              Dropout  Enrolled  Graduate
Actual Dropout   206      47        31
Actual Enrolled   31      88        40
Actual Graduate   12      42       388
```

### Per-class test performance (Random Forest)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Dropout | 0.827 | 0.725 | 0.773 | 284 |
| Enrolled | 0.497 | 0.553 | 0.524 | 159 |
| Graduate | 0.845 | 0.878 | 0.861 | 442 |

### Global feature importance (SHAP, Dropout class — top drivers)

1. Curricular units 2nd sem (approved)
2. Curricular units 2nd sem (grade)
3. Curricular units 1st sem (approved)
4. Tuition fees up to date
5. Course
6. Curricular units 1st sem (grade)
7. Application mode
8. Age at enrollment

This matches the EDA in the notebook: **in-program academic performance dominates**,
with financial status as a strong secondary signal — both are things an advisor can
actually act on.

## 7. Explainability and intervention design

Rather than surfacing a bare probability, every prediction is passed through
`src/explain.py::explain_student` (SHAP per-student attribution) and then
`src/intervention.py::build_student_report`, which:
- maps P(Dropout) to a Low/Medium/High band via `config.RISK_BAND_THRESHOLDS`,
- translates the top SHAP-ranked raw features into plain-English sentences
  (`humanize_factor`), and
- attaches a concrete, tiered recommended action (`RECOMMENDED_ACTIONS`).

This intervention layer is deliberately simple, rule-based, and inspectable — it is not
a second model. The complexity (and the risk of opacity) lives entirely in the
classifier + SHAP step; the decision rule on top is a threshold table any advisor or
auditor can read in five minutes.

## 8. Limitations

- Single-institution, single-cohort (2021, Portugal) data — no claim of generalization
  to other institutions without re-validation.
- Static snapshot labels, not longitudinal/behavioral data — the model cannot see
  trends within a semester, only end-of-semester aggregates.
- The `Enrolled` class is intrinsically harder to separate (recall 0.553) — a
  still-enrolled student's profile can resemble either eventual outcome. This is
  reported honestly rather than masked by aggregate accuracy.
- Nominal feature codes (Course, occupation, qualification) are anonymized integers in
  this dataset; the model can use them predictively, but a report cannot translate code
  `13` back into "Bachelor's degree" without the institution's original codebook.

## 9. Responsible use

- **Human-in-the-loop only** — every output is a decision-support signal for a trained
  advisor, never an automated decision. Enforced conceptually via the mandatory
  disclaimer shown in the Streamlit app on every screen.
- **No automated penalization** — this system must never feed directly into grading,
  academic probation, or financial-aid denial workflows without human review.
- **Probabilistic, not deterministic** — a "High" band reflects a statistical pattern
  from historical data, not a certainty about any individual student's future.
- **Bias risk** — models trained on historical outcomes can encode historical
  inequities. A fairness audit across demographic subgroups (age, gender, international/
  displaced status) is recommended future work (§10) before any institution-wide
  deployment, and is explicitly out of scope for this capstone submission.

## 10. Future work

- Fairness/disparate-impact audit across demographic subgroups.
- Semester-over-semester trend features (rate of change, not just point-in-time values)
  to catch deterioration earlier in the term.
- Closed-loop evaluation: track whether flagged students who received an intervention
  had better outcomes than a matched control group.
- Re-validate against a second institution's data before considering broader deployment.

## 11. Reproducibility

```bash
pip install -r requirements.txt
python -m src.main audit                    # P1
python -m src.main train                    # P2 + P4 (saves models/*.joblib)
python -m src.main explain                  # P5
python -m src.main score data/sample_batch.csv   # P6
python -m pytest tests/test_scenarios.py -v # P8
streamlit run app/app.py                    # P7
```

All artifacts (model, preprocessor, metrics) are regenerated deterministically
(`random_state=42` throughout) by the `train` command — nothing in `models/` needs to be
manually edited or is required to be committed to version control.
