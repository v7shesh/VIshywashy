# 🎓 Student Dropout Early Intervention System

**AIML Capstone Project** — [Team members: _add names here_]

## Problem statement

Higher-education institutions typically learn a student is at risk of dropping out only
after the fact — a failed semester, a missed re-enrollment, an administrative withdrawal
form. By then, the window for a low-cost intervention (a mentor, a remedial session, a
financial-aid conversation) has often closed. This project builds a system that flags
at-risk students **while there is still time to act**, using signals available from a
student's admissions profile and in-progress academic performance.

## Real-world impact

- **Advisors get a ranked worklist**, not a spreadsheet of grades — the highest-risk
  students surface automatically when staff upload a class roster.
- **Every flag comes with a reason.** A probability alone isn't actionable; this system
  pairs each score with the specific factors driving it (e.g. "0 of 6 units approved
  this semester", "tuition fees not up to date") so an advisor knows what conversation
  to have.
- **Interventions are tiered**, not one-size-fits-all: a borderline student gets a
  remedial session and monitoring; a clearly at-risk student gets an immediate
  counselor referral and mentor assignment.

## Dataset

**Source:** UCI Machine Learning Repository / Kaggle — *"Predict Students' Dropout and
Academic Success"* (Realinho et al., 2021), 4,424 students from a Portuguese higher-education
institution, 34 features covering demographics, socioeconomic status, admissions data, and
1st/2nd-semester academic performance. Target: `Dropout` / `Enrolled` / `Graduate`.
See [`data/README.md`](data/README.md) for the full column breakdown and a documented
data-quality quirk (a double-encoded BOM in the raw CSV header) that the loader handles
transparently.

No missing values, no duplicates. Class distribution is imbalanced (Graduate 49.9% /
Dropout 32.1% / Enrolled 18.0%, ~2.8x ratio) — handled via `class_weight="balanced"` and
stratified splitting/cross-validation throughout.

## Tools

Python, pandas, scikit-learn (ColumnTransformer/Pipeline, RandomForest/GradientBoosting/
LogisticRegression), SHAP (explainability), Streamlit (app), matplotlib/seaborn (EDA),
pytest (guardrail tests), joblib (model persistence).

## Workflow

```
                 ┌────────────────┐
   dataset.csv → │  P1: Data audit │  (schema/target auto-detection, imbalance check)
                 └────────┬───────┘
                          ▼
                 ┌────────────────────┐
                 │ P2: Preprocessing   │  ColumnTransformer: one-hot (nominal) +
                 │ (preprocessing.py)  │  impute/scale (numeric) — fit once, reused
                 └────────┬───────────┘  everywhere (train + inference)
                          ▼
                 ┌────────────────────┐
                 │ P4: Modeling         │  LogReg / RandomForest / GradientBoosting
                 │ (model.py)           │  5-fold stratified CV → held-out test →
                 └────────┬───────────┘  select by Dropout-class recall
                          ▼
                 ┌────────────────────┐
                 │ P5: Explainability   │  SHAP global importance +
                 │ (explain.py)         │  per-student factor attribution
                 └────────┬───────────┘
                          ▼
                 ┌────────────────────┐
                 │ P6: Intervention     │  P(Dropout) → Low/Medium/High band →
                 │ (intervention.py)    │  concrete recommended action
                 └────────┬───────────┘
                          ▼
                 ┌────────────────────┐
                 │ P7: Streamlit app    │  single-student form + batch CSV
                 │ (app/app.py)         │  triage, ranked & downloadable
                 └────────────────────┘
```

## Where the ML logic lives, and why

| File | Responsibility | Why it's separated this way |
|---|---|---|
| `src/config.py` | Paths, feature-group lists, risk-band thresholds | Single source of truth — no magic strings scattered across files |
| `src/utils.py` | BOM-safe CSV loading, target auto-detection | Reused identically by the audit, training, and scoring paths |
| `src/preprocessing.py` | `ColumnTransformer` (one-hot + impute/scale), stratified split, raw-feature-name mapping | The **same fitted transform** is saved and reused at inference — training/serving skew is structurally impossible |
| `src/model.py` | Candidate training, 5-fold CV, held-out evaluation, model selection | Selection logic is explicit and documented (Dropout recall first, macro-F1 tiebreak), not just "pick the highest accuracy" |
| `src/explain.py` | SHAP global + per-student explanations | Decoupled from `model.py` so explanations can be recomputed without retraining |
| `src/intervention.py` | Risk banding + plain-English recommendation | Simple, auditable rules on top of the model output — the "AI" already happened upstream; this layer is deliberately not another black box |
| `app/app.py` | Streamlit UI (single-student + batch modes) | Thin wrapper around `intervention.py` — no business logic duplicated in the UI layer |

## Exact run commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Inspect the raw dataset (P1)
python -m src.main audit

# 3. Preprocess + train + evaluate + save the model (P2 + P4)
python -m src.main train

# 4. Global explainability report (P5)
python -m src.main explain

# 5. Batch-score any CSV of students (P6)
python -m src.main score data/sample_batch.csv

# 6. Run the guardrail test suite (P8)
python -m pytest tests/test_scenarios.py -v

# 7. Launch the interactive app (P7)
streamlit run app/app.py
```

## Results (real, cross-validated metrics)

5-fold stratified cross-validation on the training split (headline numbers), plus a
held-out test-set evaluation. **Dropout-class recall is the prioritized metric** — in
this domain, missing an at-risk student (false negative) is worse than a false alarm.

### Cross-validation (5-fold, training split)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Dropout Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.758 | 0.719 | 0.724 | 0.717 | 0.747 |
| Random Forest | 0.775 | 0.732 | 0.720 | 0.724 | 0.742 |
| Gradient Boosting | 0.778 | 0.733 | 0.694 | 0.705 | 0.761 |

### Held-out test set (20%, never seen during training/CV)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | **Dropout Recall** |
|---|---|---|---|---|---|
| Logistic Regression | 0.722 | 0.693 | 0.697 | 0.685 | 0.690 |
| **Random Forest (selected)** | **0.771** | **0.723** | **0.719** | **0.719** | **0.725** |
| Gradient Boosting | 0.736 | 0.706 | 0.711 | 0.698 | 0.690 |

**Selected model: Random Forest.** Confusion matrix on the test set (labels: Dropout,
Enrolled, Graduate):

```
                Predicted
              Dropout  Enrolled  Graduate
Actual Dropout   206      47        31
Actual Enrolled   31      88        40
Actual Graduate   12      42       388
```

Per-class test performance:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Dropout | 0.827 | 0.725 | 0.773 | 284 |
| Enrolled | 0.497 | 0.553 | 0.524 | 159 |
| Graduate | 0.845 | 0.878 | 0.861 | 442 |

**Read honestly:** the `Enrolled` class is the hardest to separate (still-enrolled
students look statistically similar to both eventual graduates and eventual dropouts
mid-program) — this is a known, expected limitation of this dataset, not a bug. Dropout
recall (0.725) means roughly 3 in 4 actual dropouts are correctly flagged by the model;
the remaining ~27% is exactly why this is a *support tool for human advisors*, not an
autonomous decision system.

## Demo screenshots

_Add screenshots here from the running Streamlit app (single-student assessment view and
batch-triage view) before submission. EDA figures used for slides are already saved in
[`docs/figures/`](docs/figures/)._

## Limitations

- Trained on one Portuguese institution's 2021 cohort — outcome base rates, grading
  scales, and socioeconomic factors will not transfer directly to another institution
  without re-validation.
- `Enrolled` vs `Dropout` vs `Graduate` is a snapshot label at data-collection time, not
  a live, continuously-updated status — the model was not trained on time-series
  behavioral data.
- Categorical codes (Course, Application mode, parents' qualification/occupation) are
  anonymized integers in this dataset version; the model uses them, but a human reading
  a report cannot decode their real-world meaning without the institution's original
  codebook.
- Class imbalance and the inherent difficulty of the `Enrolled` category mean recall on
  that class is materially weaker than on `Dropout`/`Graduate` — see the results table.

## Responsible use

- **Human-in-the-loop only.** Every score is a prompt for an advisor conversation, never
  an automated decision. See the in-app disclaimer (shown on every page of the Streamlit
  app) and `tests/test_scenarios.py`, which encodes this as an explicit guardrail.
- **No automated penalization.** Nothing in this system should feed into grading,
  probation decisions, or financial-aid denial without human review.
- **Predictions are probabilistic, not deterministic** — a "High" risk band is a
  statistical signal from historical patterns, not a certainty about any individual
  student.
- **Bias risk.** Any model trained on historical outcome data can encode and reproduce
  historical inequities (e.g. around socioeconomic status). This system should be
  periodically audited for disparate impact across demographic groups before
  institution-wide deployment.

## Future work

- Re-validate on a second institution's data before any cross-institution deployment.
- Add semester-over-semester trend features (not just point-in-time snapshots) to catch
  deterioration earlier.
- Track intervention outcomes (did the flagged student's actual trajectory improve?) to
  close the loop and evaluate real-world impact, not just offline accuracy.
- Fairness audit across demographic subgroups (age, gender, international status) before
  wider rollout.
