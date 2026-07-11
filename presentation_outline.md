# Presentation Outline — Student Dropout Early Intervention System

10 slides. Each slide lists the content plus speaker bullet points you can drop directly
into your slide deck notes.

---

## Slide 1 — Title & Team

**Content:** Project title, your names/team, course name, date.

**Speaker bullets:**
- "Student Dropout Early Intervention System — an AI tool that flags at-risk students
  early enough for advisors to actually help."
- Team members: _[fill in]_.

---

## Slide 2 — Problem & Real-World Impact

**Content:** The gap between "when a student actually becomes at-risk" and "when the
institution finds out" — usually after the fact.

**Speaker bullets:**
- Institutions typically detect dropout risk retroactively (failed re-enrollment,
  withdrawal form) — too late for a low-cost intervention.
- Our system flags risk *during* the semester using academic + socioeconomic signals.
- Impact: advisors get a ranked, explainable worklist instead of a raw gradebook.

---

## Slide 3 — Dataset

**Content:** UCI/Kaggle "Predict Students' Dropout and Academic Success", 4,424 students,
34 features, 3-class target (Dropout/Enrolled/Graduate).

**Speaker bullets:**
- Real institutional data — Portuguese higher-ed, admissions + 1st/2nd semester
  performance + socioeconomic fields.
- No missing data, no duplicates — but a real class imbalance (32% Dropout, 18%
  Enrolled, 50% Graduate) that we explicitly handle rather than ignore.
- Show: `docs/figures/target_distribution.png`

---

## Slide 4 — Workflow / Architecture

**Content:** Pipeline diagram: Data audit → Preprocessing → Modeling → Explainability →
Intervention logic → App.

**Speaker bullets:**
- Every stage is a separate, testable module (`src/preprocessing.py`, `src/model.py`,
  `src/explain.py`, `src/intervention.py`) — not one monolithic script.
- One fitted preprocessing pipeline is reused identically for training and for live
  predictions in the app — no train/serve mismatch.
- Reference the ASCII workflow diagram in `README.md`.

---

## Slide 5 — AI/ML Innovation

**Content:** The two things that make this more than "just a classifier":
1. **3-class prediction reframed as a risk score** — P(Dropout) drives a ranked
   triage list, not just a label.
2. **Explainability baked into the output**, not bolted on — every score comes with
   the specific SHAP-ranked factors driving it, in plain English.

**Speaker bullets:**
- We don't just predict Dropout/Enrolled/Graduate — we extract P(Dropout) specifically
  as an actionable risk score, because "how urgent is this" matters as much as "what
  will happen."
- SHAP tells us *why* a specific student is flagged, e.g. "0 of 6 units approved this
  semester, tuition fees overdue" — an advisor can act on that immediately.
- The risk-to-action mapping (Low/Medium/High → recommended intervention) is a simple,
  auditable rule table on top of the model — deliberately not another black box.

---

## Slide 6 — Demo Screenshots

**Content:** Screenshots of the Streamlit app: (a) single-student form + risk
assessment, (b) batch CSV upload + ranked download.

**Speaker bullets:**
- _[Insert screenshot: single-student assessment view showing risk band, top factors,
  recommended action]_
- _[Insert screenshot: batch triage view with ranked table and download button]_
- Two modes cover both use cases: a counselor reviewing one student, and a program
  coordinator triaging an entire cohort at once.

---

## Slide 7 — Results

**Content:** The results table from `README.md` / `docs/project_report.md`.

**Speaker bullets:**
- Selected model: **Random Forest** — chosen using an explicit, stated rule: prioritize
  Dropout-class recall (0.725 on held-out test), not raw accuracy, because missing an
  at-risk student is costlier than a false alarm.
- 5-fold cross-validation used for headline numbers, not a single lucky split.
- Honest reporting: the `Enrolled` class is harder to classify (recall 0.55) — we say so
  explicitly rather than hiding it behind an aggregate accuracy number.
- Show: `docs/figures/global_feature_importance.png` and the confusion matrix.

---

## Slide 8 — Limitations & Responsible Use

**Content:** Bullet list from the README's Limitations + Responsible Use sections.

**Speaker bullets:**
- Single institution, single cohort — not validated for cross-institution deployment.
- `Enrolled` vs the other two classes is intrinsically the hardest distinction in this
  dataset.
- **Human-in-the-loop only** — this is a decision-support signal, never an automated
  penalty or denial mechanism.
- Predictions are probabilistic, not deterministic; any model trained on historical
  outcomes can encode historical bias — a fairness audit is required future work before
  wider deployment.

---

## Slide 9 — Future Work

**Content:** Roadmap bullets.

**Speaker bullets:**
- Fairness/disparate-impact audit across demographic subgroups.
- Semester-over-semester trend features to catch deterioration earlier, not just
  end-of-semester snapshots.
- Closed-loop evaluation: did students who got a flagged intervention actually do
  better?
- Validate against a second institution before any real deployment.

---

## Slide 10 — Conclusion

**Content:** One-paragraph wrap-up + thank you / Q&A.

**Speaker bullets:**
- This system turns a standard 3-class classifier into something an advisor can act on
  today: a ranked, explainable, tiered intervention list — with honest metrics and
  explicit responsible-use guardrails built in from day one.
- Thank you — happy to take questions.
