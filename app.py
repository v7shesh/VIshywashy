"""
Streamlit app: Student Dropout Early Intervention System.

Two modes:
  (a) Single-student form -> instant risk assessment + explanation.
  (b) Batch CSV upload -> ranked at-risk list, downloadable.

Run with: streamlit run app/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.intervention import build_student_report, score_batch
from src.preprocessing import get_feature_columns

st.set_page_config(
    page_title="Student Dropout Early Intervention System",
    page_icon="🎓",
    layout="wide",
)

# --- Status palette (validated: dataviz skill status ramp) ---------------
STATUS = {
    "Low":    {"color": "#0ca30c", "track": "#cfe9cf", "icon": "🟢", "word": "Low risk"},
    "Medium": {"color": "#c98500", "track": "#f6e2b8", "icon": "🟡", "word": "Medium risk"},
    "High":   {"color": "#d03b3b", "track": "#f3cccc", "icon": "🔴", "word": "High risk"},
}
DIRECTION_ICON = {"increases": "🔺", "decreases": "🔻"}

DISCLAIMER = (
    "**Responsible use:** this is a *probabilistic* risk score for **human-in-the-loop** "
    "use by academic advisors -- never an automated penalty, denial, or verdict. "
    "Treat a flag as a prompt for a supportive conversation, and remember any model "
    "trained on historical data can carry the biases in that data."
)

FEATURE_COLS = get_feature_columns()

# ---------------------------------------------------------------------------
# Human-readable labels for coded categorical fields, from the published UCI
# dataset documentation (Realinho et al., 2021), re-indexed to match this
# dataset's contiguous 1..N codes. NOTE: Mother's/Father's qualification and
# occupation are NOT included here -- this dataset's two columns show
# different max codes (29 vs 34) for what the source paper documents as one
# shared 34-item codebook, which means each column was independently
# re-indexed and a single label table cannot be reliably applied to both.
# Those two fields are kept as plain numeric codes rather than risk showing
# an incorrect label.
# ---------------------------------------------------------------------------
MARITAL_STATUS_LABELS = {
    1: "Single", 2: "Married", 3: "Widower", 4: "Divorced",
    5: "Facto union", 6: "Legally separated",
}
APPLICATION_MODE_LABELS = {
    1: "1st phase - general contingent", 2: "Ordinance No. 612/93",
    3: "1st phase - special contingent (Azores Island)",
    4: "Holders of other higher courses", 5: "Ordinance No. 854-B/99",
    6: "International student (bachelor)",
    7: "1st phase - special contingent (Madeira Island)",
    8: "2nd phase - general contingent", 9: "3rd phase - general contingent",
    10: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    11: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    12: "Over 23 years old", 13: "Transfer", 14: "Change of course",
    15: "Technological specialization diploma holders",
    16: "Change of institution/course", 17: "Short cycle diploma holders",
    18: "Change of institution/course (International)",
}
COURSE_LABELS = {
    1: "Biofuel Production Technologies", 2: "Animation and Multimedia Design",
    3: "Social Service (evening attendance)", 4: "Agronomy",
    5: "Communication Design", 6: "Veterinary Nursing",
    7: "Informatics Engineering", 8: "Equinculture", 9: "Management",
    10: "Social Service", 11: "Tourism", 12: "Nursing", 13: "Oral Hygiene",
    14: "Advertising and Marketing Management",
    15: "Journalism and Communication", 16: "Basic Education",
    17: "Management (evening attendance)",
}
PREV_QUAL_LABELS = {
    1: "Secondary education", 2: "Higher education - bachelor's degree",
    3: "Higher education - degree", 4: "Higher education - master's degree",
    5: "Higher education - doctorate", 6: "Frequency of higher education",
    7: "12th year of schooling - not completed",
    8: "11th year of schooling - not completed",
    9: "Other - 11th year of schooling", 10: "10th year of schooling",
    11: "10th year of schooling - not completed",
    12: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
    13: "Basic education 2nd cycle (6th/7th year) or equiv.",
    14: "Technological specialization course",
    15: "Higher education - degree (1st cycle)",
    16: "Professional higher technical course",
    17: "Higher education - master (2nd cycle)",
}
NATIONALITY_LABELS = {
    1: "Portuguese", 2: "German", 3: "Spanish", 4: "Italian", 5: "Dutch",
    6: "English", 7: "Lithuanian", 8: "Angolan", 9: "Cape Verdean",
    10: "Guinean", 11: "Mozambican", 12: "Santomean", 13: "Turkish",
    14: "Brazilian", 15: "Romanian", 16: "Moldova (Republic of)",
    17: "Mexican", 18: "Ukrainian", 19: "Russian", 20: "Cuban",
    21: "Colombian",
}


def _code_selectbox(col, label_widget, mapping: dict, default_label: str, session_key: str):
    """Selectbox over a code->label dict whose keys are contiguous 1..N in
    display order, so `options.index(selected) + 1` recovers the code."""
    options = list(mapping.values())
    default_label = default_label if default_label in options else options[0]
    current = st.session_state.get(session_key, default_label)
    current = current if current in options else options[0]
    selected = col.selectbox(label_widget, options, index=options.index(current), key=session_key)
    return options.index(selected) + 1

# ---------------------------------------------------------------------------
# Example profiles: pre-filled widget values so the app is usable in one
# click, without a first-time user having to understand 30 coded fields.
# ---------------------------------------------------------------------------
BASE_WIDGET_DEFAULTS = {
    "marital_status": "Single", "nationality": "Portuguese", "gender": "Female", "age": 20,
    "international": "No", "displaced": "No", "debtor": "No",
    "tuition_up_to_date": "Yes", "scholarship": "No", "special_needs": "No",
    "mother_qual": 1, "father_qual": 1, "mother_occ": 1, "father_occ": 1,
    "application_mode": "1st phase - general contingent", "application_order": 1,
    "course": "Management", "prev_qual": "Secondary education",
    "attendance": "Daytime",
    "cu1_credited": 0, "cu1_enrolled": 6, "cu1_eval": 6, "cu1_approved": 6,
    "cu1_grade": 14.0, "cu1_no_eval": 0,
    "cu2_credited": 0, "cu2_enrolled": 6, "cu2_eval": 6, "cu2_approved": 6,
    "cu2_grade": 14.0, "cu2_no_eval": 0,
    "unemployment": 11.1, "inflation": 1.2, "gdp": 0.3,
}

EXAMPLE_PROFILES = {
    "🔴 Load a high-risk example": {
        **BASE_WIDGET_DEFAULTS,
        "cu1_approved": 0, "cu1_grade": 0.0, "cu1_eval": 6,
        "cu2_approved": 0, "cu2_grade": 0.0, "cu2_eval": 6,
        "tuition_up_to_date": "No", "debtor": "Yes", "age": 27,
    },
    "🟡 Load a borderline example": {
        **BASE_WIDGET_DEFAULTS,
        "cu1_approved": 2, "cu1_grade": 10.0,
        "cu2_approved": 2, "cu2_grade": 10.0,
    },
    "🟢 Load a strong example": dict(BASE_WIDGET_DEFAULTS),
}


def _apply_example(values: dict):
    for key, val in values.items():
        st.session_state[f"f_{key}"] = val


@st.cache_resource
def _artifacts_exist():
    return config.MODEL_PATH.exists() and config.PREPROCESSOR_PATH.exists()


def inject_css():
    st.markdown(
        """
        <style>
        .hero-number { font-size: 3.2rem; font-weight: 700; line-height: 1; margin: 0; }
        .band-badge {
            display: inline-block; padding: 0.35em 0.9em; border-radius: 999px;
            font-weight: 600; font-size: 0.95rem; color: white;
        }
        .meter-track {
            width: 100%; height: 14px; border-radius: 999px; overflow: hidden; margin-top: 0.4em;
        }
        .meter-fill { height: 100%; border-radius: 999px; }
        .factor-row {
            display: flex; align-items: baseline; gap: 0.5em; padding: 0.3em 0;
            border-bottom: 1px solid rgba(128,128,128,0.15);
        }
        .action-card {
            border-radius: 0.6em; padding: 1em 1.2em; border-left: 6px solid;
        }
        .stat-tile {
            border-radius: 0.6em; padding: 0.8em 1em; text-align: center;
        }
        .stat-tile .value { font-size: 1.8rem; font-weight: 700; }
        .stat-tile .label { font-size: 0.85rem; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_report(report: dict):
    band = report["risk_band"]
    s = STATUS[band]

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown(f"<p class='hero-number' style='color:{s['color']}'>{report['p_dropout']:.0%}</p>", unsafe_allow_html=True)
        st.caption("Predicted probability of dropout")
        st.markdown(
            f"<span class='band-badge' style='background:{s['color']}'>{s['icon']} {s['word']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""<div class='meter-track' style='background:{s['track']}'>
                    <div class='meter-fill' style='width:{report['p_dropout']*100:.0f}%; background:{s['color']}'></div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Predicted outcome:** {report['predicted_class']}")
        with st.expander("Full class probabilities"):
            for cls, p in report["probabilities"].items():
                st.write(f"{cls}: {p:.1%}")

    with right:
        st.markdown("**Top contributing factors**")
        for f in report["factors_detail"]:
            icon = DIRECTION_ICON.get(f["direction"], "◆")
            st.markdown(
                f"<div class='factor-row'><span>{icon}</span><span>{f['text']}</span></div>",
                unsafe_allow_html=True,
            )
        st.caption("🔺 raises risk 🔻 lowers risk")

    st.markdown(
        f"""<div class='action-card' style='border-color:{s['color']}; background:{s['track']}22'>
                <b>Recommended action</b><br>{report['recommended_action']}
            </div>""",
        unsafe_allow_html=True,
    )


def single_student_mode():
    st.header("Single-student risk assessment")
    st.caption(
        "The fields that drive most of the risk score are up front. Everything else "
        "is optional and tucked into the sections below — sensible defaults are "
        "already filled in."
    )

    ex_cols = st.columns(len(EXAMPLE_PROFILES))
    for col, (label, values) in zip(ex_cols, EXAMPLE_PROFILES.items()):
        col.button(label, use_container_width=True, on_click=_apply_example, args=(values,))

    st.divider()

    def sv(key, default):
        return st.session_state.get(f"f_{key}", default)

    with st.form("student_form"):
        st.markdown("#### 📈 Academic performance *(strongest risk signal)*")
        c1, c2, c3 = st.columns(3)
        cu1_approved = c1.number_input("1st sem: units approved", 0, 26, sv("cu1_approved", 6), key="f_cu1_approved")
        cu1_grade = c2.number_input("1st sem: average grade (0-20)", 0.0, 20.0, sv("cu1_grade", 14.0), key="f_cu1_grade")
        cu1_enrolled = c3.number_input("1st sem: units enrolled", 0, 26, sv("cu1_enrolled", 6), key="f_cu1_enrolled")
        c1, c2, c3 = st.columns(3)
        cu2_approved = c1.number_input("2nd sem: units approved", 0, 20, sv("cu2_approved", 6), key="f_cu2_approved")
        cu2_grade = c2.number_input("2nd sem: average grade (0-20)", 0.0, 20.0, sv("cu2_grade", 14.0), key="f_cu2_grade")
        cu2_enrolled = c3.number_input("2nd sem: units enrolled", 0, 23, sv("cu2_enrolled", 6), key="f_cu2_enrolled")

        with st.expander("More academic detail (credited / evaluated units)"):
            c1, c2, c3 = st.columns(3)
            cu1_credited = c1.number_input("1st sem: credited units", 0, 20, sv("cu1_credited", 0), key="f_cu1_credited")
            cu1_eval = c2.number_input("1st sem: evaluations taken", 0, 45, sv("cu1_eval", 6), key="f_cu1_eval")
            cu1_no_eval = c3.number_input("1st sem: without evaluations", 0, 12, sv("cu1_no_eval", 0), key="f_cu1_no_eval")
            c1, c2, c3 = st.columns(3)
            cu2_credited = c1.number_input("2nd sem: credited units", 0, 20, sv("cu2_credited", 0), key="f_cu2_credited")
            cu2_eval = c2.number_input("2nd sem: evaluations taken", 0, 33, sv("cu2_eval", 6), key="f_cu2_eval")
            cu2_no_eval = c3.number_input("2nd sem: without evaluations", 0, 12, sv("cu2_no_eval", 0), key="f_cu2_no_eval")

        st.markdown("#### 💰 Financial & scholarship status *(strong secondary signal)*")
        c1, c2, c3 = st.columns(3)
        tuition_up_to_date = c1.selectbox("Tuition fees up to date?", ["Yes", "No"], index=["Yes", "No"].index(sv("tuition_up_to_date", "Yes")), key="f_tuition_up_to_date")
        debtor = c2.selectbox("Has outstanding debt?", ["No", "Yes"], index=["No", "Yes"].index(sv("debtor", "No")), key="f_debtor")
        scholarship = c3.selectbox("Scholarship holder?", ["No", "Yes"], index=["No", "Yes"].index(sv("scholarship", "No")), key="f_scholarship")

        st.markdown("#### 🧑 Demographics")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age at enrollment", 17, 70, sv("age", 20), key="f_age")
        gender = c2.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(sv("gender", "Female")), key="f_gender")

        with st.expander("More demographic detail"):
            c1, c2, c3 = st.columns(3)
            marital_status = _code_selectbox(c1, "Marital status", MARITAL_STATUS_LABELS, "Single", "f_marital_status")
            nationality = _code_selectbox(c2, "Nationality", NATIONALITY_LABELS, "Portuguese", "f_nationality")
            special_needs = c3.selectbox("Educational special needs?", ["No", "Yes"], index=["No", "Yes"].index(sv("special_needs", "No")), key="f_special_needs")
            c1, c2 = st.columns(2)
            international = c1.selectbox("International student?", ["No", "Yes"], index=["No", "Yes"].index(sv("international", "No")), key="f_international")
            displaced = c2.selectbox("Displaced student?", ["No", "Yes"], index=["No", "Yes"].index(sv("displaced", "No")), key="f_displaced")

        with st.expander("Institution-specific admission codes (optional — leave as-is if unknown)"):
            st.caption(
                "Course, application mode, and previous qualification below use the "
                "published UCI dataset codebook. Parents' qualification/occupation "
                "remain numeric codes -- the source dataset re-indexed those two "
                "columns independently, so a single label list can't be reliably "
                "matched to both columns without the institution's original codebook. "
                "All of these contribute far less to the risk score than academic "
                "performance and financial status above."
            )
            c1, c2 = st.columns(2)
            application_mode = _code_selectbox(c1, "Application mode", APPLICATION_MODE_LABELS, "1st phase - general contingent", "f_application_mode")
            application_order = c2.number_input("Application order (1st choice = 1)", 0, 9, sv("application_order", 1), key="f_application_order")
            c1, c2 = st.columns(2)
            course = _code_selectbox(c1, "Course", COURSE_LABELS, "Management", "f_course")
            prev_qual = _code_selectbox(c2, "Previous qualification", PREV_QUAL_LABELS, "Secondary education", "f_prev_qual")
            attendance = st.selectbox("Attendance", ["Daytime", "Evening"], index=["Daytime", "Evening"].index(sv("attendance", "Daytime")), key="f_attendance")
            c1, c2 = st.columns(2)
            mother_qual = c1.number_input("Mother's qualification (code, 1-29)", 1, 29, sv("mother_qual", 1), key="f_mother_qual")
            father_qual = c2.number_input("Father's qualification (code, 1-34)", 1, 34, sv("father_qual", 1), key="f_father_qual")
            c1, c2 = st.columns(2)
            mother_occ = c1.number_input("Mother's occupation (code, 1-32)", 1, 32, sv("mother_occ", 1), key="f_mother_occ")
            father_occ = c2.number_input("Father's occupation (code, 1-46)", 1, 46, sv("father_occ", 1), key="f_father_occ")

        with st.expander("Macroeconomic context at enrollment (optional)"):
            c1, c2, c3 = st.columns(3)
            unemployment = c1.number_input("Unemployment rate (%)", 0.0, 30.0, sv("unemployment", 11.1), key="f_unemployment")
            inflation = c2.number_input("Inflation rate (%)", -5.0, 10.0, sv("inflation", 1.2), key="f_inflation")
            gdp = c3.number_input("GDP growth", -10.0, 10.0, sv("gdp", 0.3), key="f_gdp")

        student_id = st.text_input("Student ID / name (for your reference only)", "")
        submitted = st.form_submit_button("Assess risk", type="primary", use_container_width=True)

    if submitted:
        yn = {"Yes": 1, "No": 0}
        row = {
            "Marital status": marital_status, "Application mode": application_mode,
            "Course": course, "Previous qualification": prev_qual, "Nacionality": nationality,
            "Mother's qualification": mother_qual, "Father's qualification": father_qual,
            "Mother's occupation": mother_occ, "Father's occupation": father_occ,
            "Daytime/evening attendance": 1 if attendance == "Daytime" else 0,
            "Displaced": yn[displaced], "Educational special needs": yn[special_needs],
            "Debtor": yn[debtor], "Tuition fees up to date": yn[tuition_up_to_date],
            "Gender": 1 if gender == "Male" else 0, "Scholarship holder": yn[scholarship],
            "International": yn[international], "Application order": application_order,
            "Age at enrollment": age,
            "Curricular units 1st sem (credited)": cu1_credited,
            "Curricular units 1st sem (enrolled)": cu1_enrolled,
            "Curricular units 1st sem (evaluations)": cu1_eval,
            "Curricular units 1st sem (approved)": cu1_approved,
            "Curricular units 1st sem (grade)": cu1_grade,
            "Curricular units 1st sem (without evaluations)": cu1_no_eval,
            "Curricular units 2nd sem (credited)": cu2_credited,
            "Curricular units 2nd sem (enrolled)": cu2_enrolled,
            "Curricular units 2nd sem (evaluations)": cu2_eval,
            "Curricular units 2nd sem (approved)": cu2_approved,
            "Curricular units 2nd sem (grade)": cu2_grade,
            "Curricular units 2nd sem (without evaluations)": cu2_no_eval,
            "Unemployment rate": unemployment, "Inflation rate": inflation, "GDP": gdp,
        }
        raw_row = pd.DataFrame([row])[FEATURE_COLS]
        label = student_id.strip() or "This student"
        report = build_student_report(raw_row, student_label=label)
        st.divider()
        render_report(report)


def batch_mode():
    st.header("Batch upload & triage")
    st.caption(
        "Upload a CSV with the same columns as the source dataset. Every row is "
        "scored and ranked by P(Dropout) so staff can triage the highest-risk "
        "students first."
    )

    sample_path = config.DATA_DIR / "sample_batch.csv"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            st.download_button("⬇ Download a sample CSV to try", f, file_name="sample_batch.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload student records CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
        df.columns = [c.replace("﻿", "").strip() for c in df.columns]
        with st.spinner(f"Scoring {len(df)} students..."):
            ranked = score_batch(df)
    except Exception as e:
        st.error(f"Could not score this file: {e}")
        return

    st.success(f"Scored {len(ranked)} students.")

    counts = ranked["risk_band"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
    tile_cols = st.columns(3)
    for col, band in zip(tile_cols, ["High", "Medium", "Low"]):
        s = STATUS[band]
        col.markdown(
            f"""<div class='stat-tile' style='background:{s['track']}22; border:1px solid {s['color']}55'>
                    <div class='value' style='color:{s['color']}'>{s['icon']} {int(counts[band])}</div>
                    <div class='label'>{band} risk</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    def _highlight(row):
        color = STATUS.get(row["risk_band"], {}).get("track", "#ffffff")
        return [f"background-color: {color}55"] * len(row)

    st.dataframe(
        ranked.style.apply(_highlight, axis=1).format({"p_dropout": "{:.1%}"}),
        use_container_width=True,
        height=500,
    )

    csv_bytes = ranked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download ranked at-risk list (CSV)", csv_bytes,
        file_name="ranked_at_risk_students.csv", mime="text/csv", type="primary",
    )


def main():
    inject_css()
    st.title("🎓 Student Dropout Early Intervention System")
    st.caption(
        "Predicts Dropout / Enrolled / Graduate outcomes and turns P(Dropout) into a "
        "ranked, explainable, actionable early-intervention signal."
    )

    if not _artifacts_exist():
        st.error("No trained model found. Run `python -m src.main train` first, then relaunch this app.")
        st.stop()

    with st.sidebar:
        st.header("🎓 Navigation")
        mode = st.radio("Mode", ["Single-student form", "Batch upload & triage"], label_visibility="collapsed")
        st.divider()
        st.info(DISCLAIMER)

    if mode == "Single-student form":
        single_student_mode()
    else:
        batch_mode()

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
