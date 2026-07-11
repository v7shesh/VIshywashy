"""
Model training, cross-validation, and evaluation.

Trains three candidate classifiers (Logistic Regression baseline, Random
Forest, Gradient Boosting), all 3-class (Dropout/Enrolled/Graduate), with
class-imbalance handling. Selection criterion is explicit and documented:
this is a sensitive domain where missing an at-risk student (a false
negative on Dropout) is worse than a false alarm, so among models with
competitive macro-F1 we prioritize **Dropout-class recall** for final
selection, not raw accuracy.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.preprocessing import (
    fit_and_save_preprocessor,
    load_clean_data,
    split_data,
)
from src.utils import section

DROPOUT_LABEL = "Dropout"


def build_candidates() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=config.RANDOM_STATE,
        ),
    }


def cross_validate_model(name, model, X, y):
    """5-fold stratified CV is used for the headline numbers (not a single
    train/test split) so reported metrics reflect stability across folds.
    (GradientBoosting has no class_weight param; sample-weighting is applied
    only at final-fit time in run_training, to keep CV scoring simple.)"""
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro",
        "macro_f1": "f1_macro",
        "dropout_recall": _dropout_recall_scorer,
    }
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    summary = {k: (results[f"test_{k}"].mean(), results[f"test_{k}"].std()) for k in scoring}
    return summary


def _dropout_recall_scorer(estimator, X, y):
    preds = estimator.predict(X)
    return recall_score(y, preds, labels=[DROPOUT_LABEL], average="macro", zero_division=0)


def evaluate_on_test(name, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=sorted(y_test.unique()))
    return {
        "name": name,
        "accuracy": accuracy_score(y_test, preds),
        "macro_precision": precision_score(y_test, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, preds, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "dropout_recall": recall_score(y_test, preds, labels=[DROPOUT_LABEL], average="macro", zero_division=0),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": sorted(y_test.unique().tolist()),
    }


def run_training() -> dict:
    section("P2 - PREPROCESSING")
    X, y, target_col = load_clean_data()
    print(f"Loaded {len(X)} rows, target='{target_col}'")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows (stratified {int(config.TEST_SIZE*100)}% test)")

    preprocessor = fit_and_save_preprocessor(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    print(f"Preprocessor fit on train and saved to {config.PREPROCESSOR_PATH}")
    print(f"Transformed feature count: {X_train_t.shape[1]}")

    section("P4 - MODELING: CROSS-VALIDATED CANDIDATE COMPARISON")
    candidates = build_candidates()
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    cv_summaries = {}
    for name, model in candidates.items():
        summary = cross_validate_model(name, model, X_train_t, y_train)
        cv_summaries[name] = summary
        print(f"\n{name} (5-fold stratified CV on training data):")
        for metric, (mean, std) in summary.items():
            print(f"  {metric:16s}: {mean:.4f} (+/- {std:.4f})")

    section("Final fit on full training split + held-out test evaluation")
    test_results = {}
    fitted_models = {}
    for name, model in candidates.items():
        if name == "GradientBoosting":
            model.fit(X_train_t, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train_t, y_train)
        fitted_models[name] = model
        result = evaluate_on_test(name, model, X_test_t, y_test)
        test_results[name] = result
        print(f"\n{name} - held-out test set:")
        print(f"  accuracy       : {result['accuracy']:.4f}")
        print(f"  macro precision: {result['macro_precision']:.4f}")
        print(f"  macro recall   : {result['macro_recall']:.4f}")
        print(f"  macro F1       : {result['macro_f1']:.4f}")
        print(f"  Dropout recall : {result['dropout_recall']:.4f}  <-- prioritized metric")
        print(f"  Confusion matrix (labels={result['labels']}):")
        for row in result["confusion_matrix"]:
            print(f"    {row}")
        print(f"  Per-class report:")
        for label, metrics in result["classification_report"].items():
            if isinstance(metrics, dict):
                print(f"    {label:12s} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1-score']:.3f} support={int(metrics['support'])}")

    section("Model selection")
    print(
        "Selection rule: among candidates, pick the one with the highest\n"
        "Dropout-class recall on the held-out test set, using macro-F1 as a\n"
        "tie-breaker if Dropout recall is within 0.01 of the top score.\n"
        "Rationale: in this domain a false negative (an at-risk student who\n"
        "is not flagged) is worse than a false positive (a routine check-in\n"
        "on a student who was not actually at risk), so recall on the\n"
        "Dropout class is prioritized over overall accuracy."
    )
    best_dropout_recall = max(r["dropout_recall"] for r in test_results.values())
    close_candidates = [
        name for name, r in test_results.items()
        if best_dropout_recall - r["dropout_recall"] <= 0.01
    ]
    best_name = max(close_candidates, key=lambda n: test_results[n]["macro_f1"])
    best_model = fitted_models[best_name]
    print(f"\nSelected model: {best_name}")
    print(f"  Dropout recall : {test_results[best_name]['dropout_recall']:.4f}")
    print(f"  Macro F1       : {test_results[best_name]['macro_f1']:.4f}")
    print(f"  Accuracy       : {test_results[best_name]['accuracy']:.4f}")

    joblib.dump(best_model, config.MODEL_PATH)
    metadata = {
        "best_model_name": best_name,
        "target_col": target_col,
        "classes": sorted(y.unique().tolist()),
        "dropout_label": DROPOUT_LABEL,
        "feature_columns": list(X.columns),
        "cv_summaries": {
            name: {k: {"mean": v[0], "std": v[1]} for k, v in summ.items()}
            for name, summ in cv_summaries.items()
        },
        "test_results": {
            name: {k: v for k, v in r.items() if k != "classification_report"}
            for name, r in test_results.items()
        },
        "full_classification_reports": {
            name: r["classification_report"] for name, r in test_results.items()
        },
    }
    joblib.dump(metadata, config.METADATA_PATH)
    with open(config.MODELS_DIR / "metrics_report.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved best model to {config.MODEL_PATH}")
    print(f"Saved metadata/metrics to {config.METADATA_PATH} and models/metrics_report.json")

    section("P4 SUMMARY")
    print(
        f"- Best model: {best_name}\n"
        f"- Test accuracy: {test_results[best_name]['accuracy']:.3f}\n"
        f"- Test macro F1: {test_results[best_name]['macro_f1']:.3f}\n"
        f"- Test Dropout recall: {test_results[best_name]['dropout_recall']:.3f} "
        "(prioritized per project brief)\n"
        "Review metrics above, then continue to P5 (explainability)."
    )
    return metadata


if __name__ == "__main__":
    run_training()
