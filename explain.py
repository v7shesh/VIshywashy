"""
Explainability: global feature importance and per-student explanations.

Uses SHAP TreeExplainer (the saved model is tree-based) when available,
falling back to the model's built-in tree feature_importances_ if SHAP
cannot be computed. This is what makes the system's output actionable for
an academic advisor -- not just a probability, but *why*.
"""
import warnings

import joblib
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import get_raw_feature_map, load_preprocessor
from src.utils import section

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _load_artifacts():
    model = joblib.load(config.MODEL_PATH)
    preprocessor = load_preprocessor()
    metadata = joblib.load(config.METADATA_PATH)
    return model, preprocessor, metadata


def _aggregate_by_raw_feature(values: np.ndarray, raw_feature_map: list) -> pd.Series:
    """Sum |value| across one-hot columns that share the same raw feature."""
    s = pd.Series(values, index=raw_feature_map)
    return s.groupby(level=0).sum().sort_values(ascending=False)


def global_feature_importance(sample_size: int = 500, top_n: int = 15) -> pd.DataFrame:
    """Mean |SHAP value| for the Dropout class, aggregated to raw feature
    names, computed on a sample of the training-style data for speed."""
    model, preprocessor, metadata = _load_artifacts()
    from src.preprocessing import load_clean_data
    X, y, _ = load_clean_data()
    raw_feature_map = get_raw_feature_map(preprocessor)
    X_t = preprocessor.transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    rng = np.random.RandomState(config.RANDOM_STATE)
    n = min(sample_size, X_t.shape[0])
    idx = rng.choice(X_t.shape[0], size=n, replace=False)
    X_sample = X_t[idx]

    classes = list(model.classes_)
    dropout_idx = classes.index("Dropout")

    if SHAP_AVAILABLE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        # shap_values shape: (n_samples, n_features, n_classes) for multi-class RF in recent SHAP
        if isinstance(shap_values, list):
            class_shap = shap_values[dropout_idx]
        elif shap_values.ndim == 3:
            class_shap = shap_values[:, :, dropout_idx]
        else:
            class_shap = shap_values
        mean_abs = np.abs(class_shap).mean(axis=0)
        method = "SHAP (mean |SHAP value| for Dropout class)"
    else:
        mean_abs = model.feature_importances_
        method = "Tree feature_importances_ (SHAP unavailable, fallback)"

    agg = _aggregate_by_raw_feature(mean_abs, raw_feature_map)
    result = agg.head(top_n).reset_index()
    result.columns = ["feature", "importance"]
    result.attrs["method"] = method
    return result


def explain_student(raw_row: pd.DataFrame, top_n: int = 5) -> dict:
    """Explain one student's prediction: top factors pushing risk up/down.

    raw_row: single-row DataFrame with the raw (untransformed) feature
    columns expected by the preprocessor (see preprocessing.get_feature_columns).
    Returns predicted class, P(Dropout), and ranked contributing factors.
    """
    model, preprocessor, metadata = _load_artifacts()
    raw_feature_map = get_raw_feature_map(preprocessor)
    classes = list(model.classes_)
    dropout_idx = classes.index("Dropout")

    X_t = preprocessor.transform(raw_row)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    proba = model.predict_proba(X_t)[0]
    pred_class = classes[int(np.argmax(proba))]
    p_dropout = float(proba[dropout_idx])

    factors = []
    if SHAP_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):
                row_shap = shap_values[dropout_idx][0]
            elif shap_values.ndim == 3:
                row_shap = shap_values[0, :, dropout_idx]
            else:
                row_shap = shap_values[0]
            signed = pd.Series(row_shap, index=raw_feature_map).groupby(level=0).sum()
            ranked = signed.reindex(signed.abs().sort_values(ascending=False).index)
            for feat, val in ranked.head(top_n).items():
                direction = "increases" if val > 0 else "decreases"
                factors.append({
                    "feature": feat,
                    "raw_value": raw_row.iloc[0][feat] if feat in raw_row.columns else None,
                    "impact": float(val),
                    "direction": direction,
                })
        except Exception:
            factors = []

    if not factors:
        # Fallback: rank by global tree importance, direction unknown.
        importances = _aggregate_by_raw_feature(model.feature_importances_, raw_feature_map)
        for feat, val in importances.head(top_n).items():
            factors.append({
                "feature": feat,
                "raw_value": raw_row.iloc[0][feat] if feat in raw_row.columns else None,
                "impact": float(val),
                "direction": "influential (direction unavailable without SHAP)",
            })

    return {
        "predicted_class": pred_class,
        "p_dropout": p_dropout,
        "probabilities": dict(zip(classes, proba.tolist())),
        "top_factors": factors,
    }


def run_explain_report() -> None:
    section("P5 - EXPLAINABILITY: GLOBAL FEATURE IMPORTANCE")
    result = global_feature_importance()
    method = result.attrs.get("method", "unknown")
    print(f"Method: {method}\n")
    print(result.to_string(index=False))

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_df = result.iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["importance"], color="#2a6f97")
        ax.set_xlabel("Mean |impact| on Dropout risk")
        ax.set_title("Top factors driving dropout risk (global)")
        fig.tight_layout()
        out_path = config.FIGURES_DIR / "global_feature_importance.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nSaved chart: {out_path}")
    except Exception as e:
        print(f"(Skipped saving chart: {e})")

    section("P5 SUMMARY")
    top3 = ", ".join(result["feature"].head(3).tolist())
    print(
        f"- Explainability method: {method}\n"
        f"- Top 3 global risk drivers: {top3}\n"
        "- Per-student explanations available via src.explain.explain_student()\n"
        "  (used by the intervention engine and the Streamlit app)\n"
        "Review above, then continue to P6 (intervention logic)."
    )


if __name__ == "__main__":
    run_explain_report()
