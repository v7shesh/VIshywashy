"""
Data cleaning, feature encoding, and train/test splitting.

A single fitted scikit-learn ColumnTransformer is the source of truth for
turning raw student records into model-ready features. It is fit once on
the training split and reused (never refit) at inference time, so training
and serving always apply the exact same transform.
"""
from typing import Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.utils import load_raw_csv, detect_target_column


def get_feature_columns() -> list:
    """All model input columns, in a fixed order."""
    return config.NOMINAL_FEATURES + config.BINARY_FEATURES + config.NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer: impute + one-hot encode nominal
    categoricals, impute + scale numeric/binary features. Median/most-frequent
    imputation is defensive (the source data has no missing values today,
    per the P1 audit) so the pipeline degrades gracefully on future data."""
    nominal_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("nominal", nominal_pipeline, config.NOMINAL_FEATURES),
            ("numeric", numeric_pipeline, config.BINARY_FEATURES + config.NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def load_clean_data() -> Tuple[pd.DataFrame, pd.Series, str]:
    """Load the raw CSV, verify expected columns exist, and return
    (X, y, target_col). Rows with a missing/unparseable target are dropped."""
    df = load_raw_csv(config.RAW_CSV_PATH)
    target_col = detect_target_column(df)

    feature_cols = get_feature_columns()
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected feature columns not found in CSV: {missing_cols}. "
            "The dataset schema may differ from the UCI/Kaggle version this "
            "pipeline was built for -- update src/config.py feature lists."
        )

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df[feature_cols].copy()
    y = df[target_col].astype(str).str.strip()
    return X, y, target_col


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified train/test split so class proportions (Dropout/Enrolled/
    Graduate) are preserved in both splits despite the ~2.8x imbalance."""
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def fit_and_save_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    return joblib.load(config.PREPROCESSOR_PATH)


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable names for the transformed feature matrix columns,
    used by explain.py to label importances/SHAP values meaningfully."""
    return list(preprocessor.get_feature_names_out())


def get_raw_feature_map(preprocessor: ColumnTransformer) -> list:
    """Map each transformed output column back to its original raw feature
    name (e.g. every one-hot column of 'Course' maps back to 'Course').
    Lets explain.py aggregate SHAP/importance values to human-readable,
    student-facing feature names instead of exposing one-hot internals."""
    raw_names = []
    nominal_pipeline = preprocessor.named_transformers_["nominal"]
    onehot = nominal_pipeline.named_steps["onehot"]
    for col_name, categories in zip(config.NOMINAL_FEATURES, onehot.categories_):
        raw_names.extend([col_name] * len(categories))
    raw_names.extend(config.BINARY_FEATURES + config.NUMERIC_FEATURES)
    return raw_names
