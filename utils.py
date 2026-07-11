"""Small shared helpers used across the pipeline."""
import sys

import pandas as pd


def section(title: str) -> None:
    """Print a visually distinct section header to stdout."""
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}", file=sys.stdout)


def load_raw_csv(path) -> pd.DataFrame:
    """Load a student-records CSV, stripping BOM/whitespace from column
    names so downstream code can rely on exact column-name matches
    regardless of how the CSV was exported. This file's BOM was
    double-encoded (UTF-8 BOM bytes re-saved as if Latin-1), which
    utf-8-sig alone does not catch, so we also strip the resulting
    mojibake prefix "ï»¿" if present."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect the target column: the one whose values include
    Dropout/Enrolled/Graduate labels, rather than assuming a fixed name."""
    candidates = {"dropout", "enrolled", "graduate"}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            values = set(str(v).strip().lower() for v in df[col].dropna().unique())
            if candidates & values:
                return col
    raise ValueError(
        "Could not auto-detect target column: no column contains "
        "Dropout/Enrolled/Graduate labels."
    )
