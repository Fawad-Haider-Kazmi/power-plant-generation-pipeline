from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import CATEGORICAL_COLUMNS_TO_CLEAN, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Cleans and preprocesses the raw power plant DataFrame."""

    def __init__(self, target_column: str = TARGET_COLUMN):
        self.target_column = target_column
        self.cleaning_report: Dict = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the full cleaning sequence and returns a cleaned copy."""
        df = df.copy()
        n_before = len(df)

        df = df.dropna(subset=[self.target_column])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_column]
        missing_numeric_before = int(df[numeric_cols].isnull().sum().sum())
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        missing_categorical_before = int(df[categorical_cols].isnull().sum().sum())
        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

        for col in CATEGORICAL_COLUMNS_TO_CLEAN:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()

        duplicates_removed = int(df.duplicated().sum())
        df = df.drop_duplicates()

        outlier_count = self._count_iqr_outliers(df, self.target_column)

        self.cleaning_report = {
            "rows_before": n_before,
            "rows_after": int(len(df)),
            "rows_dropped_missing_target": n_before - len(df) - duplicates_removed,
            "duplicates_removed": duplicates_removed,
            "missing_numeric_imputed": missing_numeric_before,
            "missing_categorical_filled": missing_categorical_before,
            "target_outliers_flagged": outlier_count,
        }
        logger.info("Cleaning report: %s", self.cleaning_report)
        return df

    @staticmethod
    def _count_iqr_outliers(df: pd.DataFrame, column: str) -> int:
        """Counts (does not remove) IQR-based outliers in `column`."""
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return int(((df[column] < lower) | (df[column] > upper)).sum())

    def encode_categoricals(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Data Preprocessing stage: label-encodes the given categorical
        columns in place on a copy, fitting and storing one encoder per
        column so the same mapping could be reused on new data later.
        """
        df = df.copy()
        for col in columns:
            if col not in df.columns:
                continue
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            self.label_encoders[col] = encoder
        logger.info("Encoded categorical columns: %s", [c for c in columns if c in df.columns])
        return df