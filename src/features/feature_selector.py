from typing import List

import pandas as pd

from src.config import (
    COLUMNS_EXCLUDED_FROM_FEATURES,
    CORRELATION_THRESHOLD,
    FALLBACK_TOP_N_FEATURES,
    TARGET_COLUMN,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Selects predictive features via correlation with the target."""

    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        threshold: float = CORRELATION_THRESHOLD,
        fallback_top_n: int = FALLBACK_TOP_N_FEATURES,
    ):
        self.target_column = target_column
        self.threshold = threshold
        self.fallback_top_n = fallback_top_n
        self.selected_features_: List[str] = []
        self.correlation_scores_: pd.Series = pd.Series(dtype=float)

    def select(self, df_encoded: pd.DataFrame) -> List[str]:
        """
        Computes correlation of every numeric column with the target
        and returns the list of selected feature names. `df_encoded`
        must already have categorical columns numerically encoded.
        """
        numeric_df = df_encoded.select_dtypes(include="number")
        self.correlation_scores_ = (
            numeric_df.corr()[self.target_column].abs().sort_values(ascending=False)
        )

        candidates = [
            f for f in self.correlation_scores_.index
            if f not in COLUMNS_EXCLUDED_FROM_FEATURES
        ]
        selected = [f for f in candidates if self.correlation_scores_[f] > self.threshold]

        if not selected:
            logger.warning(
                "No feature cleared the %.3f correlation threshold - "
                "falling back to top %d features", self.threshold, self.fallback_top_n
            )
            selected = candidates[: self.fallback_top_n]

        self.selected_features_ = selected
        logger.info("Selected features: %s", selected)
        return selected