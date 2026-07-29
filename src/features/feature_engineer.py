import numpy as np
import pandas as pd

from src.config import CURRENT_YEAR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Derives engineered features for the power plant generation model."""

    def __init__(self, current_year: int = CURRENT_YEAR):
        self.current_year = current_year

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of `df` with engineered features added."""
        df = df.copy()

        if "commissioning_year" in df.columns:
            df["plant_age_years"] = (self.current_year - df["commissioning_year"]).clip(lower=0)
            logger.info("Derived plant_age_years from commissioning_year")

        if "capacity_mw" in df.columns:
            df["capacity_mw"] = df["capacity_mw"].replace(0, np.nan)
            df["capacity_mw"] = df["capacity_mw"].fillna(df["capacity_mw"].median())

        return df