from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import RAW_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads the raw power plant dataset and profiles it."""

    def __init__(self, raw_path: Path = RAW_DATA_PATH):
        self.raw_path = raw_path

    def load(self) -> pd.DataFrame:
        """
        Data Loading stage.

        Data Collection note: the source file is downloaded manually
        from the WRI Global Power Plant Database
        (https://datasets.wri.org/dataset/globalpowerplantdatabase) and
        placed at `self.raw_path`. This method is the single entry
        point for raw data into the pipeline.
        """
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Raw dataset not found at {self.raw_path}. "
                "Download the Global Power Plant Database CSV and place "
                "it at this path before running the pipeline."
            )
        logger.info("Loading raw dataset from %s", self.raw_path)
        df = pd.read_csv(self.raw_path)
        logger.info("Loaded dataframe with shape %s", df.shape)
        return df

    @staticmethod
    def summarize(df: pd.DataFrame) -> Dict:
        """
        Data Understanding stage.

        Returns a structured summary: shape, dtypes, missing values,
        and descriptive statistics for numeric columns.
        """
        summary = {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "describe": df.describe(include="all").to_dict(),
        }
        logger.info(
            "Dataset understanding: %d rows, %d columns, %d missing cells total",
            summary["n_rows"], summary["n_columns"], sum(
                v for v in df.isnull().sum().to_dict().values()
            ),
        )
        return summary