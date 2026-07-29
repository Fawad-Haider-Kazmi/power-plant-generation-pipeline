from pathlib import Path

import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Predictor:
    """Loads a trained model and generates predictions for new data."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. "
                "Run the training pipeline first."
            )
        logger.info("Loading model from %s", self.model_path)
        return joblib.load(self.model_path)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Returns predictions for the given feature rows."""
        logger.info("Generating predictions for %d rows", len(features))
        predictions = self.model.predict(features)
        return pd.Series(predictions, index=features.index, name="predicted_generation_gwh")

    @classmethod
    def from_model_name(cls, name: str, models_dir: Path = MODELS_DIR) -> "Predictor":
        """Convenience constructor: `Predictor.from_model_name("random_forest_tuned")`."""
        return cls(models_dir / f"{name}.pkl")