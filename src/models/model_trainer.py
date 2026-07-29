from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.config import MODELS_DIR, RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Trains and saves the candidate regression models."""

    def __init__(self, models_dir: Path = MODELS_DIR, random_state: int = RANDOM_STATE):
        self.models_dir = models_dir
        self.random_state = random_state
      
        self.models: Dict[str, object] = {
            "random_forest": RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.1, random_state=random_state
            ),
        }
        self.trained_models: Dict[str, object] = {}

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
        """Fits every registered model and returns them keyed by name."""
        for name, model in self.models.items():
            logger.info("Training %s", name)
            model.fit(X_train, y_train)
            self.trained_models[name] = model
        return self.trained_models

    def save(self, name: str, model, suffix: str = "") -> Path:
        """Persists a trained model to disk via joblib."""
        filename = f"{name}{suffix}.pkl"
        path = self.models_dir / filename
        joblib.dump(model, path)
        logger.info("Saved model '%s' to %s", name, path)
        return path