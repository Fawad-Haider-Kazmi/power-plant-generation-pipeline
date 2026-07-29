from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_TRAINING_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates regression models and records their metrics."""

    def __init__(self, output_dir: Path = MODEL_TRAINING_DIR):
        self.output_dir = output_dir
        self.results: list = []

    def evaluate(self, name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Scores `model` on the test set and stores/returns the metrics."""
        predictions = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        metrics = {"Model": name, "RMSE": rmse, "MAE": mae, "R2_Score": r2}
        self.results.append(metrics)
        logger.info("%s -> RMSE: %.3f | MAE: %.3f | R2: %.3f", name, rmse, mae, r2)

        self._plot_error_distribution(name, y_test, predictions)
        return metrics

    def _plot_error_distribution(self, name: str, y_test: pd.Series, predictions) -> Path:
        errors = y_test.values - predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(errors, bins=40, kde=True, ax=ax)
        ax.set_title(f"{name}: Prediction Error Distribution")
        ax.set_xlabel("Actual - Predicted")
        path = self.output_dir / f"{name}_error_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def results_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)