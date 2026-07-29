from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import MODEL_COMPARISON_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelComparator:
    """Compares trained models' metrics and saves comparison artifacts."""

    def __init__(self, output_dir: Path = MODEL_COMPARISON_DIR):
        self.output_dir = output_dir

    def compare(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Saves the metrics table and a bar chart per metric."""
        metrics_path = self.output_dir / "model_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info("Saved comparison metrics to %s", metrics_path)

        for metric in ["RMSE", "MAE", "R2_Score"]:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="Model", y=metric, data=metrics_df, ax=ax)
            ax.set_title(f"Model Comparison - {metric}")
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
            path = self.output_dir / f"comparison_{metric.lower()}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)

        best_row = metrics_df.loc[metrics_df["RMSE"].idxmin()]
        logger.info("Best model by RMSE: %s (RMSE=%.3f)", best_row["Model"], best_row["RMSE"])
        return metrics_df