from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_DIR, MISSING_VALUE_DIR, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EDAAnalyzer:
    """Generates and persists exploratory data analysis visualizations."""

    def __init__(
        self,
        eda_dir: Path = EDA_DIR,
        missing_value_dir: Path = MISSING_VALUE_DIR,
        target_column: str = TARGET_COLUMN,
    ):
        self.eda_dir = eda_dir
        self.missing_value_dir = missing_value_dir
        self.target_column = target_column

    def plot_missing_values(self, df: pd.DataFrame, stage_label: str) -> Path:
        fig, ax = plt.subplots(figsize=(14, 5))
        df.isnull().sum().plot(kind="bar", ax=ax)
        ax.set_title(f"Missing Values per Column ({stage_label})")
        path = self.missing_value_dir / f"missing_values_{stage_label.lower()}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_target_distribution(self, df: pd.DataFrame) -> Path:
        fig, ax = plt.subplots(figsize=(8, 5))
        df[self.target_column].hist(bins=50, ax=ax)
        ax.set_title("Distribution of Estimated Generation (GWh)")
        ax.set_xlabel(self.target_column)
        path = self.eda_dir / "target_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_count_by_category(self, df: pd.DataFrame, column: str, top_n: int = 15) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        order = df[column].value_counts().head(top_n).index
        sns.countplot(x=column, data=df, order=order, ax=ax)
        ax.set_title(f"Count of Plants by {column}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        path = self.eda_dir / f"count_by_{column}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_target_by_category(self, df: pd.DataFrame, column: str) -> Path:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=column, y=self.target_column, data=df, ax=ax)
        ax.set_title(f"{self.target_column} by {column}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        path = self.eda_dir / f"generation_by_{column}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_scatter(self, df: pd.DataFrame, x_column: str, hue_column: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=x_column, y=self.target_column, hue=hue_column, data=df, alpha=0.6, ax=ax)
        ax.set_title(f"{x_column} vs {self.target_column}")
        path = self.eda_dir / f"{x_column}_vs_target.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def run_full_eda(self, df: pd.DataFrame) -> None:
        """Runs the standard EDA suite used for this dataset."""
        logger.info("Running full EDA suite")
        self.plot_target_distribution(df)
        if "primary_fuel" in df.columns:
            self.plot_count_by_category(df, "primary_fuel")
            self.plot_target_by_category(df, "primary_fuel")
        if "country" in df.columns:
            self.plot_count_by_category(df, "country")
        if "capacity_mw" in df.columns and "primary_fuel" in df.columns:
            self.plot_scatter(df, "capacity_mw", "primary_fuel")