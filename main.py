import json

from sklearn.model_selection import train_test_split

from src.config import (
    COLUMNS_TO_ENCODE,
    GRADIENT_BOOSTING_PARAM_GRID,
    MODEL_READY_DATA_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_FOREST_PARAM_GRID,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataLoader
from src.eda.eda_analyzer import EDAAnalyzer
from src.features.feature_engineer import FeatureEngineer
from src.features.feature_selector import FeatureSelector
from src.models.model_comparator import ModelComparator
from src.models.model_evaluator import ModelEvaluator
from src.models.model_trainer import ModelTrainer
from src.models.model_tuner import HyperparameterTuner
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline() -> None:
    documentation = {}

    #  Data Loading 
    loader = DataLoader()
    raw_df = loader.load()
    documentation["dataset_understanding"] = {
        "n_rows": raw_df.shape[0],
        "n_columns": raw_df.shape[1],
    }

    eda = EDAAnalyzer()
    eda.plot_missing_values(raw_df, stage_label="Before")

    #  Data Cleaning + Preprocessing 
    cleaner = DataCleaner()
    clean_df = cleaner.clean(raw_df)
    eda.plot_missing_values(clean_df, stage_label="After")
    documentation["cleaning_report"] = cleaner.cleaning_report
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    #  EDA 
    eda.run_full_eda(clean_df)

    #  Feature Engineering 
    engineer = FeatureEngineer()
    engineered_df = engineer.transform(clean_df)

    #  Preprocessing: encode categoricals 
    encoded_df = cleaner.encode_categoricals(engineered_df, COLUMNS_TO_ENCODE)

    #  Feature Selection 
    selector = FeatureSelector()
    selected_features = selector.select(encoded_df)
    model_ready_df = encoded_df[selected_features + [TARGET_COLUMN]]
    model_ready_df.to_csv(MODEL_READY_DATA_PATH, index=False)
    documentation["selected_features"] = selected_features

    #  Data Splitting 
    X = model_ready_df.drop(columns=[TARGET_COLUMN])
    y = model_ready_df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    documentation["train_test_split"] = {
        "train_rows": len(X_train), "test_rows": len(X_test),
    }

    #  Model Selection + Training + Saving 
    trainer = ModelTrainer()
    trained_models = trainer.train_all(X_train, y_train)
    for name, model in trained_models.items():
        trainer.save(name, model)

    #  Model Evaluation 
    evaluator = ModelEvaluator()
    for name, model in trained_models.items():
        evaluator.evaluate(name, model, X_test, y_test)

    #  Hyperparameter Tuning 
    rf_tuner = HyperparameterTuner(trainer.models["random_forest"], RANDOM_FOREST_PARAM_GRID)
    best_rf = rf_tuner.tune(X_train, y_train)
    trainer.save("random_forest", best_rf, suffix="_tuned")
    evaluator.evaluate("random_forest_tuned", best_rf, X_test, y_test)

    gb_tuner = HyperparameterTuner(trainer.models["gradient_boosting"], GRADIENT_BOOSTING_PARAM_GRID)
    best_gb = gb_tuner.tune(X_train, y_train)
    trainer.save("gradient_boosting", best_gb, suffix="_tuned")
    evaluator.evaluate("gradient_boosting_tuned", best_gb, X_test, y_test)

    #  Model Comparison 
    comparator = ModelComparator()
    metrics_df = evaluator.results_as_dataframe()
    comparator.compare(metrics_df)
    documentation["final_metrics"] = metrics_df.to_dict(orient="records")

    #  Documentation of Results 
    results_path = RESULTS_DIR / "pipeline_summary.json"
    with open(results_path, "w") as f:
        json.dump(documentation, f, indent=2, default=str)
    logger.info("Pipeline complete. Summary written to %s", results_path)


if __name__ == "__main__":
    run_pipeline()