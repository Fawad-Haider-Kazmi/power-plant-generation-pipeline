from pathlib import Path

# Project path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "global_power_plant_database.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed_power_plant.csv"
MODEL_READY_DATA_PATH = PROJECT_ROOT / "data" / "model_ready_power_plant.csv"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "EDA"
MISSING_VALUE_DIR = OUTPUTS_DIR / "MissingValueGraphs"
FEATURE_SELECTION_DIR = OUTPUTS_DIR / "FeatureSelection"
MODEL_TRAINING_DIR = OUTPUTS_DIR / "ModelTraining"
MODEL_COMPARISON_DIR = OUTPUTS_DIR / "ModelComparison"
RESULTS_DIR = OUTPUTS_DIR / "Results"

for _directory in (
    MODELS_DIR, EDA_DIR, MISSING_VALUE_DIR,
    FEATURE_SELECTION_DIR, MODEL_TRAINING_DIR, MODEL_COMPARISON_DIR, RESULTS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

TARGET_COLUMN = "estimated_generation_gwh_2017"
CURRENT_YEAR = 2024

CATEGORICAL_COLUMNS_TO_CLEAN = ["country", "primary_fuel", "owner", "source"]
COLUMNS_TO_ENCODE = ["country", "primary_fuel", "source", "owner", "geolocation_source"]
COLUMNS_EXCLUDED_FROM_FEATURES = {
    TARGET_COLUMN, "name", "gppd_idnr", "commissioning_year", "wepp_id", "url",
}


# Feature selection

CORRELATION_THRESHOLD = 0.02
FALLBACK_TOP_N_FEATURES = 5


# Train / test split

TEST_SIZE = 0.2
CV_FOLDS = 5


# Hyperparameter grids

RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [100, 150],
    "max_depth": [8, 12, 15],
    "min_samples_leaf": [3, 5, 10],
    "min_samples_split": [10, 20],
}
GRADIENT_BOOSTING_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
}

LOGGING_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"