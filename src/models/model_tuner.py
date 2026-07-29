import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.config import CV_FOLDS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


RMSE_SCORER = make_scorer(_rmse, greater_is_better=False)


class HyperparameterTuner:
    """Wraps GridSearchCV for a single estimator + parameter grid."""

    def __init__(self, estimator, param_grid: dict, cv: int = CV_FOLDS):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.grid_search_: GridSearchCV = None

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Runs the grid search and returns the best fitted estimator."""
        logger.info("Starting GridSearchCV over %s", self.param_grid)
        self.grid_search_ = GridSearchCV(
            self.estimator,
            param_grid=self.param_grid,
            scoring=RMSE_SCORER,
            cv=self.cv,
            n_jobs=-1,
        )
        self.grid_search_.fit(X_train, y_train)
        logger.info("Best params: %s", self.grid_search_.best_params_)
        logger.info("Best CV RMSE: %.3f", -self.grid_search_.best_score_)
        return self.grid_search_.best_estimator_