# Power Plant Generation Prediction Pipeline

An end-to-end, object-oriented machine learning pipeline that predicts
power plant electricity generation (in GWh) from plant characteristics
and location data, using the World Resources Institute's **Global Power
Plant Database**.

This project was built as part of the Research Internship Program 2026
— Batch 2, Task 1: implementing a complete, production-oriented data
science pipeline following professional software engineering
standards.

---

## Project Overview

The pipeline takes raw, messy real-world power plant data and produces
a trained regression model that estimates a plant's annual electricity
generation from features such as capacity, primary fuel type, country,
geographic location, and plant age. It covers every stage of a
standard data science workflow — data loading, understanding,
cleaning, preprocessing, exploratory data analysis, feature
engineering, feature selection, model training, hyperparameter tuning,
evaluation, comparison, saving, and inference — implemented as
reusable Python classes rather than a single procedural script.

**Problem type:** Regression
**Target variable:** `estimated_generation_gwh_2017` (estimated annual
electricity generation, in gigawatt-hours)

---

## Dataset Information

- **Source:** [World Resources Institute – Global Power Plant Database](https://datasets.wri.org/dataset/globalpowerplantdatabase)
- **Format:** CSV, ~35,000 power plant records worldwide
- **Key columns used:**
  - `capacity_mw` — installed generating capacity
  - `primary_fuel` — main fuel/energy source (Coal, Gas, Hydro, Solar, Wind, etc.)
  - `country`, `latitude`, `longitude` — location
  - `commissioning_year` — year the plant came online (used to derive plant age)
  - `owner`, `source`, `geolocation_source` — metadata fields
  - `estimated_generation_gwh_2017` — the regression target
- **Known data quality issues handled by the pipeline:** missing
  values in several numeric and categorical columns, duplicate rows,
  and inconsistent text formatting (casing/whitespace) in category
  labels.

The raw CSV is included at `data/global_power_plant_database.csv`. If
it isn't present, download it from the WRI link above and place it at
that exact path before running the pipeline.

---

## Project Structure

power-plant-generation-pipeline/
├── .github/workflows/ci.yml # GitHub Actions CI - runs tests on every push
├── .gitignore
├── requirements.txt
├── README.md
├── main.py # Orchestrates the full pipeline end-to-end
│
├── data/
│ ├── global_power_plant_database.csv # raw source data (input)
│ ├── processed_power_plant.csv # generated: after cleaning
│ └── model_ready_power_plant.csv # generated: after feature selection
│
├── models/ # generated: trained model files (.pkl)
│
├── outputs/
│ ├── EDA/ # generated: exploratory analysis charts
│ ├── MissingValueGraphs/ # generated: missing-data visualizations
│ ├── FeatureSelection/ # generated: correlation heatmap + scores
│ ├── ModelTraining/ # generated: prediction error distributions
│ ├── ModelComparison/ # generated: metrics table + comparison charts
│ └── Results/ # generated: pipeline_summary.json
│
├── src/
│ ├── config.py # paths, constants, hyperparameter grids
│ ├── data/
│ │ ├── data_loader.py # DataLoader - loading & understanding
│ │ └── data_cleaner.py # DataCleaner - cleaning & preprocessing
│ ├── features/
│ │ ├── feature_engineer.py # FeatureEngineer - derives plant_age_years etc.
│ │ └── feature_selector.py # FeatureSelector - correlation-based selection
│ ├── eda/
│ │ └── eda_analyzer.py # EDAAnalyzer - generates all EDA plots
│ ├── models/
│ │ ├── model_trainer.py # ModelTrainer - trains RF & Gradient Boosting
│ │ ├── model_evaluator.py # ModelEvaluator - RMSE / MAE / R²
│ │ ├── model_tuner.py # HyperparameterTuner - GridSearchCV wrapper
│ │ └── model_comparator.py # ModelComparator - side-by-side comparison
│ ├── inference/
│ │ └── predictor.py # Predictor - loads a saved model, scores new data
│ └── utils/
│ └── logger.py # shared logging factory
│
└── tests/
├── test_data_cleaner.py
├── test_feature_engineer.py
└── test_feature_selector.py

---

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/Fawad-Haider-Kazmi/power-plant-generation-pipeline.git
cd power-plant-generation-pipeline
```

### 2. Virtual Environment Setup

A dedicated virtual environment keeps this project's dependencies
isolated from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll know it worked when your terminal prompt shows `(venv)` at the
start of the line. To deactivate later, just run `deactivate`.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Models, train/test split, GridSearchCV, metrics |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `joblib` | Saving/loading trained models |
| `pytest` | Automated unit testing |

All pinned in `requirements.txt`.

---

## How to Run the Project

Make sure `data/global_power_plant_database.csv` exists, then simply run:

```bash
python main.py
```

This single command runs the entire pipeline in order:

1. Loads and profiles the raw dataset
2. Cleans it (missing values, duplicates, inconsistent labels, outlier flags)
3. Runs exploratory data analysis and saves all charts
4. Engineers new features (e.g. `plant_age_years`)
5. Encodes categorical columns and selects the most predictive features
6. Splits the data into train/test sets
7. Trains a Random Forest and a Gradient Boosting Regressor
8. Evaluates both, then tunes both with `GridSearchCV`
9. Compares all model variants and saves the results
10. Saves every trained model and writes a `pipeline_summary.json`

Two intermediate CSVs (`processed_power_plant.csv`,
`model_ready_power_plant.csv`) and all trained models/plots are
generated automatically — you do not need to create or supply them
yourself.

### Running the tests

```bash
pytest tests/ -v
```

### Using a trained model for a new prediction

```python
from src.inference.predictor import Predictor

predictor = Predictor.from_model_name("random_forest_tuned")
predictions = predictor.predict(new_feature_dataframe)
```

### Continuous Integration

Every push to `main` automatically runs the full test suite via
GitHub Actions (see `.github/workflows/ci.yml`). Check the **Actions**
tab on GitHub to see the latest run status.

---

## Results

*(Fill in from `outputs/ModelComparison/model_metrics.csv` after running `python main.py`)*

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Random Forest | [value] | [value] | [value] |
| Gradient Boosting | [value] | [value] | [value] |
| Random Forest (Tuned) | [value] | [value] | [value] |
| Gradient Boosting (Tuned) | [value] | [value] | [value] |

**Best model:** [model name] — selected by lowest RMSE.

Key visualizations generated by the pipeline (available in `outputs/`):
- Missing-value comparison before/after cleaning
- Target variable (generation GWh) distribution
- Generation by primary fuel type
- Capacity vs. generation scatter plot
- Correlation heatmap of encoded features
- Prediction error distributions per model
- Side-by-side model comparison charts (RMSE, MAE, R²)

Full run metadata (row counts, cleaning report, selected features,
final metrics) is saved to `outputs/Results/pipeline_summary.json`.

---

## Author

Syed Fawad Haider Kazmi
GitHub: [Fawad-Haider-Kazmi](https://github.com/Fawad-Haider-Kazmi)



## Project Structure
