import numpy as np
import pandas as pd
import pytest

from src.data.data_cleaner import DataCleaner


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "country": ["USA", "usa", None, "India", "USA"],
        "primary_fuel": ["Gas", "Gas", "Solar", None, "Gas"],
        "capacity_mw": [100.0, 100.0, np.nan, 200.0, 50.0],
        "estimated_generation_gwh_2017": [300.0, 300.0, 150.0, np.nan, 90.0],
    })


def test_clean_drops_rows_with_missing_target(sample_df):
    cleaner = DataCleaner(target_column="estimated_generation_gwh_2017")
    cleaned = cleaner.clean(sample_df)
    assert cleaned["estimated_generation_gwh_2017"].isnull().sum() == 0


def test_clean_removes_duplicates(sample_df):
    cleaner = DataCleaner(target_column="estimated_generation_gwh_2017")
    cleaned = cleaner.clean(sample_df)
    assert cleaned.duplicated().sum() == 0


def test_clean_imputes_missing_numeric_values(sample_df):
    cleaner = DataCleaner(target_column="estimated_generation_gwh_2017")
    cleaned = cleaner.clean(sample_df)
    assert cleaned["capacity_mw"].isnull().sum() == 0


def test_clean_fills_missing_categoricals(sample_df):
    cleaner = DataCleaner(target_column="estimated_generation_gwh_2017")
    cleaned = cleaner.clean(sample_df)
    assert cleaned["primary_fuel"].isnull().sum() == 0


def test_clean_populates_cleaning_report(sample_df):
    cleaner = DataCleaner(target_column="estimated_generation_gwh_2017")
    cleaner.clean(sample_df)
    assert "rows_before" in cleaner.cleaning_report
    assert "duplicates_removed" in cleaner.cleaning_report


def test_encode_categoricals_returns_numeric_columns(sample_df):
    cleaner = DataCleaner(target_column="estimated_generation_gwh_2017")
    cleaned = cleaner.clean(sample_df)
    encoded = cleaner.encode_categoricals(cleaned, ["country", "primary_fuel"])
    assert pd.api.types.is_numeric_dtype(encoded["country"])
    assert pd.api.types.is_numeric_dtype(encoded["primary_fuel"])