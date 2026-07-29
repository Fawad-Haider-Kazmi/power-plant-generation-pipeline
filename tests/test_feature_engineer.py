import pandas as pd
from src.features.feature_engineer import FeatureEngineer


def test_derives_plant_age_from_commissioning_year():
    df = pd.DataFrame({"commissioning_year": [2000, 2010], "capacity_mw": [50, 100]})
    result = FeatureEngineer(current_year=2024).transform(df)
    assert "plant_age_years" in result.columns
    assert result.loc[0, "plant_age_years"] == 24


def test_handles_zero_capacity_gracefully():
    df = pd.DataFrame({"commissioning_year": [2000, 2010], "capacity_mw": [0, 100]})
    result = FeatureEngineer(current_year=2024).transform(df)
    assert result["capacity_mw"].isnull().sum() == 0
    assert result.loc[0, "capacity_mw"] == 100