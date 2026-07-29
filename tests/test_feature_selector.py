import pandas as pd
from src.features.feature_selector import FeatureSelector


def test_select_falls_back_when_nothing_clears_threshold():
    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4],
        "feature_b": [4, 3, 2, 1],
        "estimated_generation_gwh_2017": [10, 10, 10, 10],  # constant target -> ~0 correlation
    })
    selector = FeatureSelector(target_column="estimated_generation_gwh_2017", threshold=0.9)
    selected = selector.select(df)
    assert len(selected) > 0


def test_select_excludes_identifier_columns():
    df = pd.DataFrame({
        "gppd_idnr": [1, 2, 3, 4],
        "capacity_mw": [10, 20, 30, 40],
        "estimated_generation_gwh_2017": [12, 22, 33, 39],
    })
    selector = FeatureSelector(target_column="estimated_generation_gwh_2017", threshold=0.01)
    selected = selector.select(df)
    assert "gppd_idnr" not in selected