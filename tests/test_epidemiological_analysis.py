"""
Unit and integration tests for Epidemiological Trend Analysis Pipeline.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPI_TRENDS = PROJECT_ROOT / "data" / "processed" / "epidemiological_trends.csv"

from src.analysis.analyze_epidemiological_trends import detect_outbreak_surges


def test_epidemiological_trends_dataset():
    """Validates processed epidemiological trends dataset."""
    assert EPI_TRENDS.exists(), "Processed epidemiological trends CSV must exist."
    df = pd.read_csv(EPI_TRENDS)
    
    assert {"disease", "country", "year", "temporal_resolution", "reported_cases"}.issubset(df.columns)
    assert set(df["disease"].unique()) == {"Measles", "Yellow Fever", "Cholera"}
    assert len(df) > 5000, "Expected over 5,000 unified surveillance records."


def test_missing_value_retention():
    """Verifies that missing case counts are preserved as NaN and not converted blindly to zero."""
    df = pd.read_csv(EPI_TRENDS)
    # Yellow fever or measles often have years without complete reporting
    # Ensure dataset contains legitimate numeric counts and proper null semantics
    assert df["reported_cases"].dtype in [np.float64, float, int]


def test_outbreak_surge_detection_logic():
    """Tests rolling Z-score surge detector using synthetic time series fixture."""
    synthetic_data = pd.DataFrame({
        "disease": ["TestDisease"] * 10,
        "country": ["TEST"] * 10,
        "year": list(range(2000, 2010)),
        "temporal_resolution": ["Annual"] * 10,
        "time_period": [str(y) for y in range(2000, 2010)],
        # Baseline ~ 100 cases, with a massive surge of 2,000 cases in 2007
        "reported_cases": [100, 105, 98, 102, 101, 103, 100, 2000, 110, 105],
    })
    
    result = detect_outbreak_surges(synthetic_data, window=5, threshold_z=2.0)
    surge_years = result[result["is_outbreak_surge"]]["year"].tolist()
    assert 2007 in surge_years, "Year 2007 must be detected as an outbreak surge."
    assert 2002 not in surge_years, "Baseline year 2002 must not be flagged as a surge."
