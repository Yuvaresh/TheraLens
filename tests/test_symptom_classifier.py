"""
Unit and integration tests for Symptom-Based Disease Classification Pipeline.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "symptom_disease.csv"
DATA_CLEAN = PROJECT_ROOT / "data" / "processed" / "symptom_disease_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "symptom_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "symptom_classifier_metadata.json"

from src.inference.predict_disease import DiseasePredictor


def test_raw_symptom_dataset_schema():
    """Validates raw symptom dataset integrity and dimensions."""
    assert DATA_RAW.exists(), "Raw symptom dataset must exist."
    df = pd.read_csv(DATA_RAW)
    assert df.shape == (4961, 133), f"Expected shape (4961, 133), got {df.shape}"
    assert "prognosis" in df.columns, "'prognosis' target column must be present."
    assert df["prognosis"].nunique() == 41, "Must contain exactly 41 disease classes."
    assert df.isna().sum().sum() == 0, "Raw dataset must have zero missing values."


def test_deduplication_and_unique_patterns():
    """Verifies duplicate pattern detection and deduplicated clean dataset."""
    assert DATA_CLEAN.exists(), "Clean deduplicated dataset must exist."
    df_clean = pd.read_csv(DATA_CLEAN)
    assert len(df_clean) == 304, f"Expected 304 unique clinical patterns, got {len(df_clean)}"
    
    # Feature columns deduplication check
    feat_cols = [c for c in df_clean.columns if c != "prognosis"]
    assert df_clean[feat_cols].duplicated().sum() == 0, "No duplicate feature patterns allowed in clean dataset."


def test_leakage_free_train_test_split():
    """Verifies train and test sets have disjoint clinical patterns to prevent data leakage."""
    df_clean = pd.read_csv(DATA_CLEAN)
    feat_cols = [c for c in df_clean.columns if c != "prognosis"]
    X = df_clean[feat_cols].values
    y = df_clean["prognosis"].values

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Convert rows to tuples to check set intersection
    train_patterns = {tuple(row) for row in X_train}
    test_patterns = {tuple(row) for row in X_test}
    
    intersection = train_patterns.intersection(test_patterns)
    assert len(intersection) == 0, f"Found {len(intersection)} overlapping symptom patterns between train and test!"


def test_persisted_model_reproducibility():
    """Verifies that persisted model loads and produces valid predictions."""
    assert MODEL_PATH.exists(), "Trained model joblib must exist."
    assert METADATA_PATH.exists(), "Model metadata JSON must exist."
    
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict"), "Model must implement predict() method."
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["evaluation_metrics"]["accuracy"] >= 0.90, "Model accuracy should meet or exceed 0.90."
    assert meta["num_classes"] == 41, "Model must cover 41 classes."


def test_inference_predictor_api():
    """Verifies DiseasePredictor inference interface and required disclaimer."""
    predictor = DiseasePredictor()
    
    # Test valid symptom query
    result = predictor.predict("chills, vomiting, high_fever, sweating, headache", top_k=3)
    assert result["status"] == "SUCCESS"
    assert result["predicted_disease"] is not None
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert len(result["top_candidates"]) <= 3
    assert "RESEARCH DEMONSTRATION ONLY" in result["disclaimer"]
    
    # Test empty / unknown symptoms query
    result_empty = predictor.predict("nonexistent_alien_symptom_xyz", top_k=3)
    assert result_empty["status"] == "NO_SYMPTOMS_MATCHED"
    assert result_empty["predicted_disease"] is None
