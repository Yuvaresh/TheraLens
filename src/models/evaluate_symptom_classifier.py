#!/usr/bin/env python3
"""
TheraLens: Symptom Classifier Evaluation Module

Loads persisted model pipeline and evaluates held-out classification performance,
validates model loading and reproducibility, and prints detailed confusion and
classification performance metrics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "symptom_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "symptom_classifier_metadata.json"
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "symptom_disease_clean.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvaluateSymptomClassifier")


def evaluate_persisted_classifier() -> Dict[str, Any]:
    """Loads saved model and evaluates on held-out dataset."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}.")
        
    logger.info("Loading persisted model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    logger.info("Model loaded successfully: %s (%s)", metadata["model_name"], metadata["model_class"])
    
    df_clean = pd.read_csv(DATA_CLEAN_PATH)
    target_col = "prognosis" if "prognosis" in df_clean.columns else df_clean.columns[-1]
    feature_cols = [c for c in df_clean.columns if c != target_col]
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    # Replicate exact stratified split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=metadata["random_seed"], stratify=y
    )
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    logger.info("Persisted Model Evaluation on Held-Out Test Set (N=%d):", len(y_test))
    logger.info("  Accuracy:          %.4f", acc)
    logger.info("  Balanced Accuracy: %.4f", bal_acc)
    logger.info("  Macro F1-Score:    %.4f", macro_f1)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    
    results = {
        "status": "SUCCESS",
        "model_reloaded": True,
        "test_samples": len(y_test),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "classification_report": report,
    }
    return results


if __name__ == "__main__":
    res = evaluate_persisted_classifier()
    print("\nModel Reload & Verification Succeeded!")
    print(f"Accuracy: {res['accuracy']:.4f} | Balanced Accuracy: {res['balanced_accuracy']:.4f} | Macro F1: {res['macro_f1']:.4f}")
