#!/usr/bin/env python3
"""
TheraLens: Supervised Symptom-Based Disease Classification Pipeline

Trains and evaluates multiclass classification models on clinical symptom patterns.
Crucially accounts for data duplication: evaluates on deduplicated, unique clinical
patterns using stratified train/test splitting to prevent artificial performance
inflation from synthetic row duplication.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC

# Setup Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "symptom_disease.csv"
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "symptom_disease_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "model_evaluation"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

# Styling for academic charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainSymptomClassifier")


def load_and_audit_dataset(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """Loads raw dataset, audits duplication, and creates deduplicated feature matrix."""
    logger.info("Loading symptom dataset from %s", filepath)
    df_raw = pd.read_csv(filepath)
    
    target_col = "prognosis" if "prognosis" in df_raw.columns else df_raw.columns[-1]
    feature_cols = [c for c in df_raw.columns if c != target_col]
    
    raw_rows, raw_cols = df_raw.shape
    raw_dupes = df_raw.duplicated().sum()
    unique_patterns = df_raw[feature_cols].drop_duplicates().shape[0]
    
    logger.info("Raw Dataset: %d rows, %d columns (%d symptom features)", raw_rows, raw_cols, len(feature_cols))
    logger.info("Target: '%s' with %d unique disease classes", target_col, df_raw[target_col].nunique())
    logger.info("Duplication audit: %d rows (%.2f%%) are exact duplicates", raw_dupes, (raw_dupes / raw_rows) * 100)
    logger.info("Unique clinical symptom patterns: %d", unique_patterns)
    
    # Save cleaned, deduplicated dataset
    df_clean = df_raw.drop_duplicates(subset=feature_cols).copy().reset_index(drop=True)
    df_clean.to_csv(DATA_CLEAN_PATH, index=False)
    logger.info("Saved deduplicated dataset (%d rows) to %s", len(df_clean), DATA_CLEAN_PATH)
    
    return df_raw, df_clean, feature_cols, target_col


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
    """Computes comprehensive multiclass evaluation metrics."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
        "Macro_Precision": round(prec_macro, 4),
        "Macro_Recall": round(rec_macro, 4),
        "Macro_F1": round(f1_macro, 4),
        "Weighted_F1": round(f1_weighted, 4),
        "y_pred": y_pred,
    }


def plot_class_distribution(df_clean: pd.DataFrame, target_col: str, output_path: Path):
    """Generates bar chart of unique pattern frequency per disease class."""
    counts = df_clean[target_col].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Unique Clinical Symptom Patterns per Disease Class (Deduplicated Manifold)", fontsize=13, weight="bold")
    ax.set_xlabel("Number of Unique Symptom Presentations", fontsize=11)
    ax.set_ylabel("Disease Class (Prognosis)", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved class distribution chart to %s", output_path)


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path):
    """Generates bar chart comparing models across key metrics."""
    metrics_to_plot = ["Accuracy", "Balanced_Accuracy", "Macro_F1", "Macro_Precision", "Macro_Recall"]
    melted = comparison_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette="tab10", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-Out Benchmark Performance Across Evaluated Models", fontsize=13, weight="bold")
    ax.set_ylabel("Metric Score (0.0 - 1.0)", fontsize=11)
    ax.set_xlabel("Evaluation Metric", fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved model comparison chart to %s", output_path)


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: Path):
    """Plots confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title("Confusion Matrix on Held-Out Test Set (Best Model)", fontsize=14, weight="bold")
    ax.set_xlabel("Predicted Disease", fontsize=11)
    ax.set_ylabel("True Disease", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved confusion matrix heatmap to %s", output_path)


def plot_per_class_f1(report_dict: Dict[str, Any], output_path: Path):
    """Plots per-class F1-score for held-out test predictions."""
    class_f1 = {}
    for k, v in report_dict.items():
        if isinstance(v, dict) and "f1-score" in v and k not in ["accuracy", "macro avg", "weighted avg"]:
            class_f1[k] = v["f1-score"]
            
    s = pd.Series(class_f1).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x=s.values, y=s.index, hue=s.index, palette="mako", legend=False, ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-Class Held-Out F1-Score (Best Model)", fontsize=13, weight="bold")
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_ylabel("Disease Class", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info("Saved per-class F1 chart to %s", output_path)


def train_and_benchmark_pipeline(random_seed: int = 42) -> Dict[str, Any]:
    """Executes the complete training, benchmarking, and artifact persistence pipeline."""
    df_raw, df_clean, feature_cols, target_col = load_and_audit_dataset(DATA_RAW_PATH)
    
    # Save class distribution chart
    plot_class_distribution(df_clean, target_col, OUTPUTS_DIR / "disease_class_distribution.png")
    
    # Features and Target
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    classes = sorted(list(np.unique(y)))
    
    # Stratified Train/Test Split on Unique Patterns (Leakage-Resistant)
    # 20% test size ensures all 41 classes are represented in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_seed, stratify=y
    )
    logger.info("Leakage-resistant split: Train samples = %d, Test samples = %d across %d classes",
                len(X_train), len(X_test), len(classes))
    
    # Model candidates
    candidate_models = {
        "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=random_seed),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_seed),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=random_seed),
    }
    
    results = []
    fitted_models = {}
    
    for name, clf in candidate_models.items():
        logger.info("Training candidate model: %s...", name)
        clf.fit(X_train, y_train)
        fitted_models[name] = clf
        eval_metrics = evaluate_model(clf, X_test, y_test, name)
        results.append(eval_metrics)
        logger.info(
            "  %s -> Acc: %.4f | Bal Acc: %.4f | Macro F1: %.4f | Macro Rec: %.4f",
            name, eval_metrics["Accuracy"], eval_metrics["Balanced_Accuracy"],
            eval_metrics["Macro_F1"], eval_metrics["Macro_Recall"]
        )
        
    comparison_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "y_pred"} for r in results
    ]).sort_values(by="Macro_F1", ascending=False).reset_index(drop=True)
    
    comparison_csv_path = OUTPUTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    logger.info("Saved model comparison table to %s", comparison_csv_path)
    
    plot_model_comparison(comparison_df, OUTPUTS_DIR / "model_performance_comparison.png")
    
    # Best model selection by Macro F1
    best_name = comparison_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_result = [r for r in results if r["Model"] == best_name][0]
    best_pred = best_result["y_pred"]
    
    logger.info("Selected Best Model: '%s' (Macro F1 = %.4f)", best_name, best_result["Macro_F1"])
    
    # Classification Report & Confusion Matrix
    clf_report_dict = classification_report(y_test, best_pred, output_dict=True, zero_division=0)
    clf_report_text = classification_report(y_test, best_pred, zero_division=0)
    
    with open(OUTPUTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"TheraLens Symptom Classifier - Classification Report ({best_name})\n")
        f.write("=" * 70 + "\n\n")
        f.write(clf_report_text)
    logger.info("Saved detailed classification report text to %s", OUTPUTS_DIR / "classification_report.txt")
    
    cm = confusion_matrix(y_test, best_pred, labels=classes)
    np.savetxt(OUTPUTS_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, classes, OUTPUTS_DIR / "confusion_matrix.png")
    plot_per_class_f1(clf_report_dict, OUTPUTS_DIR / "per_class_f1_scores.png")
    
    # Save Production Artifacts
    model_save_path = MODELS_DIR / "symptom_classifier.joblib"
    joblib.dump(best_model, model_save_path)
    logger.info("Persisted best model pipeline to %s", model_save_path)
    
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
        
    with open(MODELS_DIR / "class_labels.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
        
    metadata = {
        "model_name": best_name,
        "model_class": best_model.__class__.__name__,
        "dataset_rows_raw": len(df_raw),
        "dataset_rows_clean": len(df_clean),
        "synthetic_duplicates_filtered": int(df_raw.shape[0] - df_clean.shape[0]),
        "unique_symptom_patterns": len(df_clean),
        "num_classes": len(classes),
        "num_features": len(feature_cols),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_seed": random_seed,
        "evaluation_metrics": {
            "accuracy": best_result["Accuracy"],
            "balanced_accuracy": best_result["Balanced_Accuracy"],
            "macro_precision": best_result["Macro_Precision"],
            "macro_recall": best_result["Macro_Recall"],
            "macro_f1": best_result["Macro_F1"],
            "weighted_f1": best_result["Weighted_F1"],
        },
        "all_model_benchmarks": comparison_df.to_dict(orient="records"),
    }
    
    with open(MODELS_DIR / "symptom_classifier_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Persisted model metadata to %s", MODELS_DIR / "symptom_classifier_metadata.json")
    
    return metadata


if __name__ == "__main__":
    meta = train_and_benchmark_pipeline()
    print("\nTraining & Benchmarking Pipeline Completed Successfully!")
    print(f"Best Model: {meta['model_name']}")
    print(f"Accuracy: {meta['evaluation_metrics']['accuracy']:.4f}")
    print(f"Macro F1: {meta['evaluation_metrics']['macro_f1']:.4f}")
