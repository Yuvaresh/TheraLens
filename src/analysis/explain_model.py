#!/usr/bin/env python3
"""
TheraLens: Model Explainability & Interpretability Analysis

Provides interpretable explanations for:
1. Symptom Disease Classifier: Model-agnostic permutation feature importance on held-out test data.
2. Therapeutic Candidate Prioritization: Explicit mathematical score contribution breakdown.

SCIENTIFIC PRINCIPLE:
Feature importance quantifies statistical reliance in the model's decision function,
NOT biological causality or disease etiology.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "symptom_disease_clean.csv"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "processed" / "therapeutic_candidate_rankings.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "explainability"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Styling for academic charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ExplainModel")


def explain_symptom_classifier(random_seed: int = 42) -> pd.DataFrame:
    """Computes permutation feature importance on held-out test data."""
    model_path = MODELS_DIR / "symptom_classifier.joblib"
    meta_path = MODELS_DIR / "symptom_classifier_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
        
    logger.info("Loading model and metadata for permutation importance analysis...")
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    df_clean = pd.read_csv(DATA_CLEAN_PATH)
    target_col = "prognosis" if "prognosis" in df_clean.columns else df_clean.columns[-1]
    feature_cols = [c for c in df_clean.columns if c != target_col]
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    # Exact stratified split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=meta["random_seed"], stratify=y
    )
    
    logger.info("Computing permutation importance on held-out test set (N=%d samples, 20 repeats)...", len(y_test))
    perm_res = permutation_importance(
        model, X_test, y_test, n_repeats=20, random_state=random_seed, scoring="accuracy"
    )
    
    importances_df = pd.DataFrame({
        "symptom_feature": feature_cols,
        "mean_importance": perm_res.importances_mean,
        "std_importance": perm_res.importances_std,
    }).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)
    
    out_csv = OUTPUTS_DIR / "symptom_permutation_importance.csv"
    importances_df.to_csv(out_csv, index=False)
    logger.info("Saved permutation feature importance table to %s", out_csv)
    
    # Plot top 20 features
    top20 = importances_df.head(20).copy()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        top20["symptom_feature"][::-1],
        top20["mean_importance"][::-1],
        xerr=top20["std_importance"][::-1],
        color="#2b83ba",
        ecolor="#d7191c",
        capsize=3,
        alpha=0.85,
    )
    ax.set_title("Top 20 Influential Clinical Symptoms (Permutation Feature Importance on Held-Out Test Set)", fontsize=13, weight="bold")
    ax.set_xlabel("Mean Accuracy Degradation Upon Feature Permutation", fontsize=11)
    ax.set_ylabel("Clinical Symptom Feature", fontsize=11)
    
    # Add methodology annotation
    ax.text(
        0.98, 0.05,
        "Note: Feature importance indicates model statistical reliance,\nnot medical causality or disease etiology.",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "top_symptoms_importance.png", dpi=300)
    plt.close(fig)
    logger.info("Saved top symptoms importance chart to %s", OUTPUTS_DIR / "top_symptoms_importance.png")
    
    return importances_df


def explain_candidate_prioritization() -> pd.DataFrame:
    """Analyzes and plots exact mathematical contribution of scoring components for candidates."""
    if not CANDIDATES_PATH.exists():
        raise FileNotFoundError(f"Candidate rankings not found at {CANDIDATES_PATH}")
        
    logger.info("Decomposing therapeutic prioritization score components...")
    df = pd.read_csv(CANDIDATES_PATH)
    
    # Top 15 candidates across distinct diseases
    top_candidates = df.groupby("local_disease", as_index=False, group_keys=False).head(1).head(15).copy()
    
    top_candidates["Target_Assoc_Contrib"] = (top_candidates["score_target_assoc"] * 0.35).round(4)
    top_candidates["Potency_Contrib"] = (top_candidates["score_potency"] * 0.35).round(4)
    top_candidates["Druglikeness_Contrib"] = (top_candidates["score_druglike"] * 0.15).round(4)
    top_candidates["Clinical_Evidence_Contrib"] = (top_candidates["score_clinical_evidence"] * 0.15).round(4)
    
    contrib_cols = [
        "local_disease",
        "pref_name",
        "molecule_chembl_id",
        "target_symbol",
        "composite_priority_score",
        "Target_Assoc_Contrib",
        "Potency_Contrib",
        "Druglikeness_Contrib",
        "Clinical_Evidence_Contrib",
    ]
    decomp_df = top_candidates[contrib_cols].copy()
    
    out_csv = OUTPUTS_DIR / "candidate_score_decomposition.csv"
    decomp_df.to_csv(out_csv, index=False)
    logger.info("Saved candidate score decomposition table to %s", out_csv)
    
    # Stacked bar plot
    plot_df = decomp_df.copy()
    plot_df["label"] = plot_df["local_disease"] + " (" + plot_df["target_symbol"] + ")"
    plot_df = plot_df.set_index("label")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_df[
        ["Target_Assoc_Contrib", "Potency_Contrib", "Druglikeness_Contrib", "Clinical_Evidence_Contrib"]
    ].plot(kind="barh", stacked=True, colormap="viridis", ax=ax)
    
    ax.set_title("Decomposition of Candidate Prioritization Scores (CPS Components)", fontsize=13, weight="bold")
    ax.set_xlabel("Contribution to Composite Priority Score (CPS, 0.0 - 1.0)", fontsize=11)
    ax.set_ylabel("Disease Context (Biological Target)", fontsize=11)
    ax.set_xlim(0, 1.0)
    plt.legend(
        ["Target Association (35%)", "pChEMBL Potency (35%)", "Drug-Likeness (15%)", "Clinical Evidence (15%)"],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="Scoring Component",
    )
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "candidate_contribution_breakdown.png", dpi=300)
    plt.close(fig)
    logger.info("Saved candidate contribution breakdown chart to %s", OUTPUTS_DIR / "candidate_contribution_breakdown.png")
    
    return decomp_df


def run_explainability_pipeline():
    """Runs complete model explainability pipeline."""
    imp_df = explain_symptom_classifier()
    cand_df = explain_candidate_prioritization()
    
    summary = {
        "status": "SUCCESS",
        "top_5_influential_symptoms": imp_df.head(5)[["symptom_feature", "mean_importance"]].to_dict(orient="records"),
        "top_candidates_decomposed": len(cand_df),
        "causality_disclaimer": "Feature importance measures predictive model reliance, NOT medical causality.",
    }
    with open(OUTPUTS_DIR / "explainability_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved explainability summary JSON to %s", OUTPUTS_DIR / "explainability_summary.json")
    return summary


if __name__ == "__main__":
    res = run_explainability_pipeline()
    print("\nModel Explainability Pipeline Succeeded!")
    print("Top Influential Symptoms:")
    for s in res["top_5_influential_symptoms"]:
        print(f"  - {s['symptom_feature']}: {s['mean_importance']:.4f}")
