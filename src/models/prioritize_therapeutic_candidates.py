#!/usr/bin/env python3
"""
TheraLens: Therapeutic Candidate Prioritization Engine

Integrates Open Targets biological disease-target associations, ChEMBL in-vitro
bioactivity (pChEMBL potency), and physicochemical molecular properties to
systematically rank therapeutic targets and small-molecule drug candidates.

SCIENTIFIC DISCLAIMER:
Rankings produced by this engine are exploratory computational prioritizations
based on published genomic associations and in-vitro binding assays. A high score
does NOT imply that a compound is safe, clinically approved, effective in humans,
or suitable for clinical deployment.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "therapeutic_prioritization"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
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
logger = logging.getLogger("PrioritizeTherapeutics")


def rank_disease_targets(assoc_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks biological drug targets within each disease by Open Targets association score."""
    logger.info("Ranking disease targets from %d total association records...", len(assoc_df))
    
    # Sort and rank within each disease
    ranked = assoc_df.sort_values(
        by=["local_disease", "association_score"], ascending=[True, False]
    ).copy()
    ranked["target_rank_within_disease"] = (
        ranked.groupby("local_disease")["association_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    
    out_path = DATA_PROCESSED_DIR / "disease_target_rankings.csv"
    ranked.to_csv(out_path, index=False)
    logger.info("Saved %d ranked disease targets to %s", len(ranked), out_path)
    return ranked


def compute_druglikeness_score(row: pd.Series) -> Tuple[float, bool]:
    """Calculates Lipinski Rule of 5 and Veber rule compliance score in [0.0, 1.0]."""
    mw = row.get("molecular_weight")
    alogp = row.get("alogp")
    psa = row.get("psa")
    hba = row.get("hba")
    hbd = row.get("hbd")
    
    if pd.isna(mw) and pd.isna(alogp) and pd.isna(psa):
        # Missing property data: assign neutral baseline score 0.5 with missing flag
        return 0.50, False

    passed_rules = 0
    total_applicable = 5
    
    if not pd.isna(mw) and mw <= 500.0:
        passed_rules += 1
    if not pd.isna(alogp) and -0.4 <= alogp <= 5.0:
        passed_rules += 1
    if not pd.isna(hba) and hba <= 10.0:
        passed_rules += 1
    if not pd.isna(hbd) and hbd <= 5.0:
        passed_rules += 1
    if not pd.isna(psa) and psa <= 140.0:
        passed_rules += 1
        
    score = passed_rules / total_applicable
    return round(score, 4), True


def compute_evidence_bonus(max_phase: Any) -> float:
    """Calculates clinical advancement bonus score from development phase [0.1, 1.0]."""
    if pd.isna(max_phase):
        return 0.10  # Preclinical / research compound baseline
    try:
        phase = float(max_phase)
        if phase >= 4.0:
            return 1.00  # Approved drug repurposing candidate
        elif phase >= 3.0:
            return 0.80  # Phase 3 candidate
        elif phase >= 2.0:
            return 0.60  # Phase 2 candidate
        elif phase >= 1.0:
            return 0.40  # Phase 1 candidate
        else:
            return 0.20  # Phase 0 / preclinical
    except (ValueError, TypeError):
        return 0.10


def prioritize_candidates(
    assoc_df: pd.DataFrame,
    act_df: pd.DataFrame,
    mol_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Merges datasets and calculates composite prioritization score (CPS)."""
    logger.info("Merging associations (%d), bioactivity (%d), and molecules (%d)...",
                len(assoc_df), len(act_df), len(mol_df))
    
    # 1. Target linkage: merge assoc and act on target_symbol
    merged_target = assoc_df.merge(
        act_df, on="target_symbol", how="inner"
    )
    
    # 2. Molecule linkage: merge with mol on molecule_chembl_id
    merged = merged_target.merge(
        mol_df, on="molecule_chembl_id", how="left"
    )
    logger.info("Total cross-referenced candidate records: %d", len(merged))
    
    # Audit unmatched records
    unmatched_targets = set(assoc_df["target_symbol"]) - set(act_df["target_symbol"])
    audit_summary = {
        "total_associations": len(assoc_df),
        "unique_association_targets": int(assoc_df["target_symbol"].nunique()),
        "unique_activity_targets": int(act_df["target_symbol"].nunique()),
        "matched_targets": int(merged["target_symbol"].nunique()),
        "unmatched_association_targets_count": len(unmatched_targets),
        "total_bioactivity_records": len(act_df),
        "total_molecules_referenced": int(merged["molecule_chembl_id"].nunique()),
        "scoring_weights": {
            "target_association_weight": 0.35,
            "potency_pchembl_weight": 0.35,
            "druglikeness_weight": 0.15,
            "clinical_evidence_weight": 0.15,
        },
    }
    
    # Aggregation per disease-target-molecule (deduplicate multiple bioactivity assays using median pchembl)
    grouped = (
        merged.groupby(
            [
                "local_disease",
                "disease_id",
                "open_targets_disease_name",
                "target_id",
                "target_symbol",
                "target_name",
                "association_score",
                "chembl_target_id",
                "molecule_chembl_id",
                "pref_name",
                "molecule_type",
                "max_phase",
                "molecular_weight",
                "alogp",
                "psa",
                "hba",
                "hbd",
                "rotatable_bond",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            median_pchembl=("pchembl_value", "median"),
            max_pchembl=("pchembl_value", "max"),
            assay_count=("activity_id", "count"),
        )
    )
    logger.info("Aggregated distinct (disease, target, molecule) candidate pairs: %d", len(grouped))
    
    # Component Scores
    # 1. Target association score (bounded [0, 1] by Open Targets)
    grouped["score_target_assoc"] = grouped["association_score"].clip(0.0, 1.0)
    
    # 2. Potency score from median pChEMBL: clip between 4.0 (100 uM) and 10.0 (0.1 nM)
    grouped["score_potency"] = (
        (grouped["median_pchembl"] - 4.0) / (10.0 - 4.0)
    ).clip(0.0, 1.0)
    
    # 3. Drug-likeness score and completeness flag
    druglike_res = grouped.apply(compute_druglikeness_score, axis=1)
    grouped["score_druglike"] = [r[0] for r in druglike_res]
    grouped["has_complete_properties"] = [r[1] for r in druglike_res]
    
    # 4. Clinical evidence bonus
    grouped["score_clinical_evidence"] = grouped["max_phase"].apply(compute_evidence_bonus)
    
    # Composite Prioritization Score (CPS)
    w_target = 0.35
    w_potency = 0.35
    w_druglike = 0.15
    w_evidence = 0.15
    
    grouped["composite_priority_score"] = (
        w_target * grouped["score_target_assoc"]
        + w_potency * grouped["score_potency"]
        + w_druglike * grouped["score_druglike"]
        + w_evidence * grouped["score_clinical_evidence"]
    ).round(4)
    
    # Rank candidates within each disease
    grouped = grouped.sort_values(
        by=["local_disease", "composite_priority_score"], ascending=[True, False]
    ).reset_index(drop=True)
    
    grouped["candidate_rank_within_disease"] = (
        grouped.groupby("local_disease")["composite_priority_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    
    out_csv = DATA_PROCESSED_DIR / "therapeutic_candidate_rankings.csv"
    grouped.to_csv(out_csv, index=False)
    logger.info("Saved %d ranked therapeutic candidates to %s", len(grouped), out_csv)
    
    with open(OUTPUTS_DIR / "prioritization_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    logger.info("Saved prioritization audit summary to %s", OUTPUTS_DIR / "prioritization_audit_summary.json")
    
    return grouped, audit_summary


def plot_prioritization_visualizations(candidates_df: pd.DataFrame, targets_df: pd.DataFrame):
    """Generates informative publication-grade diagnostic plots."""
    logger.info("Generating prioritization visualization charts...")
    
    # 1. Top candidates for key representative diseases
    rep_diseases = ["Malaria", "Diabetes", "Hypertension", "Dengue", "Tuberculosis"]
    sub_df = candidates_df[candidates_df["local_disease"].isin(rep_diseases)].copy()
    if sub_df.empty:
        sub_df = candidates_df.head(50).copy()
        
    top_per_disease = (
        sub_df.groupby("local_disease", as_index=False, group_keys=False)
        .head(3)
        .sort_values(by="composite_priority_score", ascending=True)
    )
    top_per_disease["label"] = (
        top_per_disease["local_disease"]
        + ": "
        + top_per_disease["pref_name"].fillna(top_per_disease["molecule_chembl_id"])
        + " ("
        + top_per_disease["target_symbol"]
        + ")"
    )
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(
        data=top_per_disease,
        x="composite_priority_score",
        y="label",
        hue="label",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Top-Ranked Therapeutic Candidate Molecules Across Representative Diseases", fontsize=13, weight="bold")
    ax.set_xlabel("Composite Priority Score (CPS, 0.0 - 1.0)", fontsize=11)
    ax.set_ylabel("Disease: Candidate Molecule (Biological Target)", fontsize=11)
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "top_candidate_scores.png", dpi=300)
    plt.close(fig)
    
    # 2. Score Component Breakdown (Stacked Bar for Top 10 Overall Candidates)
    top10 = candidates_df.head(10).copy()
    top10["candidate_label"] = (
        top10["local_disease"]
        + "\n"
        + top10["pref_name"].fillna(top10["molecule_chembl_id"])
    )
    
    comp_df = pd.DataFrame({
        "Candidate": top10["candidate_label"],
        "Target Assoc (35%)": top10["score_target_assoc"] * 0.35,
        "Potency (35%)": top10["score_potency"] * 0.35,
        "Drug-Likeness (15%)": top10["score_druglike"] * 0.15,
        "Clinical Evidence (15%)": top10["score_clinical_evidence"] * 0.15,
    }).set_index("Candidate")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    comp_df.plot(kind="barh", stacked=True, colormap="Spectral", ax=ax)
    ax.set_title("Transparent Multi-Criteria Score Breakdown for Top Prioritized Candidates", fontsize=13, weight="bold")
    ax.set_xlabel("Contribution to Composite Priority Score (CPS)", fontsize=11)
    ax.set_ylabel("Candidate (Disease & Compound)", fontsize=11)
    ax.set_xlim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Score Component")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "score_component_breakdown.png", dpi=300)
    plt.close(fig)
    
    # 3. Target Association vs Potency Scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        candidates_df["score_target_assoc"],
        candidates_df["median_pchembl"],
        c=candidates_df["composite_priority_score"],
        cmap="plasma",
        alpha=0.6,
        s=30,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Composite Priority Score (CPS)")
    ax.set_title("Open Targets Association Score vs. ChEMBL Bioactivity Potency (pChEMBL)", fontsize=13, weight="bold")
    ax.set_xlabel("Open Targets Disease-Target Association Score", fontsize=11)
    ax.set_ylabel("Measured Bioactivity (Median pChEMBL)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "pchembl_vs_association_scatter.png", dpi=300)
    plt.close(fig)
    
    # 4. Disease Target Association Distribution (Top target per disease)
    top_target_per_disease = (
        targets_df.groupby("local_disease", as_index=False, group_keys=False)
        .head(1)
        .sort_values(by="association_score", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=top_target_per_disease,
        x="association_score",
        y="local_disease",
        hue="local_disease",
        palette="crest",
        legend=False,
        ax=ax,
    )
    ax.set_title("Strongest Biological Drug Target Association by Disease", fontsize=13, weight="bold")
    ax.set_xlabel("Open Targets Association Score", fontsize=11)
    ax.set_ylabel("Disease", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "disease_target_prioritization_summary.png", dpi=300)
    plt.close(fig)
    
    logger.info("Saved all 4 prioritization visualization charts to %s", OUTPUTS_DIR)


def run_therapeutic_prioritization_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Runs complete candidate prioritization pipeline."""
    assoc_path = DATA_RAW_DIR / "therapeutic_associations.csv"
    act_path = DATA_RAW_DIR / "chembl_activity.csv"
    mol_path = DATA_RAW_DIR / "chembl_molecules.csv"
    
    for p in [assoc_path, act_path, mol_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p}")
            
    assoc_df = pd.read_csv(assoc_path)
    act_df = pd.read_csv(act_path)
    mol_df = pd.read_csv(mol_path)
    
    targets_ranked = rank_disease_targets(assoc_df)
    candidates_ranked, audit_summary = prioritize_candidates(assoc_df, act_df, mol_df)
    plot_prioritization_visualizations(candidates_ranked, targets_ranked)
    
    return targets_ranked, candidates_ranked, audit_summary


if __name__ == "__main__":
    t_df, c_df, audit = run_therapeutic_prioritization_pipeline()
    print("\nTherapeutic Candidate Prioritization Pipeline Succeeded!")
    print(f"Ranked Targets: {len(t_df)} | Ranked Candidates: {len(c_df)}")
    print(f"Top Candidate: {c_df.iloc[0]['local_disease']} -> {c_df.iloc[0]['pref_name']} (CPS={c_df.iloc[0]['composite_priority_score']:.4f})")
