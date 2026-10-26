#!/usr/bin/env python3
"""
TheraLens: Therapeutic Timeline Validation & Scientific Audit Engine

Performs rigorous scientific audit of FDA Drug Approvals and ChEMBL Drug Indications.
Addresses critical regulatory nuances:
1. Distinguishes between initial FDA molecule approval year and indication-specific label expansions.
2. Identifies Cartesian product expansion: 555,323 join rows represent product-indication combinations,
   not 555,323 distinct therapeutic milestones.
3. Classifies associations into Verified Phase-4 Milestones, Investigational Clinical Trials (Phase 1-3),
   and Inferred/Unverified associations.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "timeline_analysis"

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
logger = logging.getLogger("ValidateTimeline")


def audit_fda_timeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audits FDA Drug Approvals dataset."""
    fda_path = DATA_PROCESSED_DIR / "fda_drug_approval_timeline.csv"
    logger.info("Auditing FDA approvals timeline from %s...", fda_path)
    df_fda = pd.read_csv(fda_path)
    
    total_records = len(df_fda)
    unique_appl = df_fda["appl_no"].nunique()
    unique_ingredients = df_fda["active_ingredient"].nunique()
    unique_drugs = df_fda["drug_name"].nunique()
    
    missing_years = int(df_fda["first_approval_year"].isna().sum())
    valid_years = df_fda["first_approval_year"].dropna()
    min_year = int(valid_years.min()) if not valid_years.empty else None
    max_year = int(valid_years.max()) if not valid_years.empty else None
    
    audit_fda = {
        "total_fda_records": total_records,
        "unique_applications": unique_appl,
        "unique_active_ingredients": unique_ingredients,
        "unique_trade_names": unique_drugs,
        "missing_approval_years": missing_years,
        "missing_approval_year_pct": round((missing_years / total_records) * 100, 2),
        "earliest_approval_year": min_year,
        "latest_approval_year": max_year,
    }
    return df_fda, audit_fda


def audit_fda_chembl_mapping() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Audits FDA-to-ChEMBL entity resolution and match rates."""
    audit_path = DATA_PROCESSED_DIR / "fda_chembl_match_audit.csv"
    logger.info("Auditing FDA-ChEMBL match audit from %s...", audit_path)
    df_audit = pd.read_csv(audit_path)
    
    status_counts = df_audit["match_status"].value_counts().to_dict()
    matched_count = status_counts.get("matched", 0)
    unmatched_count = status_counts.get("no_match", 0)
    total_audited = len(df_audit)
    match_rate = round((matched_count / total_audited) * 100, 2) if total_audited else 0.0
    
    methods = df_audit[df_audit["match_status"] == "matched"]["match_method"].value_counts().to_dict()
    
    mapping_audit = {
        "total_active_ingredients_audited": total_audited,
        "matched_ingredients": matched_count,
        "unmatched_ingredients": unmatched_count,
        "overall_match_rate_pct": match_rate,
        "match_methods_breakdown": methods,
    }
    return df_audit, mapping_audit


def audit_master_timeline(chunk_size: int = 100000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Efficiently processes and audits the large (114 MB) master timeline dataset in chunks."""
    timeline_path = DATA_PROCESSED_DIR / "therapeutic_development_timeline.csv"
    logger.info("Processing master therapeutic timeline in chunks (%d per chunk)...", chunk_size)
    
    total_rows = 0
    phase_counts = {}
    disease_set = set()
    molecule_set = set()
    disease_mol_pairs = set()
    year_distribution = {}
    sample_records = []

    use_cols = ["disease_name", "chembl_molecule_id", "first_approval_year", "max_phase_for_ind"]

    for chunk in pd.read_csv(timeline_path, usecols=use_cols, chunksize=chunk_size):
        total_rows += len(chunk)
        
        # Accumulate phases
        p_counts = chunk["max_phase_for_ind"].value_counts(dropna=False)
        for p, count in p_counts.items():
            k = f"Phase_{p}" if pd.notna(p) else "Phase_Null"
            phase_counts[k] = phase_counts.get(k, 0) + int(count)
            
        disease_set.update(chunk["disease_name"].dropna().unique())
        molecule_set.update(chunk["chembl_molecule_id"].dropna().unique())
        
        # Accumulate unique pairs
        pairs = set(zip(chunk["disease_name"].astype(str), chunk["chembl_molecule_id"].astype(str)))
        disease_mol_pairs.update(pairs)
        
        # Years for Phase 4 (Approved)
        p4_years = chunk[chunk["max_phase_for_ind"] == 4.0]["first_approval_year"].dropna().astype(int)
        for yr, cnt in p4_years.value_counts().items():
            year_distribution[int(yr)] = year_distribution.get(int(yr), 0) + int(cnt)
            
        if len(sample_records) < 5:
            sample_records.extend(chunk.head(5).to_dict(orient="records"))

    # Evidence categorization
    p4_count = phase_counts.get("Phase_4.0", 0)
    investigational_count = (
        phase_counts.get("Phase_1.0", 0)
        + phase_counts.get("Phase_2.0", 0)
        + phase_counts.get("Phase_3.0", 0)
        + phase_counts.get("Phase_0.5", 0)
    )
    preclinical_unverified = phase_counts.get("Phase_Null", 0) + phase_counts.get("Phase_-1.0", 0)

    timeline_audit = {
        "total_timeline_records": total_rows,
        "unique_standardized_diseases": len(disease_set),
        "unique_chembl_molecules": len(molecule_set),
        "unique_disease_molecule_milestone_pairs": len(disease_mol_pairs),
        "average_cartesian_redundancy_factor": round(total_rows / max(len(disease_mol_pairs), 1), 2),
        "evidence_classification": {
            "verified_phase4_approved_indication_records": p4_count,
            "verified_phase4_pct": round((p4_count / total_rows) * 100, 2),
            "investigational_clinical_trials_phase1_3_records": investigational_count,
            "investigational_pct": round((investigational_count / total_rows) * 100, 2),
            "preclinical_or_unverified_records": preclinical_unverified,
        },
        "phase_breakdown_detail": phase_counts,
        "scientific_caveat_summary": (
            "CRITICAL REGULATORY DISTINCTION: 1) Initial FDA approval year indicates when a chemical entity "
            "first received any regulatory marketing approval in the United States. 2) In secondary indications, "
            "initial approval year precedes the indication approval. 3) Records with Phase < 4.0 represent "
            "clinical investigation, NOT verified disease-specific market authorizations."
        ),
    }
    
    # Yearly summary dataframe
    year_df = pd.DataFrame(
        [{"year": y, "phase4_milestones": c} for y, c in sorted(year_distribution.items())]
    )
    return year_df, timeline_audit


def plot_timeline_visualizations(
    df_fda: pd.DataFrame,
    year_df: pd.DataFrame,
    timeline_audit: Dict[str, Any],
):
    """Generates 4 publication-grade analytical diagnostic plots."""
    logger.info("Generating timeline audit visualization charts...")
    
    # 1. Evidence Classification Breakdown (Verified Phase 4 vs Investigational vs Preclinical)
    classes = [
        "Verified Phase 4\nRegulatory Approval",
        "Investigational Trials\n(Phase 1 - 3)",
        "Preclinical / Unverified\nIndications",
    ]
    counts = [
        timeline_audit["evidence_classification"]["verified_phase4_approved_indication_records"],
        timeline_audit["evidence_classification"]["investigational_clinical_trials_phase1_3_records"],
        timeline_audit["evidence_classification"]["preclinical_or_unverified_records"],
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=classes, y=counts, hue=classes, palette=["#2ca02c", "#ff7f0e", "#7f7f7f"], legend=False, ax=ax)
    ax.set_title("Master Therapeutic Timeline: Evidence Categorization Audit", fontsize=13, weight="bold")
    ax.set_ylabel("Record Volume (Cartesian Expanded)", fontsize=11)
    for bar in bars.patches:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}\n({(height/timeline_audit['total_timeline_records'])*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "timeline_evidence_classification.png", dpi=300)
    plt.close(fig)
    
    # 2. Historical FDA First Approval Year Distribution
    valid_fda = df_fda["first_approval_year"].dropna().astype(int)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(valid_fda, bins=50, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Historical FDA Initial Product Approvals Over Time (1939–2026)", fontsize=13, weight="bold")
    ax.set_xlabel("First Approval Year", fontsize=11)
    ax.set_ylabel("Number of FDA Product Approvals", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "fda_approval_year_distribution.png", dpi=300)
    plt.close(fig)
    
    # 3. Cartesian Duplication Impact (555k join records vs unique entities)
    metric_labels = [
        "Master Join Rows\n(Products x Indications)",
        "Unique Disease-Molecule\nMilestone Pairs",
        "Unique ChEMBL\nMolecules",
        "Unique Standardized\nDiseases",
    ]
    metric_values = [
        timeline_audit["total_timeline_records"],
        timeline_audit["unique_disease_molecule_milestone_pairs"],
        timeline_audit["unique_chembl_molecules"],
        timeline_audit["unique_standardized_diseases"],
    ]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = sns.barplot(x=metric_labels, y=metric_values, hue=metric_labels, palette="rocket", legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Audit of Cartesian Join Duplication vs. Unique Biological Entities", fontsize=13, weight="bold")
    ax.set_ylabel("Entity Count (Logarithmic Scale)", fontsize=11)
    for bar, val in zip(bars.patches, metric_values):
        ax.annotate(
            f"{val:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2),
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "cartesian_duplication_impact.png", dpi=300)
    plt.close(fig)
    
    # 4. Annual Volume of Phase-4 Approved Milestones (Historical Trend)
    if not year_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=year_df, x="year", y="phase4_milestones", color="#2ca02c", linewidth=2.5, ax=ax)
        ax.set_title("Historical Volume of Phase-4 Approved Therapeutic Milestones by Approval Year", fontsize=13, weight="bold")
        ax.set_xlabel("FDA First Approval Year", fontsize=11)
        ax.set_ylabel("Associated Phase-4 Indication Milestones", fontsize=11)
        plt.tight_layout()
        fig.savefig(OUTPUTS_DIR / "phase4_milestones_timeline_trend.png", dpi=300)
        plt.close(fig)
        
    logger.info("Saved all 4 timeline audit charts to %s", OUTPUTS_DIR)


def run_timeline_validation_pipeline() -> Dict[str, Any]:
    """Runs complete timeline validation and quality audit pipeline."""
    df_fda, audit_fda = audit_fda_timeline()
    df_audit_map, audit_map = audit_fda_chembl_mapping()
    year_df, audit_timeline = audit_master_timeline()
    
    # Save completeness tables
    fda_completeness = pd.DataFrame([audit_fda])
    fda_completeness.to_csv(OUTPUTS_DIR / "approval_year_completeness.csv", index=False)
    
    plot_timeline_visualizations(df_fda, year_df, audit_timeline)
    
    full_audit = {
        "fda_approvals_audit": audit_fda,
        "mapping_resolution_audit": audit_map,
        "master_timeline_audit": audit_timeline,
    }
    
    with open(OUTPUTS_DIR / "mapping_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)
    logger.info("Saved mapping audit summary to %s", OUTPUTS_DIR / "mapping_audit_summary.json")
    
    # Generate readable text report
    report_text = f"""================================================================================
TheraLens: Master Therapeutic Development Timeline Quality & Scientific Audit
================================================================================

1. FDA APPROVALS TIMELINE SUMMARY
--------------------------------------------------------------------------------
- Total FDA Records:             {audit_fda['total_fda_records']:,}
- Unique Applications (NDAs):    {audit_fda['unique_applications']:,}
- Unique Active Ingredients:     {audit_fda['unique_active_ingredients']:,}
- Unique Trade Names:            {audit_fda['unique_trade_names']:,}
- Records with Missing Year:     {audit_fda['missing_approval_years']:,} ({audit_fda['missing_approval_year_pct']}%)
- Historical Year Range:         {audit_fda['earliest_approval_year']} to {audit_fda['latest_approval_year']}

2. FDA-TO-CHEMBL ENTITY RESOLUTION AUDIT
--------------------------------------------------------------------------------
- Ingredients Audited:           {audit_map['total_active_ingredients_audited']:,}
- Successfully Matched:          {audit_map['matched_ingredients']:,} ({audit_map['overall_match_rate_pct']}%)
- Unmatched Ingredients:         {audit_map['unmatched_ingredients']:,}
- Top Match Methods:             {audit_map['match_methods_breakdown']}

3. MASTER TIMELINE CARTESIAN JOIN AUDIT
--------------------------------------------------------------------------------
- Total Join Records:            {audit_timeline['total_timeline_records']:,}
- Unique Diseases Covered:       {audit_timeline['unique_standardized_diseases']:,}
- Unique ChEMBL Molecules:       {audit_timeline['unique_chembl_molecules']:,}
- Unique Disease-Molecule Pairs: {audit_timeline['unique_disease_molecule_milestone_pairs']:,}
- Cartesian Redundancy Factor:   {audit_timeline['average_cartesian_redundancy_factor']}x
  (Average records generated per distinct disease-molecule pair due to multiple product dosages)

4. EVIDENCE CLASSIFICATION BREAKDOWN
--------------------------------------------------------------------------------
- Verified Phase-4 Milestones:   {audit_timeline['evidence_classification']['verified_phase4_approved_indication_records']:,} ({audit_timeline['evidence_classification']['verified_phase4_pct']}%)
- Investigational (Phase 1-3):   {audit_timeline['evidence_classification']['investigational_clinical_trials_phase1_3_records']:,} ({audit_timeline['evidence_classification']['investigational_pct']}%)
- Preclinical / Unverified:      {audit_timeline['evidence_classification']['preclinical_or_unverified_records']:,}

5. CRITICAL SCIENTIFIC & REGULATORY STATEMENT
--------------------------------------------------------------------------------
{audit_timeline['scientific_caveat_summary']}
================================================================================
"""
    with open(OUTPUTS_DIR / "timeline_quality_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Saved timeline quality report text to %s", OUTPUTS_DIR / "timeline_quality_report.txt")
    
    return full_audit


if __name__ == "__main__":
    audit = run_timeline_validation_pipeline()
    print("\nTherapeutic Timeline Validation Pipeline Succeeded!")
    print(f"Total Timeline Rows: {audit['master_timeline_audit']['total_timeline_records']:,}")
    print(f"Unique Disease-Molecule Pairs: {audit['master_timeline_audit']['unique_disease_molecule_milestone_pairs']:,}")
    print(f"Verified Phase 4 Milestone Rows: {audit['master_timeline_audit']['evidence_classification']['verified_phase4_approved_indication_records']:,}")
