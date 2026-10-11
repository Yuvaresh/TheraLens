"""
Unit and integration tests for Therapeutic Candidate Prioritization Engine.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSOC_PATH = PROJECT_ROOT / "data" / "raw" / "therapeutic_associations.csv"
ACT_PATH = PROJECT_ROOT / "data" / "raw" / "chembl_activity.csv"
MOL_PATH = PROJECT_ROOT / "data" / "raw" / "chembl_molecules.csv"
RANKED_TARGETS = PROJECT_ROOT / "data" / "processed" / "disease_target_rankings.csv"
RANKED_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "therapeutic_candidate_rankings.csv"

from src.models.prioritize_therapeutic_candidates import (
    compute_druglikeness_score,
    compute_evidence_bonus,
)


def test_input_schemas_and_identifiers():
    """Validates identifier preservation across Open Targets and ChEMBL."""
    assert ASSOC_PATH.exists() and ACT_PATH.exists() and MOL_PATH.exists()
    
    df_assoc = pd.read_csv(ASSOC_PATH)
    assert {"local_disease", "target_symbol", "association_score"}.issubset(df_assoc.columns)
    
    df_act = pd.read_csv(ACT_PATH)
    assert {"target_symbol", "molecule_chembl_id", "pchembl_value"}.issubset(df_act.columns)
    
    df_mol = pd.read_csv(MOL_PATH)
    assert {"molecule_chembl_id", "molecular_weight", "alogp", "psa"}.issubset(df_mol.columns)


def test_druglikeness_scoring_logic():
    """Tests Lipinski Rule of 5 scoring logic on synthetic molecular profiles."""
    # Ideal drug-like molecule
    ideal_row = pd.Series({"molecular_weight": 350.0, "alogp": 2.5, "psa": 65.0, "hba": 5, "hbd": 2})
    score_ideal, has_data = compute_druglikeness_score(ideal_row)
    assert score_ideal == 1.0
    assert has_data is True
    
    # Molecule failing MW and LogP
    failing_row = pd.Series({"molecular_weight": 750.0, "alogp": 6.5, "psa": 65.0, "hba": 5, "hbd": 2})
    score_failing, _ = compute_druglikeness_score(failing_row)
    assert score_failing == 0.6  # Passes 3 of 5 rules
    
    # Molecule with missing property values
    missing_row = pd.Series({"molecular_weight": np.nan, "alogp": np.nan, "psa": np.nan, "hba": np.nan, "hbd": np.nan})
    score_missing, has_data_missing = compute_druglikeness_score(missing_row)
    assert score_missing == 0.50
    assert has_data_missing is False


def test_clinical_evidence_bonus_logic():
    """Tests phase-based evidence bonus computation."""
    assert compute_evidence_bonus(4.0) == 1.00
    assert compute_evidence_bonus(3.0) == 0.80
    assert compute_evidence_bonus(2.0) == 0.60
    assert compute_evidence_bonus(1.0) == 0.40
    assert compute_evidence_bonus(np.nan) == 0.10


def test_candidate_rankings_output_integrity():
    """Verifies generated candidate rankings file and score boundaries."""
    assert RANKED_CANDIDATES.exists(), "Candidate rankings CSV must exist."
    df = pd.read_csv(RANKED_CANDIDATES)
    assert len(df) > 0, "Ranked candidates dataframe must not be empty."
    
    # Check that scores are bounded in [0.0, 1.0]
    cps = df["composite_priority_score"]
    assert (cps >= 0.0).all() and (cps <= 1.0).all(), "Composite priority score must be within [0.0, 1.0]."
    
    # Check that candidate_rank_within_disease is ordered
    sample_disease = df["local_disease"].iloc[0]
    sub = df[df["local_disease"] == sample_disease]
    assert sub["composite_priority_score"].is_monotonic_decreasing or (sub["candidate_rank_within_disease"].diff().dropna() >= 0).all()
