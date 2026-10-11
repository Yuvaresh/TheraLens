"""
Unit and integration tests for Therapeutic Timeline Validation & Audit Engine.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FDA_TIMELINE = PROJECT_ROOT / "data" / "processed" / "fda_drug_approval_timeline.csv"
FDA_AUDIT = PROJECT_ROOT / "data" / "processed" / "fda_chembl_match_audit.csv"
AUDIT_SUMMARY = PROJECT_ROOT / "outputs" / "timeline_analysis" / "mapping_audit_summary.json"


def test_fda_timeline_schema():
    """Validates FDA approval timeline dataset fields."""
    assert FDA_TIMELINE.exists(), "FDA approval timeline CSV must exist."
    df = pd.read_csv(FDA_TIMELINE)
    required = {"appl_no", "drug_name", "active_ingredient", "first_approval_year"}
    assert required.issubset(df.columns)
    assert len(df) > 10000, "Expected over 10,000 FDA records."


def test_fda_chembl_matching_rate():
    """Validates active ingredient matching rate exceeding 95%."""
    assert FDA_AUDIT.exists(), "FDA-ChEMBL match audit CSV must exist."
    df = pd.read_csv(FDA_AUDIT)
    matched = (df["match_status"] == "matched").sum()
    rate = matched / len(df)
    assert rate >= 0.95, f"Expected match rate >= 95%, got {rate * 100:.2f}%"


def test_evidence_classification_audit():
    """Verifies that timeline audit differentiates Phase 4 approvals from investigational trials."""
    assert AUDIT_SUMMARY.exists(), "Audit summary JSON must exist."
    with open(AUDIT_SUMMARY, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    tl_audit = data["master_timeline_audit"]
    evidence = tl_audit["evidence_classification"]
    
    assert "verified_phase4_approved_indication_records" in evidence
    assert "investigational_clinical_trials_phase1_3_records" in evidence
    # Ensure they are distinct and accounted for
    assert evidence["verified_phase4_approved_indication_records"] > 0
    assert evidence["investigational_clinical_trials_phase1_3_records"] > 0
    assert tl_audit["unique_disease_molecule_milestone_pairs"] < tl_audit["total_timeline_records"]
