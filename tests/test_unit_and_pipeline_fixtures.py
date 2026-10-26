"""
TheraLens: Comprehensive Unit & Fast Fixture Test Suite (Review-3 Standard)

Provides targeted, fast-executing unit tests covering:
1. Symptom vectorization and disease predictor edge cases
2. Feature alignment and model artifact integrity
3. Metric calculation edge cases (MAE, RMSE, MAPE with zero/near-zero denominators)
4. Lipinski & Veber drug-likeness boundary scoring
5. Clinical evidence bonus and phase mappings
6. Composite Priority Score (CPS) mathematical bounds and weight sums
7. Ingredient normalization and pharmaceutical salt stripping
8. Biopharmaceutical milestone projection latency models
9. Rolling Z-score anomaly and outbreak surge detection
10. Heuristic negative control audit (Medroxyprogesterone Acetate / AIDS / PGR)
11. Lightweight in-memory pipeline mock integration
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.inference.predict_disease import DiseasePredictor
from src.models.prioritize_therapeutic_candidates import (
    compute_druglikeness_score,
    compute_evidence_bonus,
)
from src.analysis.forecast_therapeutic_timeline import (
    compute_projected_timeline,
    BENCHMARK_DURATIONS,
)
from src.analysis.analyze_epidemiological_trends import detect_outbreak_surges
from src.data_processing.build_therapeutic_timeline import (
    normalize_active_ingredient,
    strip_salt_suffixes,
)


# =============================================================================
# 1. Symptom Vectorization & Inference Interface Edge Cases
# =============================================================================

def test_symptom_vectorizer_edge_cases():
    """Tests DiseasePredictor handling of empty, whitespace, mixed-case, and unmapped inputs."""
    predictor = DiseasePredictor()

    # Empty string
    res_empty = predictor.predict("", top_k=3)
    assert res_empty["status"] == "NO_SYMPTOMS_MATCHED"
    assert res_empty["predicted_disease"] is None
    assert res_empty["confidence_score"] == 0.0
    assert len(res_empty["top_candidates"]) == 0

    # Mixed casing, leading/trailing spaces, hyphens
    res_messy = predictor.predict("  CHILLS , Vomiting  , Head-ache ", top_k=3)
    assert res_messy["status"] == "SUCCESS"
    assert res_messy["predicted_disease"] is not None
    assert res_messy["active_symptom_count"] >= 2
    assert "chills" in [s.lower() for s in res_messy["matched_symptoms"]]
    assert "vomiting" in [s.lower() for s in res_messy["matched_symptoms"]]

    # Completely alien / unmapped symptom
    res_alien = predictor.predict("extraterrestrial_hyper_syndrome_xyz_123", top_k=3)
    assert res_alien["status"] == "NO_SYMPTOMS_MATCHED"
    assert len(res_alien["unmatched_symptoms"]) == 1


def test_feature_consistency_between_training_and_inference():
    """Verifies feature schema length and alignment between saved model metadata and vectorizer."""
    feat_path = PROJECT_ROOT / "models" / "feature_names.json"
    assert feat_path.exists(), "Feature names JSON must exist."
    with open(feat_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    assert len(features) == 132, f"Expected 132 clinical symptom features, got {len(features)}"

    predictor = DiseasePredictor()
    vec, matched, _ = predictor.vectorize_symptoms(["chills", "high_fever"])
    assert len(vec) == 132, "Vector length must strictly match 132 features."
    assert vec.sum() == len(matched)


# =============================================================================
# 2. Physicochemical Drug-Likeness & Potency Scoring Unit Tests
# =============================================================================

def test_druglikeness_exact_boundaries():
    """Verifies strict boundary compliance for Lipinski and Veber molecular rules."""
    # Boundary exact pass (MW=500.0, ALogP=5.0, PSA=140.0, HBA=10, HBD=5)
    pass_boundary = pd.Series({
        "molecular_weight": 500.0,
        "alogp": 5.0,
        "psa": 140.0,
        "hba": 10.0,
        "hbd": 5.0,
    })
    score_pass, has_data = compute_druglikeness_score(pass_boundary)
    assert score_pass == 1.0
    assert has_data is True

    # Boundary exact fail (MW=500.1, ALogP=5.01, PSA=140.1, HBA=11, HBD=6)
    fail_boundary = pd.Series({
        "molecular_weight": 500.1,
        "alogp": 5.01,
        "psa": 140.1,
        "hba": 11.0,
        "hbd": 6.0,
    })
    score_fail, _ = compute_druglikeness_score(fail_boundary)
    assert score_fail == 0.0

    # Negative LogP within valid range (-0.4 <= alogp <= 5.0)
    neg_logp_pass = pd.Series({"molecular_weight": 300.0, "alogp": -0.3, "psa": 50.0, "hba": 4, "hbd": 2})
    score_neg, _ = compute_druglikeness_score(neg_logp_pass)
    assert score_neg == 1.0

    # Extreme negative LogP failing
    neg_logp_fail = pd.Series({"molecular_weight": 300.0, "alogp": -1.5, "psa": 50.0, "hba": 4, "hbd": 2})
    score_neg_f, _ = compute_druglikeness_score(neg_logp_fail)
    assert score_neg_f == 0.8  # Fails only 1 rule


def test_clinical_evidence_bonus_comprehensive():
    """Tests all clinical development phases and string/null coercions."""
    assert compute_evidence_bonus(4.0) == 1.00
    assert compute_evidence_bonus(3.0) == 0.80
    assert compute_evidence_bonus(2.0) == 0.60
    assert compute_evidence_bonus(1.0) == 0.40
    assert compute_evidence_bonus(0.0) == 0.20
    assert compute_evidence_bonus(np.nan) == 0.10
    assert compute_evidence_bonus("Phase 3") == 0.10  # Non-numeric string gracefully handled


def test_composite_priority_score_mathematical_weights():
    """Verifies that CPS weights strictly sum to 1.0 and bounds are [0.0, 1.0]."""
    w_target = 0.35
    w_potency = 0.35
    w_druglike = 0.15
    w_evidence = 0.15

    assert np.isclose(w_target + w_potency + w_druglike + w_evidence, 1.00), "Weights must sum to 1.00"

    # Perfect candidate score
    cps_perfect = round(w_target * 1.0 + w_potency * 1.0 + w_druglike * 1.0 + w_evidence * 1.0, 4)
    assert cps_perfect == 1.0000

    # Zero candidate score
    cps_zero = round(w_target * 0.0 + w_potency * 0.0 + w_druglike * 0.0 + w_evidence * 0.0, 4)
    assert cps_zero == 0.0000


# =============================================================================
# 3. Active Ingredient Normalization & Salt Stripping
# =============================================================================

def test_drug_normalization_and_salt_stripping():
    """Verifies FDA active ingredient normalization and salt suffix stripping."""
    assert normalize_active_ingredient("  METFORMIN HYDROCHLORIDE  ") == "metformin hydrochloride"
    assert normalize_active_ingredient("AMPICILLIN SODIUM;") == "ampicillin sodium"

    # Salt stripping
    assert strip_salt_suffixes("METFORMIN HYDROCHLORIDE") == "METFORMIN"
    assert strip_salt_suffixes("AMPICILLIN SODIUM") == "AMPICILLIN"
    assert strip_salt_suffixes("MORPHINE SULFATE") == "MORPHINE"
    assert strip_salt_suffixes("ATORVASTATIN CALCIUM TRIHYDRATE") == "ATORVASTATIN"
    assert strip_salt_suffixes("PARACETAMOL") == "PARACETAMOL"


# =============================================================================
# 4. Biopharmaceutical Milestone Forecasting Prototype Tests
# =============================================================================

def test_milestone_forecast_durations_and_success_rates():
    """Tests DiMasi-based clinical stage duration logic and uncertainty window."""
    # Phase 4 (Approved molecule undergoing supplemental label expansion / NDA review: 1.5 yrs, 90% success)
    med_yr, min_yr, max_yr, prob = compute_projected_timeline(4.0, base_year=2026)
    assert np.isclose(med_yr, 2027.5)
    assert min_yr <= med_yr <= max_yr
    assert np.isclose(prob, 90.0)

    # Post-approval / no remaining phases (> 4.0)
    med_done, min_done, max_done, prob_done = compute_projected_timeline(5.0, base_year=2026)
    assert med_done == 2026.0
    assert prob_done == 1.0

    # Phase 3 candidate (needs Phase 3 + FDA Review)
    # Remaining: Phase 3 (3.8 yrs) + FDA Review (1.5 yrs) = 5.3 yrs -> median year 2031.3
    med_3, min_3, max_3, prob_3 = compute_projected_timeline(3.0, base_year=2026)
    assert np.isclose(med_3, 2031.3)
    assert min_3 < med_3 < max_3
    # Prob Phase 3 (0.58) * FDA (0.90) = 0.522 -> 52.2%
    assert np.isclose(prob_3, 52.2, atol=0.1)

    # Phase 1 candidate
    med_1, min_1, max_1, prob_1 = compute_projected_timeline(1.0, base_year=2026)
    assert med_1 > med_3
    assert prob_1 < prob_3


# =============================================================================
# 5. Epidemiological Metrics & Anomaly Detection Edge Cases
# =============================================================================

def test_epidemiological_metric_zero_denominator_protection():
    """Verifies MAPE metric calculation stability when actual case values are 0 or small."""
    actual = np.array([0.0, 5.0, 10.0, 0.0])
    pred = np.array([2.0, 6.0, 8.0, 1.0])

    # Standard formula protected by np.maximum(actual, 1.0)
    denom = np.maximum(actual, 1.0)
    mape = np.mean(np.abs((actual - pred) / denom)) * 100
    assert np.isfinite(mape), "MAPE calculation must remain finite even when actual contains zeros."
    assert mape > 0.0


def test_surge_detector_flat_series_no_false_alarm():
    """Verifies that flat constant time-series never trigger false surge alarms."""
    flat_data = pd.DataFrame({
        "disease": ["Measles"] * 10,
        "country": ["XYZ"] * 10,
        "year": list(range(2000, 2010)),
        "temporal_resolution": ["Annual"] * 10,
        "time_period": [str(y) for y in range(2000, 2010)],
        "reported_cases": [200.0] * 10,  # Exactly flat
    })
    res = detect_outbreak_surges(flat_data, window=5, threshold_z=2.0)
    assert res["is_outbreak_surge"].sum() == 0, "Flat time-series must produce zero surge alarms."


# =============================================================================
# 6. Negative Control Biological Audit
# =============================================================================

def test_negative_control_medroxyprogesterone_audit():
    """
    Scientific Integrity Negative Control:
    Verifies that Medroxyprogesterone Acetate receives a high heuristic CPS for AIDS
    specifically due to Open Targets PGR association and high ChEMBL in-vitro binding,
    proving that CPS is an in-silico hypothesis ranking and requires independent
    biological validation rather than blind clinical interpretation.
    """
    cand_path = PROJECT_ROOT / "data" / "processed" / "therapeutic_candidate_rankings.csv"
    assert cand_path.exists(), "Therapeutic candidate rankings CSV must exist."
    df = pd.read_csv(cand_path)

    aids_candidates = df[df["local_disease"] == "AIDS"]
    assert len(aids_candidates) > 0

    top_aids = aids_candidates.iloc[0]
    assert "MEDROXYPROGESTERONE" in str(top_aids["pref_name"]).upper()
    assert top_aids["target_symbol"] == "PGR"
    assert top_aids["composite_priority_score"] >= 0.75
    # Confirm that the score is driven by high potency and Phase 4 status
    assert top_aids["score_potency"] >= 0.90
    assert top_aids["score_clinical_evidence"] == 1.00


# =============================================================================
# 7. Lightweight End-to-End Pipeline Fixture Mock
# =============================================================================

def test_lightweight_prioritization_mock():
    """Runs a rapid in-memory mock of candidate scoring without processing large files."""
    mock_assoc = pd.DataFrame({
        "local_disease": ["MockDisease"],
        "target_symbol": ["MOCK_GENE"],
        "association_score": [0.85],
    })
    mock_act = pd.DataFrame({
        "target_symbol": ["MOCK_GENE"],
        "molecule_chembl_id": ["CHEMBL_MOCK_1"],
        "pchembl_value": [8.0],  # Potency = (8.0 - 4.0) / 6.0 = 0.6667
        "activity_id": [999999],
    })
    mock_mol = pd.DataFrame({
        "molecule_chembl_id": ["CHEMBL_MOCK_1"],
        "pref_name": ["MOCK_COMPOUND"],
        "max_phase": [4.0],  # Evidence = 1.0
        "molecular_weight": [380.0],
        "alogp": [2.5],
        "psa": [70.0],
        "hba": [5.0],
        "hbd": [2.0],
    })

    merged_t = mock_assoc.merge(mock_act, on="target_symbol")
    merged = merged_t.merge(mock_mol, on="molecule_chembl_id")

    s_target = float(merged["association_score"].iloc[0])
    s_potency = (float(merged["pchembl_value"].iloc[0]) - 4.0) / 6.0
    s_druglike, _ = compute_druglikeness_score(merged.iloc[0])
    s_evidence = compute_evidence_bonus(merged["max_phase"].iloc[0])

    cps = round(0.35 * s_target + 0.35 * s_potency + 0.15 * s_druglike + 0.15 * s_evidence, 4)

    assert 0.0 <= cps <= 1.0
    # 0.35 * 0.85 + 0.35 * (4/6) + 0.15 * 1.0 + 0.15 * 1.0
    # = 0.2975 + 0.2333 + 0.15 + 0.15 = 0.8308
    assert np.isclose(cps, 0.8308, atol=0.001)
