#!/usr/bin/env python3
"""
Unit and integration tests for the TheraLens emerging disease research workflow:
- Symptom similarity calculation (open-world Jaccard / Cosine / Overlap)
- Disease-to-therapeutic-evidence linkage (targets, ChEMBL candidates, negative controls)
- Scientific forecasting & scenario analysis (DiMasi/Hay benchmarks, technology scenarios)
- End-to-end orchestration and report generation (JSON & Markdown dossiers)
"""

import json
import pytest
from pathlib import Path

from src.models.symptom_similarity import SymptomSimilarityEngine
from src.models.therapeutic_evidence_retriever import TherapeuticEvidenceRetriever
from src.analysis.emerging_disease_forecaster import EmergingDiseaseForecaster
from src.inference.forecast_emerging_disease import TheraLensUnifiedOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# MODULE FIXTURES (Scope: Module for fast execution without deprecation warnings)
# ============================================================================

@pytest.fixture(scope="module")
def similarity_engine():
    return SymptomSimilarityEngine()


@pytest.fixture(scope="module")
def evidence_retriever():
    return TherapeuticEvidenceRetriever()


@pytest.fixture(scope="module")
def forecaster():
    return EmergingDiseaseForecaster(base_year=2026)


@pytest.fixture(scope="module")
def orchestrator():
    return TheraLensUnifiedOrchestrator()


# ============================================================================
# 1. SYMPTOM SIMILARITY TESTS
# ============================================================================

class TestSymptomSimilarityEngine:
    def test_vocabulary_loaded(self, similarity_engine):
        assert len(similarity_engine.feature_names) == 132
        assert len(similarity_engine.disease_profiles) > 0
        assert "Malaria" in similarity_engine.disease_profiles

    def test_exact_symptom_match(self, similarity_engine):
        malaria_symptoms = similarity_engine.disease_profiles["Malaria"]["symptoms"]
        res = similarity_engine.calculate_similarity(malaria_symptoms, top_k=5, min_similarity_threshold=0.1)
        assert res["status"] == "SUCCESS"
        assert len(res["matches"]) > 0
        top_match = res["matches"][0]
        assert top_match["disease"] == "Malaria"
        assert top_match["jaccard_similarity"] == 1.0
        assert top_match["differing_reference_symptoms"] == []
        assert top_match["differing_query_symptoms"] == []

    def test_partial_overlap(self, similarity_engine):
        query = ["chills", "vomiting", "high_fever", "sweating"]
        res = similarity_engine.calculate_similarity(query, top_k=3, min_similarity_threshold=0.1)
        assert res["status"] == "SUCCESS"
        diseases = [m["disease"] for m in res["matches"]]
        assert "Malaria" in diseases
        malaria_match = next(m for m in res["matches"] if m["disease"] == "Malaria")
        assert 0.0 < malaria_match["jaccard_similarity"] < 1.0
        assert len(malaria_match["matching_symptoms"]) >= 3

    def test_empty_input(self, similarity_engine):
        res = similarity_engine.calculate_similarity("", top_k=5)
        assert res["status"] == "EMPTY_QUERY"
        assert res["matches"] == []
        assert res["active_symptom_count"] == 0

    def test_duplicate_symptoms(self, similarity_engine):
        query = "chills, chills, vomiting, CHILLS, vomiting"
        res = similarity_engine.calculate_similarity(query, top_k=5)
        assert res["status"] == "SUCCESS"
        assert len(res["recognized_symptoms"]) == 2
        assert "chills" in res["recognized_symptoms"]
        assert "vomiting" in res["recognized_symptoms"]

    def test_unrecognized_and_alien_symptoms(self, similarity_engine):
        query = "chills, alien_tentacles, glowing_eyes, high_fever"
        res = similarity_engine.calculate_similarity(query, top_k=5)
        assert res["status"] == "SUCCESS"
        assert "alien_tentacles" in res["unrecognized_symptoms"]
        assert "glowing_eyes" in res["unrecognized_symptoms"]
        assert len(res["recognized_symptoms"]) == 2

    def test_case_and_whitespace_normalization(self, similarity_engine):
        query = "  HIGH_FEVER ,   ChIlLs  ,  HEADACHE  "
        res = similarity_engine.calculate_similarity(query, top_k=5)
        assert res["status"] == "SUCCESS"
        assert "high_fever" in res["recognized_symptoms"]
        assert "chills" in res["recognized_symptoms"]
        assert "headache" in res["recognized_symptoms"]

    def test_alias_mapping(self, similarity_engine):
        query = "fever, rash, joint_pain"
        res = similarity_engine.calculate_similarity(query, top_k=5)
        assert res["status"] == "SUCCESS"
        assert "high_fever" in res["recognized_symptoms"]
        assert "skin_rash" in res["recognized_symptoms"]

    def test_threshold_filtering(self, similarity_engine):
        query = "itching"
        res_strict = similarity_engine.calculate_similarity(query, top_k=10, min_similarity_threshold=0.8)
        assert res_strict["status"] == "INSUFFICIENT_SIMILARITY"
        assert res_strict["qualifying_count"] == 0
        assert res_strict["top_match"]["jaccard_similarity"] < 0.8

    def test_ranking_order_descending(self, similarity_engine):
        query = "chills, vomiting, high_fever, sweating, headache, nausea"
        res = similarity_engine.calculate_similarity(query, top_k=10, min_similarity_threshold=0.01)
        scores = [m["jaccard_similarity"] for m in res["matches"]]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# 2. THERAPEUTIC EVIDENCE LINKAGE TESTS
# ============================================================================

class TestTherapeuticEvidenceRetriever:
    def test_retriever_initialization(self, evidence_retriever):
        assert len(evidence_retriever.candidate_diseases) > 0
        assert "malaria" in evidence_retriever.candidate_diseases

    def test_retrieve_malaria_evidence(self, evidence_retriever):
        evidence = evidence_retriever.retrieve_evidence_for_disease("Malaria", top_targets=3, top_candidates=3)
        assert evidence["status"] == "EVIDENCE_RETRIEVED"
        assert len(evidence["top_targets"]) > 0
        assert len(evidence["top_candidates"]) > 0
        lead_candidate = evidence["top_candidates"][0]
        assert "composite_priority_score" in lead_candidate
        assert "chembl_id" in lead_candidate
        assert "clinical_phase" in lead_candidate

    def test_negative_control_annotation(self, evidence_retriever):
        evidence = evidence_retriever.retrieve_evidence_for_disease("AIDS", top_targets=10, top_candidates=10)
        assert evidence["status"] == "EVIDENCE_RETRIEVED"
        candidates = evidence["top_candidates"]
        medroxy_candidates = [
            c for c in candidates if "MEDROXYPROGESTERONE" in c.get("compound_name", "").upper()
        ]
        if medroxy_candidates:
            assert medroxy_candidates[0]["is_negative_control_surrogate"] is True
            assert "Negative-Control" in medroxy_candidates[0]["evidence_category"]

    def test_disease_without_candidates(self, evidence_retriever):
        evidence = evidence_retriever.retrieve_evidence_for_disease("Completely Unknown Disease XYZ")
        assert evidence["status"] == "NO_EVIDENCE_FOUND"
        assert evidence["top_candidates"] == []
        assert "missing_reasons" in evidence

    def test_case_insensitive_lookup(self, evidence_retriever):
        ev1 = evidence_retriever.retrieve_evidence_for_disease("malaria")
        ev2 = evidence_retriever.retrieve_evidence_for_disease("MALARIA")
        assert ev1["status"] == ev2["status"]
        assert len(ev1["top_candidates"]) == len(ev2["top_candidates"])

    def test_lindane_disqualified_as_malaria_surrogate(self, evidence_retriever):
        """REGRESSION TEST: Lindane (pesticide/scabicide) must be disqualified as Tier 4 toxicant and excluded from top candidates."""
        evidence = evidence_retriever.retrieve_evidence_for_disease("Malaria", top_targets=5, top_candidates=5)
        assert evidence["status"] == "EVIDENCE_RETRIEVED"
        cands = evidence["top_candidates"]
        assert len(cands) > 0
        # In the top candidates, Lindane must NOT appear because it is pushed to the bottom of all candidates
        assert all("LINDANE" not in c["compound_name"].upper() for c in cands)
        assert all(c.get("is_disqualified", False) is False for c in cands)
        # Check that Lindane itself is registered as disqualified
        is_direct, _, _ = evidence_retriever._get_indication_evidence("CHEMBL15891", "Malaria")
        assert is_direct is False


# ============================================================================
# 3. SCIENTIFIC FORECASTING TESTS
# ============================================================================

class TestEmergingDiseaseForecaster:
    def test_repurposing_pathway_forecast(self, forecaster):
        candidate = {
            "chembl_id": "CHEMBL416",
            "compound_name": "TEST_PHASE_4_DRUG",
            "clinical_phase": 4,
            "has_direct_indication": True,
            "indication_phase": 4.0,
            "target_symbol": "TEST_TARGET",
            "composite_priority_score": 0.85,
        }
        res = forecaster.forecast_timeline(
            emerging_disease_name="Test Syndrome",
            closest_disease_name="Malaria",
            symptom_similarity=0.75,
            lead_candidate=candidate,
            current_year=2026,
        )
        assert res["status"] == "FORECAST_GENERATED"
        assert res["development_pathway"] == "Approved Drug Repurposing"
        assert res["base_duration_years"] == 8.5  # Phase 2 (3.2) + Phase 3 (3.8) + FDA Review (1.5) = 8.5y
        base_scenario = res["scenarios"]["historical_baseline"]
        assert base_scenario["predicted_year"] in [2034, 2035]
        assert base_scenario["prediction_interval_80pct"]["min_year"] < base_scenario["predicted_year"]
        assert base_scenario["prediction_interval_80pct"]["max_year"] > base_scenario["predicted_year"]

    def test_preclinical_lead_pathway_forecast(self, forecaster):
        candidate = {
            "chembl_id": "CHEMBL999",
            "compound_name": "NOVEL_LEAD",
            "clinical_phase": 0,
            "target_symbol": "NOVEL_TARGET",
            "composite_priority_score": 0.60,
        }
        res = forecaster.forecast_timeline(
            emerging_disease_name="Test Preclinical Syndrome",
            closest_disease_name="Malaria",
            symptom_similarity=0.60,
            lead_candidate=candidate,
            current_year=2026,
        )
        assert res["status"] == "FORECAST_GENERATED"
        assert res["development_pathway"] == "De Novo / Preclinical Lead Advancement"
        # Preclinical base duration: 4.5 + 2.1 + 3.2 + 3.8 + 1.5 = 15.1y
        assert res["base_duration_years"] == 15.1
        base_scenario = res["scenarios"]["historical_baseline"]
        assert base_scenario["predicted_year"] in [2041, 2042]

    def test_malaria_evidence_does_not_forecast_lindane_repurposing(self, forecaster, evidence_retriever):
        """REGRESSION TEST: Real malaria evidence must NOT forecast Lindane as an approved repurposing drug (8.5y)."""
        malaria_evidence = evidence_retriever.retrieve_evidence_for_disease("Malaria", top_targets=5, top_candidates=5)
        sim_match = {"disease": "Malaria", "jaccard_similarity": 0.625}
        res = forecaster.forecast_from_evidence(
            emerging_disease_name="Novel Febrile Syndrome",
            similarity_match=sim_match,
            therapeutic_evidence=malaria_evidence,
        )
        assert res["status"] == "FORECAST_GENERATED"
        # Lindane must be filtered out as Tier 4 toxicant
        assert "LINDANE" not in res["lead_candidate"]["compound_name"].upper()
        # Must not be classified as approved repurposing
        assert "Repurposing" not in res["pathway_type"]
        assert res["baseline_duration_years"] >= 10.0
        assert res["is_validated_ml_prediction"] is False
        assert "Exploratory Scenario Simulation" in res["forecast_methodology"]
        assert "therapeutic_transferability_assessment" in res

        # SCIENTIFIC AUDIT: Verify Dual-Milestone Accounting
        assert "milestone_accounting" in res
        m_acc = res["milestone_accounting"]
        assert "horizon_a_continuation" in m_acc
        assert "horizon_b_revalidation" in m_acc
        ha = m_acc["horizon_a_continuation"]
        hb = m_acc["horizon_b_revalidation"]
        # Horizon A assumes Phase 1 completed: remaining are Phase 2, Phase 3, FDA Review (8.5y)
        assert ha["baseline_duration_years"] == 8.5
        assert ha["phases_remaining"] == ["phase_2", "phase_3", "fda_review"]
        # Horizon B conservatively accounts for Phase 1 in emerging cohort: Phase 1, Phase 2, Phase 3, FDA Review (10.6y)
        assert hb["baseline_duration_years"] == 10.6
        assert hb["phases_remaining"] == ["phase_1", "phase_2", "phase_3", "fda_review"]

        # Pharmacological role check: Ritonavir must be annotated as PK booster / host target interactor
        lead = res["lead_candidate"]
        assert "RITONAVIR" in lead["compound_name"].upper()
        assert "CYP3A" in lead.get("biological_role", "") or "Pharmacokinetic" in lead.get("biological_role", "")
        assert lead.get("suitability_warning") is not None
        assert "Host" in lead.get("target_type", "")

    def test_dual_milestone_accounting_phase2_candidate(self, forecaster):
        """Phase 2 candidate: Horizon A = 5.3y (P3 + Review), Horizon B = 8.5y (P2 + P3 + Review)."""
        cand = {
            "compound_name": "TEST_PHASE2_MOL",
            "chembl_id": "CHEMBL9999",
            "target_symbol": "TARGET_A",
            "clinical_phase": 2.0,
            "indication_phase": 2.0,
            "has_direct_indication": True,
            "evidence_tier": "Tier 2: Indication-Specific Investigational Candidate",
        }
        res = forecaster.forecast_from_evidence(
            emerging_disease_name="Test Syndrome",
            similarity_match={"disease": "Malaria", "jaccard_similarity": 0.50},
            therapeutic_evidence={"status": "EVIDENCE_RETRIEVED", "candidate_count": 1, "top_candidates": [cand]},
        )
        assert res["status"] == "FORECAST_GENERATED"
        ha = res["milestone_accounting"]["horizon_a_continuation"]
        hb = res["milestone_accounting"]["horizon_b_revalidation"]
        assert ha["baseline_duration_years"] == 5.3
        assert ha["phases_remaining"] == ["phase_3", "fda_review"]
        assert hb["baseline_duration_years"] == 8.5
        assert hb["phases_remaining"] == ["phase_2", "phase_3", "fda_review"]

    def test_all_candidates_disqualified_withholds_forecast(self, forecaster):
        """When all retrieved candidates are toxic surrogates, forecast must be withheld."""
        toxic_evidence = {
            "status": "EVIDENCE_RETRIEVED",
            "candidate_count": 1,
            "top_candidates": [
                {
                    "compound_name": "LINDANE",
                    "chembl_id": "CHEMBL15891",
                    "clinical_phase": 4.0,
                    "is_disqualified": True,
                    "is_negative_control_surrogate": True,
                    "evidence_tier": "Tier 4: Non-Therapeutic Surrogate / Toxicant",
                }
            ],
        }
        res = forecaster.forecast_from_evidence(
            emerging_disease_name="Toxic Syndrome",
            similarity_match={"disease": "Malaria", "jaccard_similarity": 0.50},
            therapeutic_evidence=toxic_evidence,
        )
        assert res["status"] == "INSUFFICIENT_EVIDENCE_ALL_CANDIDATES_DISQUALIFIED"
        assert res["can_forecast"] is False
        assert res["estimated_availability_year"] is None

    def test_multi_scenario_progression(self, forecaster):
        candidate = {
            "chembl_id": "CHEMBL100",
            "clinical_phase": 2,
            "compound_name": "PHASE_2_DRUG",
        }
        res = forecaster.forecast_timeline(
            emerging_disease_name="Syndrome",
            closest_disease_name="Dengue",
            symptom_similarity=0.70,
            lead_candidate=candidate,
            current_year=2026,
        )
        scenarios = res["scenarios"]
        base_dur = scenarios["historical_baseline"]["scenario_duration_years"]
        mod_dur = scenarios["moderate_progress"]["scenario_duration_years"]
        agg_dur = scenarios["aggressive_ai_acceleration"]["scenario_duration_years"]
        assert base_dur > mod_dur > agg_dur
        assert (
            scenarios["historical_baseline"]["predicted_year"]
            >= scenarios["moderate_progress"]["predicted_year"]
            >= scenarios["aggressive_ai_acceleration"]["predicted_year"]
        )

    def test_insufficient_evidence_when_similarity_too_low(self, forecaster):
        candidate = {"compound_name": "DRUG_A", "clinical_phase": 4}
        res = forecaster.forecast_timeline(
            emerging_disease_name="Alien Bug",
            closest_disease_name="Flu",
            symptom_similarity=0.08,  # Below minimum threshold 0.15
            lead_candidate=candidate,
        )
        assert res["status"] == "INSUFFICIENT_EVIDENCE_SIMILARITY_TOO_LOW"
        assert res["primary_forecast"] is None
        assert "below the minimum" in res["forecasting_justification"].lower()

    def test_insufficient_evidence_when_no_candidate(self, forecaster):
        res = forecaster.forecast_timeline(
            emerging_disease_name="Unknown Disease",
            closest_disease_name="Unknown Reference",
            symptom_similarity=0.80,
            lead_candidate=None,
        )
        assert res["status"] == "INSUFFICIENT_EVIDENCE_NO_CANDIDATES"
        assert res["primary_forecast"] is None

    def test_no_hardcoded_2077_year(self, forecaster):
        candidate = {"compound_name": "DRUG_A", "clinical_phase": 4}
        res = forecaster.forecast_timeline(
            emerging_disease_name="Syndrome",
            closest_disease_name="Malaria",
            symptom_similarity=0.50,
            lead_candidate=candidate,
            current_year=2026,
        )
        assert res["primary_forecast"]["estimated_year"] != 2077


# ============================================================================
# 4. END-TO-END ORCHESTRATION & INTEGRATION TESTS
# ============================================================================

class TestTheraLensUnifiedOrchestrator:
    def test_end_to_end_malaria_syndrome(self, orchestrator, tmp_path):
        query = "chills, vomiting, high_fever, sweating, headache"
        report = orchestrator.analyze_emerging_disease(
            symptom_input=query,
            disease_name="Test_Febrile_Emergence",
            top_k_diseases=3,
        )
        assert report["status"] == "COMPLETED"
        assert report["symptom_analysis"]["active_symptom_count"] == 5
        
        top_match = report["symptom_analysis"]["matches"][0]
        assert top_match["disease"] == "Malaria"
        assert top_match["jaccard_similarity"] > 0.5
        assert len(top_match["shared_symptoms"]) >= 4
        assert "weighted_jaccard" in top_match

        # Cross-disease evaluations
        assert "cross_disease_evaluations" in report
        assert len(report["cross_disease_evaluations"]) >= 1

        assert len(report["therapeutic_evidence"]) > 0
        malaria_evidence = report["therapeutic_evidence"][0]["therapeutic_evidence"]
        assert malaria_evidence["disease_name"] == "Malaria"
        assert len(malaria_evidence["top_candidates"]) > 0

        # SCIENTIFIC CHECK: Lindane must NOT be the selected lead candidate!
        lead_candidate = report["development_forecast"]["lead_candidate"]
        assert "LINDANE" not in lead_candidate["compound_name"].upper()

        assert report["development_forecast"]["status"] == "FORECAST_GENERATED"
        primary_fc_year = report["development_forecast"]["estimated_availability_year"]
        assert 2028 <= primary_fc_year <= 2045

        json_path, md_path = orchestrator.save_report(report, output_dir=tmp_path)
        assert json_path.exists()
        assert md_path.exists()

        with open(json_path, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["disease_name"] == "Test_Febrile_Emergence"
        assert "research_conclusions" in saved_json

        md_text = md_path.read_text(encoding="utf-8")
        assert "## A. Query Summary" in md_text
        assert "## B. Symptom-Similarity Analysis" in md_text
        assert "## C. Cross-Disease Therapeutic Evidence Evaluation" in md_text
        assert "## D. Candidate Prioritization & Disqualification Audit" in md_text
        assert "## E. Development-History Analysis" in md_text
        assert "## F. Future-Year Forecast & Scenario Simulations" in md_text
        assert "## G. Research Conclusion" in md_text
        assert "MANDATORY SCIENTIFIC DISCLAIMER" in md_text
        assert "Dual-Milestone Horizon Accounting" in md_text
        assert "Ritonavir" in md_text

        # Verify dual-milestone accounting structure in the generated report
        m_acc = report["development_forecast"]["milestone_accounting"]
        assert "horizon_a_continuation" in m_acc
        assert "horizon_b_revalidation" in m_acc
        assert m_acc["horizon_a_continuation"]["baseline_duration_years"] == 8.5
        assert m_acc["horizon_b_revalidation"]["baseline_duration_years"] == 10.6

    def test_end_to_end_alien_symptoms_graceful_handling(self, orchestrator):
        query = "crystallized_scales, telepathic_beaming, purple_aura"
        report = orchestrator.analyze_emerging_disease(
            symptom_input=query,
            disease_name="Alien_Condition",
        )
        assert report["status"] == "COMPLETED"
        assert report["symptom_analysis"]["active_symptom_count"] == 0
        assert len(report["symptom_analysis"]["unrecognized_symptoms"]) == 3
        assert report["development_forecast"]["status"] in [
            "INSUFFICIENT_EVIDENCE_SIMILARITY_TOO_LOW",
            "INSUFFICIENT_EVIDENCE_NO_CANDIDATES",
            "INSUFFICIENT_EVIDENCE_NO_SIMILAR_DISEASE",
        ]
        assert report["development_forecast"]["estimated_availability_year"] is None

