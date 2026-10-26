#!/usr/bin/env python3
"""
TheraLens: Unified End-to-End Emerging Disease Research & Forecasting Interface

Primary unified entry point executing the complete translational research workflow:
1. Accepts an arbitrary clinical symptom profile for a newly emerging disease.
2. Performs explicit open-world symptom similarity comparison against reference disease manifolds.
3. Reports transparent feature overlap, query-only symptoms, and reference-only symptoms.
4. Dynamically retrieves therapeutic evidence (biological targets, candidate compounds, bioactivities, clinical phases).
5. Assesses development maturity and projects multi-scenario medication availability windows.
6. Emits an explainable research dossier (terminal printout, JSON audit log, Markdown report).

MANDATORY SCIENTIFIC WARNING:
TheraLens is an exploratory research support tool for translational computational biology.
It does NOT diagnose patients, prescribe medication, or establish clinical efficacy.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.symptom_similarity import SymptomSimilarityEngine
from src.models.therapeutic_evidence_retriever import TherapeuticEvidenceRetriever
from src.analysis.emerging_disease_forecaster import EmergingDiseaseForecaster
from src.inference.predict_disease import DiseasePredictor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TheraLensUnified")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANDATORY_DISCLAIMER = (
    "RESEARCH DEMONSTRATION & TRANSLATIONAL EXPLORATION ONLY:\n"
    "TheraLens is an exploratory computational research-support tool developed for academic evaluation. "
    "Symptom similarity does NOT establish disease identity or shared etiology. "
    "Therapeutic target associations and candidate prioritization scores do NOT establish clinical efficacy, "
    "safety, or regulatory approval. Projected milestone decision windows are multi-scenario research estimates, "
    "NOT guaranteed drug approval dates. Output is NOT intended for clinical diagnosis, prescribing, or patient care."
)


class TheraLensUnifiedOrchestrator:
    """Orchestrates the complete end-to-end emerging disease research pipeline."""

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing TheraLens Unified Subsystems...")
        self.similarity_engine = SymptomSimilarityEngine()
        self.evidence_retriever = TherapeuticEvidenceRetriever()
        self.forecaster = EmergingDiseaseForecaster(base_year=2026)

        # Optional supplementary classifier
        try:
            self.classifier = DiseasePredictor()
            self.classifier_available = True
        except Exception as e:
            logger.warning("Classifier initialization skipped: %s", e)
            self.classifier = None
            self.classifier_available = False

    def analyze_emerging_disease(
        self,
        symptom_input: str,
        disease_name: str = "Emerging_Pathogen_X",
        top_k_diseases: int = 3,
        min_similarity: float = 0.10,
        research_notes: str = "",
        include_classifier: bool = True,
    ) -> Dict[str, Any]:
        """Convenience alias for execute_workflow."""
        return self.execute_workflow(
            symptoms=symptom_input,
            emerging_disease_name=disease_name,
            top_k=top_k_diseases,
            min_similarity=min_similarity,
            research_notes=research_notes,
            include_classifier=include_classifier,
        )

    def execute_workflow(
        self,
        symptoms: str,
        emerging_disease_name: str = "Emerging_Pathogen_X",
        top_k: int = 3,
        min_similarity: float = 0.10,
        research_notes: str = "",
        include_classifier: bool = True,
    ) -> Dict[str, Any]:
        """Executes the complete connected pipeline from symptoms to future-year forecast."""
        query_id = f"TL_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.datetime.now().isoformat()

        logger.info("Executing TheraLens workflow for query '%s' (%s)...", query_id, emerging_disease_name)

        # ---------------------------------------------------------
        # Stage 1: Symptom Similarity Analysis (Open-World Metric)
        # ---------------------------------------------------------
        similarity_res = self.similarity_engine.compute_similarity(
            symptom_input=symptoms,
            top_k=top_k,
            min_similarity_threshold=min_similarity,
        )

        # Supplementary Classifier Signal (Closed-World Benchmark Check)
        classifier_output = None
        if include_classifier and self.classifier_available:
            try:
                classifier_output = self.classifier.predict(symptoms, top_k=top_k)
            except Exception as e:
                logger.warning("Supplementary classifier failed: %s", e)

        # ---------------------------------------------------------
        # Stage 2: Therapeutic Evidence Retrieval for Similar Diseases
        # ---------------------------------------------------------
        ranked_matches = similarity_res.get("matches", [])
        enriched_evidence: List[Dict[str, Any]] = []

        if ranked_matches:
            enriched_evidence = self.evidence_retriever.retrieve_evidence_for_similarity_matches(
                similarity_matches=ranked_matches,
                top_candidates_per_disease=3,
            )

        # ---------------------------------------------------------
        # Stage 3: Cross-Disease Evidence Evaluation & Milestone Forecasting
        # ---------------------------------------------------------
        forecast_result: Optional[Dict[str, Any]] = None
        cluster_evaluations: List[Dict[str, Any]] = []

        if ranked_matches and enriched_evidence:
            for item in enriched_evidence:
                s_ctx = item["similarity_context"]
                t_ev = item["therapeutic_evidence"]
                dis_name = t_ev["disease_name"]
                cands = t_ev.get("top_candidates", [])

                viable = [
                    c for c in cands
                    if not c.get("is_disqualified", False)
                    and not c.get("is_negative_control_surrogate", False)
                    and not str(c.get("evidence_tier", "")).startswith("Tier 4")
                ]
                disqualified = [
                    c for c in cands
                    if c.get("is_disqualified", False)
                    or c.get("is_negative_control_surrogate", False)
                    or str(c.get("evidence_tier", "")).startswith("Tier 4")
                ]
                tier1 = [c for c in viable if str(c.get("evidence_tier", "")).startswith("Tier 1")]
                tier2 = [c for c in viable if str(c.get("evidence_tier", "")).startswith("Tier 2")]
                tier3 = [c for c in viable if str(c.get("evidence_tier", "")).startswith("Tier 3")]

                fc_dis = self.forecaster.forecast_from_evidence(
                    emerging_disease_name=emerging_disease_name,
                    similarity_match=s_ctx,
                    therapeutic_evidence=t_ev,
                    min_similarity_threshold=min_similarity,
                )

                cluster_evaluations.append({
                    "disease": dis_name,
                    "jaccard_similarity": s_ctx.get("jaccard_similarity", 0.0),
                    "cosine_similarity": s_ctx.get("cosine_similarity", 0.0),
                    "weighted_jaccard": s_ctx.get("weighted_jaccard", 0.0),
                    "target_count": t_ev.get("target_count", 0),
                    "candidate_count": len(cands),
                    "viable_candidate_count": len(viable),
                    "tier1_count": len(tier1),
                    "tier2_count": len(tier2),
                    "tier3_count": len(tier3),
                    "disqualified_count": len(disqualified),
                    "lead_viable_candidate": viable[0] if viable else None,
                    "disqualified_candidates": [c.get("compound_name") for c in disqualified],
                    "forecast": fc_dis,
                })

            # SCIENTIFIC SELECTION: Evaluate evidence across the similar disease cluster
            top_eval = cluster_evaluations[0]
            if top_eval["viable_candidate_count"] > 0:
                selected_eval = top_eval
                selection_rationale = (
                    f"Selected highest symptom match '{selected_eval['disease']}' (Jaccard = {selected_eval['jaccard_similarity']:.4f}). "
                    f"Evaluated {selected_eval['viable_candidate_count']} viable candidate(s)."
                )
            else:
                # Top match has NO viable candidates (e.g. all candidates disqualified as toxicants or 0 found)
                # Prioritize closest similar disease with viable leads
                viable_alt = next((e for e in cluster_evaluations if e["viable_candidate_count"] > 0), None)
                if viable_alt is not None:
                    selected_eval = viable_alt
                    selection_rationale = (
                        f"Top symptom match '{top_eval['disease']}' lacked viable clinical leads "
                        f"(all candidates disqualified as toxicants or surrogates). "
                        f"Prioritized closest similar disease with viable leads: '{selected_eval['disease']}' "
                        f"(Jaccard = {selected_eval['jaccard_similarity']:.4f})."
                    )
            cluster_summary = [
                {k: v for k, v in ce.items() if k != "forecast"}
                for ce in cluster_evaluations
            ]
            forecast_result = dict(selected_eval["forecast"])
            forecast_result["selected_reference_disease"] = selected_eval["disease"]
            forecast_result["selection_rationale"] = selection_rationale
            forecast_result["cluster_summary"] = cluster_summary
        else:
            forecast_result = {
                "status": "INSUFFICIENT_EVIDENCE_NO_SIMILAR_DISEASE",
                "can_forecast": False,
                "emerging_disease": emerging_disease_name,
                "reference_disease": None,
                "symptom_similarity": 0.0,
                "estimated_availability_year": None,
                "plausible_range": None,
                "primary_reason": "No reference diseases met the required symptom similarity threshold.",
                "data_limitations": ["Input symptoms lacked sufficient overlap with reference disease profiles."],
                "scenarios": {},
            }
            cluster_summary = []

        # ---------------------------------------------------------
        # Stage 4: Package Complete Research Dossier
        # ---------------------------------------------------------
        dossier: Dict[str, Any] = {
            "status": "COMPLETED",
            "disease_name": emerging_disease_name,
            "query_metadata": {
                "query_id": query_id,
                "timestamp": timestamp,
                "emerging_disease_name": emerging_disease_name,
                "raw_symptom_input": symptoms,
                "research_notes": research_notes,
            },
            "symptom_analysis": similarity_res,
            "supplementary_classifier_signal": classifier_output,
            "therapeutic_evidence": enriched_evidence,
            "cross_disease_evaluations": cluster_summary,
            "development_forecast": forecast_result,
            "future_year_forecast": forecast_result,
            "research_conclusions": {
                "strongest_matches": [
                    {"disease": m["disease"], "jaccard": m["jaccard_similarity"]}
                    for m in similarity_res.get("matches", [])[:3]
                ],
                "forecast_status": forecast_result.get("status"),
                "estimated_year": forecast_result.get("estimated_availability_year"),
                "plausible_range": forecast_result.get("plausible_range"),
                "illustrative_sensitivity_range": forecast_result.get("illustrative_sensitivity_range"),
                "forecast_methodology": forecast_result.get("forecast_methodology"),
            },
            "disclaimer": MANDATORY_DISCLAIMER,
        }

        # ---------------------------------------------------------
        # Stage 5: Persist Outputs (JSON & Markdown)
        # ---------------------------------------------------------
        json_path, md_path = self.save_report(dossier)
        dossier["saved_artifacts"] = {
            "json_report": str(json_path),
            "markdown_report": str(md_path),
        }

        return dossier

    def save_report(
        self, dossier: Dict[str, Any], output_dir: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """Persists the research dossier as structured JSON and readable Markdown."""
        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        query_id = dossier.get("query_metadata", {}).get("query_id", f"TL_{uuid.uuid4().hex[:6]}")

        json_path = target_dir / f"{query_id}_forecast_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(dossier, f, indent=2)

        md_path = target_dir / f"{query_id}_forecast_report.md"
        self._write_markdown_report(dossier, md_path)

        dossier["saved_artifacts"] = {
            "json_report": str(json_path),
            "markdown_report": str(md_path),
        }
        return json_path, md_path

    def print_console_summary(self, dossier: Dict[str, Any]):
        """Prints a human-readable console summary of the dossier."""
        return print_cli_summary(dossier)

    def _write_markdown_report(self, dossier: Dict[str, Any], file_path: Path):
        """Generates a comprehensive markdown research report matching Sections A through G."""
        meta = dossier["query_metadata"]
        sym = dossier["symptom_analysis"]
        clf = dossier.get("supplementary_classifier_signal")
        ev_list = dossier["therapeutic_evidence"]
        cluster_evals = dossier.get("cross_disease_evaluations", [])
        fc = dossier["future_year_forecast"]

        lines: List[str] = [
            f"# TheraLens Research Dossier: {meta['emerging_disease_name']}",
            f"**Query ID:** `{meta['query_id']}` | **Date:** {meta['timestamp']}",
            "",
            "## A. Query Summary",
            f"- **Emerging Disease Identified**: `{meta['emerging_disease_name']}`",
            f"- **Input Symptoms Reported**: `{meta['raw_symptom_input']}`",
            f"- **Recognized Standard Features ({len(sym.get('query_symptoms', []))})**: {', '.join(sym.get('query_symptoms', [])) if sym.get('query_symptoms') else 'None'}",
            f"- **Unrecognized Terms**: {', '.join(sym.get('unrecognized_symptoms', [])) if sym.get('unrecognized_symptoms') else 'None'}",
            f"- **Duplicate Terms Ignored**: {', '.join(sym.get('duplicate_inputs', [])) if sym.get('duplicate_inputs') else 'None'}",
            f"- **Data Representation**: Binary feature vector mapped against standard 132-symptom vocabulary.",
            f"- **Researcher Notes**: {meta['research_notes'] or 'None provided.'}",
            "",
            "---",
            "",
            "## B. Symptom-Similarity Analysis",
            f"**Similarity Status**: `{sym.get('status')}` ({sym.get('message')})",
            "**Methodology**: Explicit open-world Jaccard similarity J(A,B), frequency-weighted Jaccard, and Cosine similarity against reference disease symptom manifolds.",
            "",
            "| Rank | Reference Disease | Jaccard Similarity | Weighted Jaccard | Cosine Similarity | Shared Symptoms (with Reference Prevalence) | Differing Reference Symptoms | Differing Query Symptoms |",
            "| :---: | :--- | :---: | :---: | :---: | :--- | :--- | :--- |",
        ]

        for i, m in enumerate(sym.get("matches", []), 1):
            if m.get("shared_symptoms_detail"):
                shared_str = ", ".join([f"{d['symptom']} ({int(d.get('prevalence', d.get('frequency', 1.0))*100)}%)" for d in m["shared_symptoms_detail"]])
            else:
                shared_str = ", ".join(m["shared_symptoms"]) if m["shared_symptoms"] else "None"
            ref_only_str = ", ".join(m["reference_only_symptoms"][:4]) + ("..." if len(m["reference_only_symptoms"]) > 4 else "")
            q_only_str = ", ".join(m["query_only_symptoms"]) if m["query_only_symptoms"] else "None"
            w_jacc = m.get("weighted_jaccard", m["jaccard_similarity"])
            lines.append(
                f"| {i} | **{m['disease']}** | **{m['jaccard_similarity']:.4f}** | {w_jacc:.4f} | {m['cosine_similarity']:.4f} | {shared_str} | {ref_only_str} | {q_only_str} |"
            )

        if clf and clf.get("status") == "SUCCESS":
            lines.extend([
                "",
                "### Supplementary Closed-World Classifier Diagnosis (Baseline Reference)",
                f"> **Model Predicted Label**: `{clf.get('predicted_disease')}` (Softmax margin: {clf.get('confidence_score', 0):.4f})",
                "> *Notice: Closed-world classification forces input into one of 41 known classes. It does not measure open-world distance and must not be confused with true biological identity.*",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## C. Cross-Disease Therapeutic Evidence Evaluation",
            "> **Scientific Principle**: Evidence is evaluated across all similar reference diseases rather than naively relying solely on the single top symptom match.",
            "",
        ])

        if cluster_evals:
            lines.extend([
                "### Multi-Disease Candidate Synthesis Table",
                "| Disease | Symptom Jaccard | Targets | Viable Leads | Tier 1 (Approved) | Tier 2 (Investigational) | Tier 3 (Indirect) | Tier 4 (Disqualified) | Primary Viable Lead |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ])
            for ce in cluster_evals:
                lead_str = ce["lead_viable_candidate"]["compound_name"] if ce.get("lead_viable_candidate") else "None (All Disqualified)"
                lines.append(
                    f"| **{ce['disease']}** | {ce['jaccard_similarity']:.4f} | {ce['target_count']} | **{ce['viable_candidate_count']}** | {ce['tier1_count']} | {ce['tier2_count']} | {ce['tier3_count']} | {ce['disqualified_count']} | {lead_str} |"
                )
            lines.append("")

        if not ev_list:
            lines.append("No therapeutic evidence retrieved for matching diseases.")
        else:
            for item in ev_list:
                s_ctx = item["similarity_context"]
                t_ev = item["therapeutic_evidence"]
                dis_name = t_ev["disease_name"]
                jacc = s_ctx["jaccard_similarity"]

                lines.extend([
                    f"### Detailed Evidence for Reference Disease: {dis_name} (Symptom Similarity: {jacc:.4f})",
                    f"- **Clinical Pipeline Maturity**: {t_ev.get('regulatory_maturity')}",
                    f"- **Biological Drug Targets Documented**: {t_ev.get('target_count', 0)}",
                    f"- **Small-Molecule Candidates Ranked**: {t_ev.get('candidate_count', 0)}",
                    "",
                    "#### Associated Biological Targets (Open Targets Platform):",
                ])

                for t in t_ev.get("top_targets", [])[:3]:
                    lines.append(f"- **{t['target_symbol']}** ({t['target_name']}) — Association Score: `{t['association_score']:.4f}`")

                lines.extend([
                    "",
                    f"#### Ranked Candidates for {dis_name}:",
                    "| Rank | Compound Name | ChEMBL ID | Target & Class | Biological Role | Indication Phase | Evidence Tier | CPS | Status | Suitability & Rationale |",
                    "| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
                ])

                has_ritonavir = False
                for c in t_ev.get("top_candidates", []):
                    disq_label = "⚠️ DISQUALIFIED" if c.get("is_disqualified") or c.get("disqualified_from_repurposing") else "Eligible"
                    tier_str = c.get("evidence_tier", "Unclassified")
                    ind_ph_str = f"Phase {int(c['indication_phase'])}" if c.get("indication_phase") is not None else "None"
                    t_desc = f"{c['target_symbol']} ({c.get('target_type', 'Host Target')})"
                    bio_role = c.get("biological_role", "Exploratory Lead")
                    relevance_str = c.get("clinical_context", c.get("therapeutic_relevance", ""))
                    if "RITONAVIR" in str(c.get("compound_name", "")).upper():
                        has_ritonavir = True
                    lines.append(
                        f"| {c['rank']} | **{c['compound_name']}** | `{c['chembl_id']}` | {t_desc} | {bio_role} | {ind_ph_str} | {tier_str} | **{c['composite_priority_score']:.4f}** | {disq_label} | {relevance_str} |"
                    )
                lines.append("")

                if has_ritonavir:
                    lines.extend([
                        "> ⚠️ **Critical Biological & Target Suitability Notice (Ritonavir / CYP3A5)**:",
                        "> Ritonavir (`CHEMBL163`) is an FDA-approved HIV protease inhibitor and potent CYP3A inhibitor. In malaria, historical Phase 1 trials evaluated Ritonavir exclusively as a **pharmacokinetic enhancer (adjuvant booster)** to elevate plasma concentrations of co-administered partner antimalarials, **NOT as a direct standalone antiparasitic curative agent**. It interacts with human host metabolizing enzymes (`CYP3A5`, `SLCO1B1`), not pathogen proteins. Projecting clinical development as a standalone curative monotherapy for an emerging syndrome carries severe biological limitations.",
                        "",
                    ])

        lines.extend([
            "---",
            "",
            "## D. Candidate Prioritization & Disqualification Audit",
            "> **Scientific Restraint Notice**: A high Composite Priority Score (CPS) or molecule-level approval in unrelated conditions does NOT qualify a compound as an approved treatment for a new indication.",
            "- **Tier 1 (Approved Indication Drug)**: Compound has documented regulatory approvals specifically indicated for the reference disease.",
            "- **Tier 2 (Investigational Indication Candidate)**: Compound has documented clinical trials specifically for the reference disease.",
            "- **Tier 3 (Indirect Mechanistic Target Lead)**: Compound exhibits in-vitro potency against a shared protein target, but lacks clinical trials for this disease. Cannot bypass early-stage development.",
            "- **Tier 4 (Disqualified Surrogate / Toxicant)**: Molecule is an insecticide, toxicant, or assay negative-control (e.g., Lindane). Disqualified from human clinical repurposing.",
            "",
            "---",
            "",
            "## E. Development-History Analysis",
            "- **Observational Units**: Biopharmaceutical phase benchmarks derived from Tufts CSDD / DiMasi et al. and Hay et al.",
            "- **Standard Sequential Durations**: Preclinical: 4.5y | Phase 1: 2.1y | Phase 2: 3.2y | Phase 3: 3.8y | FDA NDA/BLA Review: 1.5y.",
            "- **Historical Phase Transition Success Rates**: Preclinical->P1: 40.0% | P1->P2: 59.5% | P2->P3: 35.1% | P3->NDA: 58.6% | NDA->Approval: 90.1%.",
            "",
            "---",
            "",
            "## F. Future-Year Forecast & Scenario Simulations",
        ])

        if not fc or not fc.get("can_forecast"):
            lines.extend([
                f"### Status: ⚠️ `{fc.get('status', 'CANNOT_FORECAST')}`",
                f"> **Scientific Restraint Notice**: {fc.get('primary_reason', 'Insufficient evidence to project availability.')}",
                "",
                "**Identified Evidentiary Limitations**:",
            ])
            for lim in fc.get("data_limitations", []):
                lines.append(f"- {lim}")
        else:
            lead = fc.get("lead_candidate", {})
            m_acc = fc.get("milestone_accounting", {})
            ha = m_acc.get("horizon_a_continuation", {})
            hb = m_acc.get("horizon_b_revalidation", {})

            lines.extend([
                f"- **Forecasting Methodology**: `{fc.get('forecast_methodology', 'Scenario Simulation')}`",
                f"- **Evaluated Reference Disease**: `{fc['reference_disease']}` (Symptom Similarity: {fc['symptom_similarity']:.4f})",
                f"- **Lead Investigated Candidate**: `{lead.get('compound_name')}` (Target: `{lead.get('target_symbol')}`, Evidence Tier: `{lead.get('evidence_tier')}`)",
                f"- **Biological Target Type**: `{lead.get('target_type', 'Host Target')}`",
                f"- **Biological & Pharmacological Role**: `{lead.get('biological_role', 'Exploratory Candidate')}`",
                f"- **Compound Suitability**: `{lead.get('compound_suitability', 'Unspecified')}`",
                f"- **Development Pathway**: **{fc['pathway_type']}**",
                f"- **Clinical Pathway Explanation**: {fc['pathway_explanation']}",
                f"- **Therapeutic Transferability Assessment**: {fc.get('therapeutic_transferability_assessment')}",
                "",
            ])

            if lead.get("suitability_warning"):
                lines.extend([
                    f"> ⚠️ **Candidate Suitability Notice**: {lead['suitability_warning']}",
                    "",
                ])

            if m_acc and ha and hb:
                lines.extend([
                    "### Dual-Milestone Horizon Accounting (Addressing Phase 1 Ambiguity)",
                    "> **Evidentiary Interpretation Notice**:",
                    f"> Static registry records from ChEMBL indicate a historical Phase {int(m_acc.get('recorded_indication_phase', 1))} trial record for this indication. ",
                    "> However, static database archives do **not** verify whether trials are currently active, terminated, or completed with positive clinical efficacy. ",
                    "> To resolve whether recorded Phase 1 constitutes completed maturity or remaining work, TheraLens presents two formal milestone horizons:",
                    "",
                    "| Milestone Horizon | Core R&D Assumption | Phases Remaining | Baseline Duration | Benchmark Pipeline Success | Estimated Milestone Year | Illustrative Sensitivity Window |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
                    f"| **Horizon A: Direct Phase Advancement** | {ha.get('core_assumption', '')} | `{', '.join(ha.get('phases_remaining', []))}` | **{ha.get('baseline_duration_years')} yrs** | {ha.get('historical_benchmark_success_rate_pct')}% | **{ha.get('estimated_milestone_year')}** | **[{ha.get('illustrative_sensitivity_window', [0, 0])[0]} – {ha.get('illustrative_sensitivity_window', [0, 0])[1]}]** |",
                    f"| **Horizon B: Clinical Re-Validation / In-Progress** | {hb.get('core_assumption', '')} | `{', '.join(hb.get('phases_remaining', []))}` | **{hb.get('baseline_duration_years')} yrs** | {hb.get('historical_benchmark_success_rate_pct')}% | **{hb.get('estimated_milestone_year')}** | **[{hb.get('illustrative_sensitivity_window', [0, 0])[0]} – {hb.get('illustrative_sensitivity_window', [0, 0])[1]}]** |",
                    "",
                    "### Technological Progress Scenario Simulations (Horizon B: Clinical Re-Validation Baseline)",
                    "> *Note: These scenarios represent sensitivity analysis based on predefined technological advancement assumptions, NOT empirical machine-learning predictions.*",
                    "",
                    "| Scenario | Speedup | Projected Duration | Estimated Milestone Year | Illustrative Sensitivity Window | R&D Assumptions |",
                    "| :--- | :---: | :---: | :---: | :---: | :--- |",
                ])

                for sc_id, sc in hb.get("scenarios", {}).items():
                    w_str = f"{sc['plausible_window'][0]} – {sc['plausible_window'][1]}"
                    lines.append(
                        f"| **{sc['scenario_name']}** | {sc['speedup_pct']}% | {sc['projected_duration_years']} yrs | **{sc['estimated_milestone_year']}** | **[{w_str}]** | {sc['scenario_description']} |"
                    )

                lines.extend([
                    "",
                    "### Technological Progress Scenario Simulations (Horizon A: Direct Phase Advancement)",
                    "",
                    "| Scenario | Speedup | Projected Duration | Estimated Milestone Year | Illustrative Sensitivity Window | R&D Assumptions |",
                    "| :--- | :---: | :---: | :---: | :---: | :--- |",
                ])

                for sc_id, sc in ha.get("scenarios", {}).items():
                    w_str = f"{sc['plausible_window'][0]} – {sc['plausible_window'][1]}"
                    lines.append(
                        f"| **{sc['scenario_name']}** | {sc['speedup_pct']}% | {sc['projected_duration_years']} yrs | **{sc['estimated_milestone_year']}** | **[{w_str}]** | {sc['scenario_description']} |"
                    )
            else:
                lines.extend([
                    f"- **Phases to Complete**: `{', '.join(fc['phases_to_complete'])}` (Baseline Duration: {fc['baseline_duration_years']} years)",
                    f"- **Historical Benchmark Pipeline Success Rate**: `{fc.get('historical_benchmark_pipeline_success_rate_pct')}%`",
                    "",
                    "### Technological Progress Scenario Simulations",
                    "| Scenario | Speedup | Projected Duration | Estimated Milestone Year | Illustrative Sensitivity Window | R&D Assumptions |",
                    "| :--- | :---: | :---: | :---: | :---: | :--- |",
                ])
                for sc_id, sc in fc.get("scenarios", {}).items():
                    w_str = f"{sc['plausible_window'][0]} – {sc['plausible_window'][1]}"
                    lines.append(
                        f"| **{sc['scenario_name']}** | {sc['speedup_pct']}% | {sc['projected_duration_years']} yrs | **{sc['estimated_milestone_year']}** | **[{w_str}]** | {sc['scenario_description']} |"
                    )

            lines.extend([
                "",
                "**Critical Methodological Limitations**:",
            ])
            for lim in fc.get("data_limitations", []):
                lines.append(f"- {lim}")

        lines.extend([
            "",
            "---",
            "",
            "## G. Research Conclusion",
            "MANDATORY SCIENTIFIC DISCLAIMER:",
            "> TheraLens is an exploratory research-support system for computational biology. Symptom similarity does not establish disease identity, and therapeutic associations do not establish treatment efficacy. Forecasts are scenario simulations based on literature benchmarks, not guarantees of future drug discovery, approval, or availability. The results are not intended for clinical diagnosis or treatment decisions.",
        ])

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def print_cli_summary(dossier: Dict[str, Any]):
    """Prints a beautiful, structured terminal output of the research dossier."""
    meta = dossier["query_metadata"]
    sym = dossier["symptom_analysis"]
    clf = dossier.get("supplementary_classifier_signal")
    ev_list = dossier["therapeutic_evidence"]
    cluster_evals = dossier.get("cross_disease_evaluations", [])
    fc = dossier["future_year_forecast"]

    print("\n" + "=" * 80)
    print("               THERALENS: EMERGING DISEASE FORECASTING DOSSIER")
    print("=" * 80)
    print(f"Query ID:             {meta['query_id']}")
    print(f"Emerging Disease:     {meta['emerging_disease_name']}")
    print(f"Input Symptoms:       {meta['raw_symptom_input']}")
    print(f"Recognized Features:  {len(sym.get('query_symptoms', []))} features: {sym.get('query_symptoms', [])}")
    if sym.get("unrecognized_symptoms"):
        print(f"Unrecognized Terms:   {sym.get('unrecognized_symptoms')} (Not in 132-symptom vocabulary)")
    print("-" * 80)

    print("\n[A] OPEN-WORLD SYMPTOM-SIMILARITY RANKING:")
    if not sym.get("matches"):
        print("  [!] No reference disease met the similarity threshold.")
    else:
        for i, m in enumerate(sym.get("matches", []), 1):
            w_jacc = m.get("weighted_jaccard", m["jaccard_similarity"])
            print(f"  {i}. {m['disease']:<26} Jaccard: {m['jaccard_similarity']:.4f} | Weighted: {w_jacc:.4f} | Cosine: {m['cosine_similarity']:.4f}")
            if m.get("hallmark_symptoms"):
                print(f"     |-- Hallmark Matches (>=50%): {m['hallmark_symptoms']}")
            print(f"     |-- Shared Features:          {m['shared_symptoms']}")
            if m["query_only_symptoms"]:
                print(f"     |-- Query-Only (Unique):      {m['query_only_symptoms']}")
            if m["reference_only_symptoms"]:
                sample_ref = m["reference_only_symptoms"][:3]
                suffix = "..." if len(m["reference_only_symptoms"]) > 3 else ""
                print(f"     |-- Reference-Only (Missing): {sample_ref}{suffix}")

    if clf and clf.get("status") == "SUCCESS":
        print(f"\n  [Supplementary Classifier Check]: Predicted Label = '{clf['predicted_disease']}' (Softmax margin: {clf['confidence_score']:.4f})")
        print("  *(Note: Classifier forces closed-world argmax; metric similarity above reflects open-world distance.)*")

    print("\n" + "-" * 80)
    print("[B] CROSS-DISEASE THERAPEUTIC EVIDENCE EVALUATION:")
    if cluster_evals:
        for ce in cluster_evals:
            lead_name = ce["lead_viable_candidate"]["compound_name"] if ce.get("lead_viable_candidate") else "None (All Disqualified)"
            disq_str = f" | Disqualified: {ce['disqualified_count']} {ce['disqualified_candidates']}" if ce['disqualified_count'] > 0 else ""
            print(f"  * {ce['disease']:<22} (Jaccard: {ce['jaccard_similarity']:.4f}) -> Viable Leads: {ce['viable_candidate_count']} (Tier 1: {ce['tier1_count']}, Tier 2: {ce['tier2_count']}, Tier 3: {ce['tier3_count']}){disq_str}")
            print(f"    |-- Primary Viable Lead: {lead_name}")

    print("\n" + "-" * 80)
    print("[C] PROSPECTIVE MEDICATION AVAILABILITY FORECAST:")
    if not fc or not fc.get("can_forecast"):
        print(f"  [!] FORECAST WITHHELD: {fc.get('status')}")
        print(f"  Reason: {fc.get('primary_reason')}")
    else:
        lead = fc["lead_candidate"]
        print(f"  Selected Reference:    '{fc['reference_disease']}' (Jaccard: {fc['symptom_similarity']:.4f})")
        print(f"  Lead Candidate:        '{lead['compound_name']}' ({lead['target_symbol']}) — {lead.get('evidence_tier', 'Unclassified')}")
        if lead.get("biological_role"):
            print(f"  Biological Role:       {lead['biological_role']}")
        if lead.get("target_type"):
            print(f"  Target Classification: {lead['target_type']}")
        if lead.get("suitability_warning"):
            print(f"  Suitability Notice:    {lead['suitability_warning']}")
        print(f"  Development Pathway:   {fc['pathway_type']}")

        m_acc = fc.get("milestone_accounting", {})
        if m_acc and "horizon_a_continuation" in m_acc and "horizon_b_revalidation" in m_acc:
            ha = m_acc["horizon_a_continuation"]
            hb = m_acc["horizon_b_revalidation"]
            print(f"\n  DUAL-MILESTONE HORIZON ACCOUNTING (Milestone & Status Accounting):")
            print(f"  Registry Trial Status: {m_acc.get('clinical_trial_status', 'Indeterminate')}")
            print(f"  * Horizon A (Direct Phase Advancement / Optimistic Continuation):")
            print(f"    - Remaining Milestones: {', '.join(ha['phases_remaining'])} (Baseline: {ha['baseline_duration_years']} yrs)")
            print(f"    - Estimated Milestone:  Year {ha['estimated_milestone_year']} (Spread: {ha['illustrative_sensitivity_window'][0]}-{ha['illustrative_sensitivity_window'][1]} | Transition benchmark: {ha['historical_benchmark_success_rate_pct']}%)")
            print(f"  * Horizon B (Clinical Re-Validation / Conservative In-Progress):")
            print(f"    - Remaining Milestones: {', '.join(hb['phases_remaining'])} (Baseline: {hb['baseline_duration_years']} yrs)")
            print(f"    - Estimated Milestone:  Year {hb['estimated_milestone_year']} (Spread: {hb['illustrative_sensitivity_window'][0]}-{hb['illustrative_sensitivity_window'][1]} | Transition benchmark: {hb['historical_benchmark_success_rate_pct']}%)")

            print(f"\n  Technological Sensitivity Scenarios (Horizon B Conservative Baseline):")
            for sc_id, sc in hb.get("scenarios", {}).items():
                w = sc["plausible_window"]
                print(f"    * {sc['scenario_name']:<48}: Year {sc['estimated_milestone_year']} (Spread: {w[0]}-{w[1]})")
            print(f"\n  Technological Sensitivity Scenarios (Horizon A Direct Advancement):")
            for sc_id, sc in ha.get("scenarios", {}).items():
                w = sc["plausible_window"]
                print(f"    * {sc['scenario_name']:<48}: Year {sc['estimated_milestone_year']} (Spread: {w[0]}-{w[1]})")
        else:
            print(f"  Phases Remaining:      {', '.join(fc['phases_to_complete'])} (Baseline duration: {fc['baseline_duration_years']} yrs)")
            print(f"  Historical Benchmark:  {fc.get('historical_benchmark_pipeline_success_rate_pct', fc.get('cumulative_pipeline_success_rate_pct'))}% transition success (sequential trial benchmark)")
            print(f"\n  Multi-Scenario Availability Horizons (Illustrative Sensitivity Ranges):")
            for sc_id, sc in fc.get("scenarios", {}).items():
                w = sc["plausible_window"]
                print(f"    * {sc['scenario_name']:<48}: Year {sc['estimated_milestone_year']} (Spread: {w[0]}-{w[1]})")

    print("\n" + "-" * 80)
    print("SAVED AUDIT ARTIFACTS:")
    if "saved_artifacts" in dossier:
        print(f"  - Structured JSON:   {dossier['saved_artifacts']['json_report']}")
        print(f"  - Markdown Dossier:  {dossier['saved_artifacts']['markdown_report']}")
    print("=" * 80)
    print(f"DISCLAIMER:\n{MANDATORY_DISCLAIMER}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TheraLens: Unified Emerging Disease Symptom Similarity, Evidence Linkage & Availability Forecasting"
    )
    parser.add_argument(
        "--symptoms",
        type=str,
        default="chills, vomiting, high_fever, sweating, headache",
        help="Comma-separated list of observed clinical symptoms (e.g. 'fever, cough, chills')",
    )
    parser.add_argument(
        "--disease_name",
        type=str,
        default="Novel_Febrile_Syndrome_2026",
        help="Name or identifier of the newly emerging disease",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of similar reference diseases to evaluate",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.10,
        help="Minimum Jaccard similarity threshold for evidence linkage",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional researcher notes or epidemiological context",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save structured JSON and Markdown research reports",
    )
    parser.add_argument(
        "--no_classifier",
        action="store_true",
        help="Disable supplementary closed-world classifier diagnosis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    orchestrator = TheraLensUnifiedOrchestrator(output_dir=Path(args.output_dir))
    dossier = orchestrator.execute_workflow(
        symptoms=args.symptoms,
        emerging_disease_name=args.disease_name,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        research_notes=args.notes,
        include_classifier=not args.no_classifier,
    )
    print_cli_summary(dossier)
