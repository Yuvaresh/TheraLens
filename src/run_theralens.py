#!/usr/bin/env python3
"""
TheraLens: End-to-End Orchestration Workflow

Master pipeline execution script unifying all core project modules:
1. Symptom-based disease classifier training & held-out evaluation
2. Multi-criteria therapeutic candidate & target prioritization
3. Longitudinal epidemiological trend analysis & time-series forecasting
4. Regulatory therapeutic timeline audit & evidence classification
5. Experimental biopharmaceutical development milestone forecasting
6. Model explainability (permutation feature importance & score decomposition)

Supports modular execution via CLI flags, graceful degradation when large datasets
are skipped, structured logging, and comprehensive summary reporting.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_symptom_classifier import train_and_benchmark_pipeline
from src.models.evaluate_symptom_classifier import evaluate_persisted_classifier
from src.models.prioritize_therapeutic_candidates import run_therapeutic_prioritization_pipeline
from src.analysis.analyze_epidemiological_trends import run_epidemiological_analysis_pipeline
from src.analysis.validate_therapeutic_timeline import run_timeline_validation_pipeline
from src.analysis.forecast_therapeutic_timeline import generate_experimental_forecasts
from src.analysis.explain_model import run_explainability_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RunTheraLens")


def validate_environment() -> Dict[str, bool]:
    """Validates existence of required datasets and paths."""
    checks = {
        "symptom_disease": (PROJECT_ROOT / "data" / "raw" / "symptom_disease.csv").exists(),
        "measles_who": (PROJECT_ROOT / "data" / "raw" / "measles_who.csv").exists(),
        "yellow_fever_who": (PROJECT_ROOT / "data" / "raw" / "yellow_fever_who.csv").exists(),
        "therapeutic_assoc": (PROJECT_ROOT / "data" / "raw" / "therapeutic_associations.csv").exists(),
        "chembl_activity": (PROJECT_ROOT / "data" / "raw" / "chembl_activity.csv").exists(),
        "chembl_molecules": (PROJECT_ROOT / "data" / "raw" / "chembl_molecules.csv").exists(),
        "fda_timeline": (PROJECT_ROOT / "data" / "processed" / "fda_drug_approval_timeline.csv").exists(),
        "fda_audit": (PROJECT_ROOT / "data" / "processed" / "fda_chembl_match_audit.csv").exists(),
        "master_timeline": (PROJECT_ROOT / "data" / "processed" / "therapeutic_development_timeline.csv").exists(),
    }
    return checks


def parse_args():
    parser = argparse.ArgumentParser(
        description="TheraLens: Applied Machine Learning Biomedical Pipeline Orchestrator"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=True,
        help="Execute all pipeline modules (default: True)",
    )
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Train and benchmark symptom-based disease classifier",
    )
    parser.add_argument(
        "--prioritize",
        action="store_true",
        help="Execute therapeutic target and compound candidate prioritization",
    )
    parser.add_argument(
        "--epidemiology",
        action="store_true",
        help="Execute longitudinal epidemiological trend analysis and forecasting",
    )
    parser.add_argument(
        "--validate-timeline",
        action="store_true",
        help="Validate FDA approvals and ChEMBL indications timeline",
    )
    parser.add_argument(
        "--forecast-timeline",
        action="store_true",
        help="Execute experimental biopharma milestone forecast prototype",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate model explainability charts and permutation importances",
    )
    parser.add_argument(
        "--skip-large-timeline",
        action="store_true",
        default=False,
        help="Skip processing large (114 MB) therapeutic development timeline if desired",
    )
    return parser.parse_args()


def run_pipeline() -> Dict[str, Any]:
    args = parse_args()
    specific_flags = [
        args.train_classifier,
        args.prioritize,
        args.epidemiology,
        args.validate_timeline,
        args.forecast_timeline,
        args.explain,
    ]
    run_all = not any(specific_flags)

    print("\n" + "=" * 75)
    print("       TheraLens Applied Machine Learning Pipeline Orchestrator")
    print("=" * 75)
    logger.info("Initializing TheraLens execution pipeline...")
    
    env_checks = validate_environment()
    logger.info("Input Dataset Integrity Status:")
    for name, exists in env_checks.items():
        logger.info("  - %-22s: %s", name, "PRESENT" if exists else "MISSING")

    summary = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modules_executed": {},
        "modules_skipped": {},
        "artifacts_generated": [],
    }

    # 1. Symptom Classifier Training & Evaluation
    if run_all or args.train_classifier:
        if env_checks["symptom_disease"]:
            t0 = time.time()
            logger.info("\n>>> [1/6] Executing Symptom Classifier Pipeline...")
            try:
                meta = train_and_benchmark_pipeline()
                eval_res = evaluate_persisted_classifier()
                duration = round(time.time() - t0, 2)
                summary["modules_executed"]["symptom_classifier"] = {
                    "status": "SUCCESS",
                    "best_model": meta["model_name"],
                    "accuracy": meta["evaluation_metrics"]["accuracy"],
                    "macro_f1": meta["evaluation_metrics"]["macro_f1"],
                    "reloaded_verified": eval_res["model_reloaded"],
                    "elapsed_sec": duration,
                }
                summary["artifacts_generated"].extend([
                    "models/symptom_classifier.joblib",
                    "models/symptom_classifier_metadata.json",
                    "outputs/model_evaluation/model_comparison.csv",
                    "outputs/model_evaluation/confusion_matrix.png",
                ])
            except Exception as e:
                logger.error("Error in Symptom Classifier: %s", str(e), exc_info=True)
                summary["modules_executed"]["symptom_classifier"] = {"status": "ERROR", "error": str(e)}
        else:
            logger.warning("Skipping Symptom Classifier: symptom_disease.csv missing.")
            summary["modules_skipped"]["symptom_classifier"] = "symptom_disease.csv not found"

    # 2. Therapeutic Candidate Prioritization
    if run_all or args.prioritize:
        if env_checks["therapeutic_assoc"] and env_checks["chembl_activity"] and env_checks["chembl_molecules"]:
            t0 = time.time()
            logger.info("\n>>> [2/6] Executing Therapeutic Prioritization Pipeline...")
            try:
                t_df, c_df, audit = run_therapeutic_prioritization_pipeline()
                duration = round(time.time() - t0, 2)
                summary["modules_executed"]["therapeutic_prioritization"] = {
                    "status": "SUCCESS",
                    "ranked_targets_count": len(t_df),
                    "ranked_candidates_count": len(c_df),
                    "top_candidate": f"{c_df.iloc[0]['local_disease']} -> {c_df.iloc[0]['pref_name']} (CPS={c_df.iloc[0]['composite_priority_score']})",
                    "elapsed_sec": duration,
                }
                summary["artifacts_generated"].extend([
                    "data/processed/disease_target_rankings.csv",
                    "data/processed/therapeutic_candidate_rankings.csv",
                    "outputs/therapeutic_prioritization/top_candidate_scores.png",
                    "outputs/therapeutic_prioritization/score_component_breakdown.png",
                ])
            except Exception as e:
                logger.error("Error in Therapeutic Prioritization: %s", str(e), exc_info=True)
                summary["modules_executed"]["therapeutic_prioritization"] = {"status": "ERROR", "error": str(e)}
        else:
            logger.warning("Skipping Therapeutic Prioritization: ChEMBL or Open Targets data missing.")
            summary["modules_skipped"]["therapeutic_prioritization"] = "Open Targets or ChEMBL raw files missing"

    # 3. Epidemiological Trend Analysis
    if run_all or args.epidemiology:
        if env_checks["measles_who"] and env_checks["yellow_fever_who"]:
            t0 = time.time()
            logger.info("\n>>> [3/6] Executing Epidemiological Analysis Pipeline...")
            try:
                epi_df, epi_summary = run_epidemiological_analysis_pipeline()
                duration = round(time.time() - t0, 2)
                summary["modules_executed"]["epidemiological_analysis"] = {
                    "status": "SUCCESS",
                    "total_records": epi_summary["total_records"],
                    "surges_detected": epi_summary["detected_outbreak_surges"],
                    "forecasting_period": epi_summary["forecasting_benchmarks"].get("evaluation_period", ""),
                    "elapsed_sec": duration,
                }
                summary["artifacts_generated"].extend([
                    "data/processed/epidemiological_trends.csv",
                    "outputs/epidemiological_analysis/global_incidence_trends.png",
                    "outputs/epidemiological_analysis/forecasting_model_comparison.png",
                ])
            except Exception as e:
                logger.error("Error in Epidemiological Analysis: %s", str(e), exc_info=True)
                summary["modules_executed"]["epidemiological_analysis"] = {"status": "ERROR", "error": str(e)}
        else:
            logger.warning("Skipping Epidemiological Analysis: WHO surveillance datasets missing.")
            summary["modules_skipped"]["epidemiological_analysis"] = "WHO surveillance files missing"

    # 4. Therapeutic Timeline Validation
    if run_all or args.validate_timeline:
        if args.skip_large_timeline:
            logger.info("Skipping master timeline processing (--skip-large-timeline enabled).")
            summary["modules_skipped"]["therapeutic_timeline"] = "Skipped per user --skip-large-timeline flag"
        elif env_checks["fda_timeline"] and env_checks["master_timeline"]:
            t0 = time.time()
            logger.info("\n>>> [4/6] Executing Therapeutic Timeline Validation...")
            try:
                tl_audit = run_timeline_validation_pipeline()
                duration = round(time.time() - t0, 2)
                summary["modules_executed"]["timeline_validation"] = {
                    "status": "SUCCESS",
                    "total_timeline_rows": tl_audit["master_timeline_audit"]["total_timeline_records"],
                    "unique_disease_molecule_milestones": tl_audit["master_timeline_audit"]["unique_disease_molecule_milestone_pairs"],
                    "verified_phase4_milestones": tl_audit["master_timeline_audit"]["evidence_classification"]["verified_phase4_approved_indication_records"],
                    "elapsed_sec": duration,
                }
                summary["artifacts_generated"].extend([
                    "outputs/timeline_analysis/timeline_quality_report.txt",
                    "outputs/timeline_analysis/mapping_audit_summary.json",
                    "outputs/timeline_analysis/timeline_evidence_classification.png",
                ])
            except Exception as e:
                logger.error("Error in Timeline Validation: %s", str(e), exc_info=True)
                summary["modules_executed"]["timeline_validation"] = {"status": "ERROR", "error": str(e)}
        else:
            logger.warning("Skipping Timeline Validation: FDA or master timeline dataset missing.")
            summary["modules_skipped"]["therapeutic_timeline"] = "FDA timeline or master timeline file missing"

    # 5. Experimental Milestone Forecasting Prototype
    if run_all or args.forecast_timeline:
        t0 = time.time()
        logger.info("\n>>> [5/6] Executing Experimental Therapeutic Milestone Forecasting...")
        try:
            fc_df, fc_summary = generate_experimental_forecasts()
            duration = round(time.time() - t0, 2)
            summary["modules_executed"]["experimental_forecast"] = {
                "status": "SUCCESS",
                "diseases_assessed": fc_summary["total_diseases_assessed"],
                "median_forecast_horizon_years": fc_summary["median_forecast_horizon_years"],
                "elapsed_sec": duration,
            }
            summary["artifacts_generated"].extend([
                "outputs/timeline_analysis/experimental_forecast_projections.csv",
                "outputs/timeline_analysis/forecast_pipeline_latencies.png",
            ])
        except Exception as e:
            logger.error("Error in Milestone Forecasting: %s", str(e), exc_info=True)
            summary["modules_executed"]["experimental_forecast"] = {"status": "ERROR", "error": str(e)}

    # 6. Model Explainability Pipeline
    if run_all or args.explain:
        t0 = time.time()
        logger.info("\n>>> [6/6] Executing Model Explainability Pipeline...")
        try:
            exp_res = run_explainability_pipeline()
            duration = round(time.time() - t0, 2)
            summary["modules_executed"]["explainability"] = {
                "status": "SUCCESS",
                "top_influential_symptom": exp_res["top_5_influential_symptoms"][0]["symptom_feature"],
                "elapsed_sec": duration,
            }
            summary["artifacts_generated"].extend([
                "outputs/explainability/symptom_permutation_importance.csv",
                "outputs/explainability/top_symptoms_importance.png",
                "outputs/explainability/candidate_score_decomposition.csv",
                "outputs/explainability/candidate_contribution_breakdown.png",
            ])
        except Exception as e:
            logger.error("Error in Explainability: %s", str(e), exc_info=True)
            summary["modules_executed"]["explainability"] = {"status": "ERROR", "error": str(e)}

    # Persist summary
    summary_path = PROJECT_ROOT / "outputs" / "run_pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("                     PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    for mod, details in summary["modules_executed"].items():
        print(f"  [+] {mod:<28}: {details.get('status')} ({details.get('elapsed_sec', 0):.2f}s)")
    if summary["modules_skipped"]:
        for mod, reason in summary["modules_skipped"].items():
            print(f"  [-] {mod:<28}: SKIPPED ({reason})")
    print("-" * 75)
    print(f"Total Master Artifacts Tracked: {len(summary['artifacts_generated'])}")
    print(f"Master Summary File Saved:      {summary_path}")
    print("=" * 75 + "\n")
    return summary


if __name__ == "__main__":
    run_pipeline()
