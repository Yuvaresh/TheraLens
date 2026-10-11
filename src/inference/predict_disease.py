#!/usr/bin/env python3
"""
TheraLens: Symptom-Based Disease Prediction Inference Interface

Provides a reusable prediction pipeline to map presented clinical symptoms to
disease probabilities and candidate diagnoses.

IMPORTANT DISCLAIMER:
This interface is strictly an academic research demonstration and engineering
prototype developed for college review evaluation. It does NOT constitute medical
advice, clinical diagnosis, or therapeutic instruction.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "symptom_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "symptom_classifier_metadata.json"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.json"
CLASS_LABELS_PATH = PROJECT_ROOT / "models" / "class_labels.json"

DISCLAIMER_NOTICE = (
    "RESEARCH DEMONSTRATION ONLY: This output is produced by an academic machine learning "
    "model as part of the TheraLens research project. It does NOT constitute clinical "
    "diagnosis, medical guidance, or healthcare advice. Consult licensed healthcare "
    "professionals for medical diagnosis and care."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PredictDisease")


class DiseasePredictor:
    """Reusable inference interface for symptom-based disease classification."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_path: Path = FEATURE_NAMES_PATH,
        labels_path: Path = CLASS_LABELS_PATH,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train model first.")
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature names file not found at {feature_path}.")
            
        self.model = joblib.load(model_path)
        with open(feature_path, "r", encoding="utf-8") as f:
            self.feature_names = json.load(f)
            
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
        else:
            self.classes = list(getattr(self.model, "classes_", []))
            
        # Map normalized lowercase symptom names for robust user lookup
        self.feature_lookup = {
            feat.lower().replace(" ", "_"): feat for feat in self.feature_names
        }

    def vectorize_symptoms(self, active_symptoms: Union[List[str], str, Dict[str, int]]) -> np.ndarray:
        """Converts symptom inputs into aligned binary feature vector."""
        vector = np.zeros(len(self.feature_names), dtype=int)
        matched_symptoms = []
        unmatched_symptoms = []

        if isinstance(active_symptoms, str):
            symptom_list = [s.strip() for s in active_symptoms.split(",") if s.strip()]
        elif isinstance(active_symptoms, list):
            symptom_list = active_symptoms
        elif isinstance(active_symptoms, dict):
            symptom_list = [k for k, v in active_symptoms.items() if v]
        else:
            raise TypeError(f"Unsupported symptom input type: {type(active_symptoms)}")

        for item in symptom_list:
            key = str(item).lower().strip().replace(" ", "_").replace("-", "_")
            if key in self.feature_lookup:
                orig_feat = self.feature_lookup[key]
                idx = self.feature_names.index(orig_feat)
                vector[idx] = 1
                matched_symptoms.append(orig_feat)
            else:
                # Substring/partial match attempt
                partial_matches = [
                    orig for k, orig in self.feature_lookup.items() if key in k or k in key
                ]
                if partial_matches:
                    chosen = partial_matches[0]
                    idx = self.feature_names.index(chosen)
                    vector[idx] = 1
                    matched_symptoms.append(chosen)
                else:
                    unmatched_symptoms.append(item)

        return vector, matched_symptoms, unmatched_symptoms

    def predict(
        self,
        active_symptoms: Union[List[str], str, Dict[str, int]],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Predicts disease class with top-k candidates and model decision confidence."""
        vector, matched, unmatched = self.vectorize_symptoms(active_symptoms)
        
        if vector.sum() == 0:
            return {
                "status": "NO_SYMPTOMS_MATCHED",
                "predicted_disease": None,
                "confidence_score": 0.0,
                "confidence_type": "None",
                "top_candidates": [],
                "matched_symptoms": [],
                "unmatched_symptoms": unmatched,
                "disclaimer": DISCLAIMER_NOTICE,
            }

        X_input = vector.reshape(1, -1)
        pred_disease = self.model.predict(X_input)[0]

        # Check probability support
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_input)[0]
            confidence_type = "Model Softmax Score (Uncalibrated)"
        elif hasattr(self.model, "decision_function"):
            df_vals = self.model.decision_function(X_input)[0]
            # Softmax normalized decision values
            exp_vals = np.exp(df_vals - np.max(df_vals))
            probs = exp_vals / exp_vals.sum()
            confidence_type = "Normalized Decision Function Margin"
        else:
            probs = np.zeros(len(self.classes))
            if pred_disease in self.classes:
                probs[self.classes.index(pred_disease)] = 1.0
            confidence_type = "Binary Decision Output"

        # Sort top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_candidates = [
            {
                "disease": self.classes[idx],
                "confidence_score": round(float(probs[idx]), 4),
            }
            for idx in top_indices
        ]

        primary_confidence = float(top_candidates[0]["confidence_score"]) if top_candidates else 1.0

        return {
            "status": "SUCCESS",
            "predicted_disease": pred_disease,
            "confidence_score": primary_confidence,
            "confidence_type": confidence_type,
            "top_candidates": top_candidates,
            "matched_symptoms": matched,
            "unmatched_symptoms": unmatched,
            "active_symptom_count": int(vector.sum()),
            "disclaimer": DISCLAIMER_NOTICE,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="TheraLens Symptom-Based Disease Prediction")
    parser.add_argument(
        "--symptoms",
        type=str,
        default="itching,skin_rash,nodal_skin_eruptions",
        help="Comma-separated list of symptom keywords (e.g., 'chills,cough,high_fever')",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top candidate diagnoses to return",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = DiseasePredictor()
    result = predictor.predict(args.symptoms, top_k=args.top_k)
    
    print("\n" + "=" * 65)
    print("  TheraLens Symptom Prediction Interface (Research Prototype)")
    print("=" * 65)
    print(f"Active Symptoms Input:  {args.symptoms}")
    print(f"Matched Features ({result['active_symptom_count']}):  {result['matched_symptoms']}")
    if result["unmatched_symptoms"]:
        print(f"Unmatched Inputs:       {result['unmatched_symptoms']}")
    print("-" * 65)
    print(f"PREDICTED DISEASE:      {result['predicted_disease']}")
    print(f"CONFIDENCE SCORE:       {result['confidence_score']:.4f} ({result['confidence_type']})")
    print("\nTop Candidate Differential Diagnoses:")
    for i, c in enumerate(result["top_candidates"], 1):
        print(f"  {i}. {c['disease']:<32} Score: {c['confidence_score']:.4f}")
    print("-" * 65)
    print(f"DISCLAIMER:\n{result['disclaimer']}")
    print("=" * 65 + "\n")
