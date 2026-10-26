#!/usr/bin/env python3
"""
TheraLens: Open-World Symptom Similarity Engine

Implements explicit, transparent metric similarity comparison between an arbitrary
emerging disease symptom profile and reference disease symptom profiles.

SCIENTIFIC PRINCIPLES:
1. Metric Similarity != Categorical Classification:
   An emerging disease is, by definition, not an established closed-world class.
   This engine calculates mathematical set-theoretic distance (Jaccard) and vector
   angle (Cosine) against reference disease manifolds, rather than forcing an argmax
   decision boundary.
2. Transparent Feature Decomposition:
   For every candidate match, explicitly reports:
   - Overlapping symptoms (shared clinical features)
   - Symptoms present in query but absent in reference
   - Symptoms present in reference but absent in query
3. Evidence Sufficiency & Negative Reporting:
   If all reference diseases fall below a minimum similarity threshold, the engine
   reports an explicit insufficient-evidence status rather than forcing an invalid match.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "symptom_disease_clean.csv"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.json"

logger = logging.getLogger("SymptomSimilarity")

# Common clinical symptom alias mapping to standard 132 feature vocabulary
COMMON_SYMPTOM_ALIASES: Dict[str, str] = {
    "fever": "high_fever",
    "pyrexia": "high_fever",
    "elevated_temperature": "high_fever",
    "rash": "skin_rash",
    "exanthema": "skin_rash",
    "headache": "headache",
    "cephalalgia": "headache",
    "nausea": "nausea",
    "emesis": "vomiting",
    "throw_up": "vomiting",
    "diarrhea": "diarrhoea",
    "loose_stools": "diarrhoea",
    "joint_aches": "joint_pain",
    "arthralgia": "joint_pain",
    "muscle_aches": "muscle_pain",
    "myalgia": "muscle_pain",
    "body_ache": "muscle_pain",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "lethargy": "lethargy",
    "sneezing": "continuous_sneezing",
    "runny_nose": "runny_nose",
    "stomach_ache": "stomach_pain",
    "abdominal_pain": "abdominal_pain",
    "breathlessness": "breathlessness",
    "shortness_of_breath": "breathlessness",
    "dyspnea": "breathlessness",
    "sweating": "sweating",
    "diaphoresis": "sweating",
    "shivering": "shivering",
    "rigors": "chills",
    "icterus": "yellowish_skin",
    "jaundice": "yellowish_skin",
    "itch": "itching",
    "pruritus": "itching",
}


class SymptomSimilarityEngine:
    """Computes explicit metric similarity between query symptom sets and reference diseases."""

    def __init__(
        self,
        clean_data_path: Path = DATA_CLEAN_PATH,
        feature_names_path: Path = FEATURE_NAMES_PATH,
    ):
        if not clean_data_path.exists():
            raise FileNotFoundError(f"Clean symptom dataset not found at {clean_data_path}")
            
        self.df_clean = pd.read_csv(clean_data_path)
        self.target_col = "prognosis" if "prognosis" in self.df_clean.columns else self.df_clean.columns[-1]
        self.features = [c for c in self.df_clean.columns if c != self.target_col]
        
        # Build normalized feature lookup
        self.feature_lookup: Dict[str, str] = {}
        for feat in self.features:
            norm = self._normalize_token(feat)
            self.feature_lookup[norm] = feat

        # Load reference profiles
        self.reference_profiles, self.reference_frequencies, self.reference_hallmarks = self._build_reference_profiles()
        self.known_diseases = sorted(list(self.reference_profiles.keys()))

    @property
    def feature_names(self) -> List[str]:
        return self.features

    @property
    def disease_profiles(self) -> Dict[str, Dict[str, Any]]:
        return {
            d: {
                "symptoms": sorted(list(s)),
                "frequencies": self.reference_frequencies.get(d, {}),
                "hallmarks": sorted(list(self.reference_hallmarks.get(d, set()))),
            }
            for d, s in self.reference_profiles.items()
        }

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalizes a symptom string to lowercase snake_case with stripped punctuation."""
        t = str(token).lower().strip()
        t = re.sub(r"[\s\-]+", "_", t)
        t = re.sub(r"[^\w_]", "", t)
        return t

    def _build_reference_profiles(
        self, min_consensus_prevalence: float = 0.0
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]], Dict[str, Set[str]]]:
        """Builds consensus symptom sets, symptom frequencies, and hallmark symptoms for each reference disease."""
        profiles: Dict[str, Set[str]] = {}
        frequencies: Dict[str, Dict[str, float]] = {}
        hallmarks: Dict[str, Set[str]] = {}

        grouped = self.df_clean.groupby(self.target_col)
        for disease_raw, group in grouped:
            disease_name = str(disease_raw).strip()
            # Mean symptom occurrence across presentations of this disease
            mean_vals = group[self.features].mean()
            all_active = set(mean_vals[mean_vals > 0.0].index)
            
            if min_consensus_prevalence > 0.0:
                active_symptoms = set(mean_vals[mean_vals >= min_consensus_prevalence].index)
                if not active_symptoms:
                    active_symptoms = all_active
            else:
                active_symptoms = all_active

            profiles[disease_name] = active_symptoms
            
            freq_dict = {feat: float(mean_vals[feat]) for feat in all_active}
            frequencies[disease_name] = freq_dict
            # Hallmark symptoms present in >= 50% of presentations
            hallmarks[disease_name] = set(mean_vals[mean_vals >= 0.50].index)

        return profiles, frequencies, hallmarks

    def parse_symptom_input(
        self,
        symptom_input: Union[List[str], Set[str], Tuple[str, ...], str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Normalizes user symptom inputs, detecting aliases, duplicates, and unrecognized terms."""
        if isinstance(symptom_input, str):
            raw_tokens = [s.strip() for s in symptom_input.split(",") if s.strip()]
        elif isinstance(symptom_input, (list, set, tuple)):
            raw_tokens = [str(s).strip() for s in symptom_input if str(s).strip()]
        elif isinstance(symptom_input, dict):
            raw_tokens = [str(k).strip() for k, v in symptom_input.items() if v]
        else:
            raise TypeError(f"Unsupported symptom input type: {type(symptom_input)}")

        recognized: List[str] = []
        unrecognized: List[str] = []
        duplicates: List[str] = []
        seen: Set[str] = set()

        for token in raw_tokens:
            norm = self._normalize_token(token)
            if not norm:
                continue

            # Check duplicate in current query
            if norm in seen:
                duplicates.append(token)
                continue
            seen.add(norm)

            # 1. Exact match in feature lookup
            if norm in self.feature_lookup:
                canonical = self.feature_lookup[norm]
                recognized.append(canonical)
            # 2. Match in curated alias dictionary
            elif norm in COMMON_SYMPTOM_ALIASES and COMMON_SYMPTOM_ALIASES[norm] in self.features:
                canonical = COMMON_SYMPTOM_ALIASES[norm]
                recognized.append(canonical)
            else:
                # 3. Substring match fallback
                substring_candidates = [
                    canon for k, canon in self.feature_lookup.items()
                    if norm in k or k in norm
                ]
                if substring_candidates:
                    # Pick closest length match
                    chosen = min(substring_candidates, key=lambda x: abs(len(x) - len(norm)))
                    recognized.append(chosen)
                else:
                    unrecognized.append(token)

        # Remove internal duplicate mappings if multiple tokens mapped to same canonical feature
        canonical_unique: List[str] = []
        for feat in recognized:
            if feat not in canonical_unique:
                canonical_unique.append(feat)

        return {
            "raw_input_count": len(raw_tokens),
            "canonical_symptoms": canonical_unique,
            "recognized_count": len(canonical_unique),
            "unrecognized_symptoms": unrecognized,
            "duplicate_inputs": duplicates,
        }

    def compute_similarity(
        self,
        symptom_input: Union[List[str], str, Dict[str, Any]],
        top_k: int = 5,
        min_similarity_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """Calculates explicit Jaccard, Overlap, and Cosine similarity against all reference diseases."""
        parsed = self.parse_symptom_input(symptom_input)
        query_set = set(parsed["canonical_symptoms"])

        if len(query_set) == 0:
            return {
                "status": "EMPTY_QUERY" if parsed["raw_input_count"] == 0 else "NO_SYMPTOMS_MATCHED",
                "query_symptoms": [],
                "recognized_symptoms": [],
                "active_symptom_count": 0,
                "unrecognized_symptoms": parsed["unrecognized_symptoms"],
                "duplicate_inputs": parsed["duplicate_inputs"],
                "total_reference_diseases": len(self.known_diseases),
                "matches": [],
                "top_match": None,
                "message": "Empty query." if parsed["raw_input_count"] == 0 else "No recognized symptoms could be identified from input.",
            }

        # Vector representation for cosine similarity
        query_vector = np.zeros(len(self.features), dtype=float)
        for s in query_set:
            if s in self.features:
                query_vector[self.features.index(s)] = 1.0
        query_norm = np.linalg.norm(query_vector)

        results: List[Dict[str, Any]] = []

        for disease, ref_set in self.reference_profiles.items():
            intersection = query_set.intersection(ref_set)
            union = query_set.union(ref_set)
            
            # Jaccard Similarity: |A n B| / |A u B|
            jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

            # Overlap Coefficient: |A n B| / min(|A|, |B|)
            overlap_denom = min(len(query_set), len(ref_set))
            overlap_coeff = len(intersection) / overlap_denom if overlap_denom > 0 else 0.0

            # Frequency-weighted Jaccard
            freqs = self.reference_frequencies.get(disease, {})
            weighted_num = sum(freqs.get(s, 1.0) for s in intersection)
            weighted_denom = len(query_set) + sum(freqs.get(s, 0.0) for s in (ref_set - query_set))
            weighted_jaccard = (weighted_num / weighted_denom) if weighted_denom > 0 else 0.0

            # Cosine similarity against disease frequency vector
            ref_vector = np.zeros(len(self.features), dtype=float)
            for feat, freq in freqs.items():
                if feat in self.features:
                    ref_vector[self.features.index(feat)] = freq
            ref_norm = np.linalg.norm(ref_vector)

            cosine = 0.0
            if query_norm > 0 and ref_norm > 0:
                cosine = float(np.dot(query_vector, ref_vector) / (query_norm * ref_norm))

            # Feature differences
            query_only = sorted(list(query_set - ref_set))
            reference_only = sorted(list(ref_set - query_set))
            shared = sorted(list(intersection))

            # Prevalence annotations
            shared_detail = [
                {"symptom": s, "prevalence": round(float(freqs.get(s, 1.0)), 2)}
                for s in shared
            ]
            ref_only_detail = [
                {"symptom": s, "prevalence": round(float(freqs.get(s, 0.0)), 2)}
                for s in reference_only
            ]

            hallmark_set = self.reference_hallmarks.get(disease, set())
            matching_hallmarks = sorted(list(intersection.intersection(hallmark_set)))

            results.append({
                "disease": disease,
                "jaccard_similarity": round(float(jaccard), 4),
                "weighted_jaccard": round(float(weighted_jaccard), 4),
                "weighted_jaccard_similarity": round(float(weighted_jaccard), 4),
                "overlap_coefficient": round(float(overlap_coeff), 4),
                "cosine_similarity": round(float(cosine), 4),
                "shared_symptom_count": len(shared),
                "shared_symptoms": shared,
                "shared_symptoms_detail": shared_detail,
                "matching_symptoms": shared,
                "hallmark_symptoms": matching_hallmarks,
                "query_only_symptoms": query_only,
                "differing_query_symptoms": query_only,
                "reference_only_symptoms": reference_only,
                "reference_only_detail": ref_only_detail,
                "differing_reference_symptoms": reference_only,
                "reference_total_symptoms": len(ref_set),
            })

        # Sort descending by Jaccard, then Cosine, then shared count
        results.sort(
            key=lambda x: (x["jaccard_similarity"], x["cosine_similarity"], x["shared_symptom_count"]),
            reverse=True,
        )

        # Filter by threshold
        qualifying_matches = [
            m for m in results if m["jaccard_similarity"] >= min_similarity_threshold
        ]

        if not qualifying_matches:
            status = "INSUFFICIENT_SIMILARITY"
            message = (
                f"No reference disease met the minimum similarity threshold of "
                f"{min_similarity_threshold * 100:.1f}%. Strongest candidate was "
                f"'{results[0]['disease']}' with Jaccard={results[0]['jaccard_similarity']:.4f}."
            )
            top_matches = results[:top_k]
        else:
            status = "SUCCESS"
            message = f"Identified {len(qualifying_matches)} reference diseases meeting similarity threshold."
            top_matches = qualifying_matches[:top_k]

        return {
            "status": status,
            "message": message,
            "query_symptoms": parsed["canonical_symptoms"],
            "recognized_symptoms": parsed["canonical_symptoms"],
            "active_symptom_count": len(parsed["canonical_symptoms"]),
            "unrecognized_symptoms": parsed["unrecognized_symptoms"],
            "duplicate_inputs": parsed["duplicate_inputs"],
            "min_similarity_threshold": min_similarity_threshold,
            "qualifying_count": len(qualifying_matches),
            "total_reference_diseases": len(self.known_diseases),
            "top_match": top_matches[0] if top_matches else None,
            "matches": top_matches,
        }

    def calculate_similarity(
        self,
        symptom_input: Union[List[str], Set[str], Tuple[str, ...], str, Dict[str, Any]],
        top_k: int = 5,
        min_similarity_threshold: float = 0.10,
    ) -> Dict[str, Any]:
        """Convenience alias for compute_similarity."""
        return self.compute_similarity(
            symptom_input=symptom_input,
            top_k=top_k,
            min_similarity_threshold=min_similarity_threshold,
        )
