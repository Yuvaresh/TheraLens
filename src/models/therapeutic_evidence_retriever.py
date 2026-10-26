#!/usr/bin/env python3
"""
TheraLens: Therapeutic Evidence Retrieval & Entity Linkage Engine

Links reference diseases identified by symptom similarity to:
1. Genomic disease-target associations (Open Targets)
2. Bioactive small molecules & in-vitro potency measurements (ChEMBL)
3. Physicochemical drug-likeness compliance (Lipinski / Veber)
4. Clinical development status & regulatory approval milestones (Drugs@FDA)

SCIENTIFIC PRINCIPLES:
1. Heuristic Prioritization != Clinical Proof of Efficacy:
   Composite Priority Scores (CPS) are in-silico hypothesis-generating ranks.
   They do NOT establish safety, efficacy, or approval for an emerging disease.
2. Indication Specificity vs. Indirect Target Association:
   Distinguishes compounds with disease-specific clinical investigation from
   those linked indirectly through shared protein targets.
3. Negative Control Awareness:
   Explicitly annotates potential surrogate / non-etiological associations
   (such as Medroxyprogesterone Acetate / AIDS / PGR).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CANDIDATES_PATH = DATA_PROCESSED_DIR / "therapeutic_candidate_rankings.csv"
TARGETS_PATH = DATA_PROCESSED_DIR / "disease_target_rankings.csv"
FDA_TIMELINE_PATH = DATA_PROCESSED_DIR / "fda_drug_approval_timeline.csv"

logger = logging.getLogger("TherapeuticEvidence")

# Canonical disease name mapping across heterogeneous dataset schemas
DISEASE_NAME_NORMALIZATION: Dict[str, str] = {
    "diabetes": "Diabetes",
    "diabetes ": "Diabetes",
    "fungal infection": "Fungal Infection",
    "gerd": "GERD",
    "chronic cholestasis": "Chronic Cholestasis",
    "drug reaction": "Drug Reaction",
    "peptic ulcer diseae": "Peptic Ulcer Disease",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    "aids": "AIDS",
    "gastroenteritis": "Gastroenteritis",
    "bronchial asthma": "Bronchial Asthma",
    "hypertension": "Hypertension",
    "migraine": "Migraine",
    "cervical spondylosis": "Cervical Spondylosis",
    "paralysis (brain hemorrhage)": "Paralysis (brain hemorrhage)",
    "jaundice": "Jaundice",
    "malaria": "Malaria",
    "chicken pox": "Chickenpox",
    "chickenpox": "Chickenpox",
    "dengue": "Dengue",
    "typhoid": "Typhoid",
    "hepatitis a": "Hepatitis A",
    "hepatitis b": "Hepatitis B",
    "hepatitis c": "Hepatitis C",
    "hepatitis d": "Hepatitis D",
    "hepatitis e": "Hepatitis E",
    "alcoholic hepatitis": "Alcoholic Hepatitis",
    "tuberculosis": "Tuberculosis",
    "common cold": "Common Cold",
    "pneumonia": "Pneumonia",
    "dimorphic hemmorhoids(piles)": "Dimorphic Hemmorhoids (piles)",
    "dimorphic hemmorhoids (piles)": "Dimorphic Hemmorhoids (piles)",
    "heart attack": "Heart Attack",
    "varicose veins": "Varicose Veins",
    "hypothyroidism": "Hypothyroidism",
    "hyperthyroidism": "Hyperthyroidism",
    "hypoglycemia": "Hypoglycemia",
    "osteoarthritis": "Osteoarthritis",
    "arthritis": "Arthritis",
    "(vertigo) paroymsal  positional vertigo": "Vertigo",
    "vertigo": "Vertigo",
    "acne": "Acne",
    "urinary tract infection": "Urinary Tract Infection",
    "psoriasis": "Psoriasis",
    "impetigo": "Impetigo",
}


INDICATIONS_PATH = DATA_PROCESSED_DIR / "chembl_drug_indications.csv"

# Curated registry of known non-therapeutic screen compounds, pesticides, toxicants, and negative controls
KNOWN_NON_THERAPEUTIC_OR_SURROGATES: Dict[str, str] = {
    "CHEMBL15891": "Organochlorine insecticide / pesticide (Lindane); GABA-A receptor blocker. Not an anti-infective or antimalarial.",
    "LINDANE": "Organochlorine insecticide / pesticide (Lindane); GABA-A receptor blocker. Not an anti-infective or antimalarial.",
    "MEDROXYPROGESTERONE": "Progestin steroid hormone; progesterone receptor (PGR) ligand for contraception. Not an antiretroviral for AIDS.",
    "MEDROXYPROGESTERONE ACETATE": "Progestin steroid hormone; progesterone receptor (PGR) ligand for contraception. Not an antiretroviral for AIDS.",
    "ROTENONE": "Mitochondrial complex I pesticide / toxicant. Non-therapeutic screen artifact.",
    "PARAQUAT": "Herbicide toxicant. Non-therapeutic screen artifact.",
    "DDT": "Organochlorine pesticide. Non-therapeutic screen artifact.",
}

# Curated registry of human host targets (drug-metabolizing enzymes and receptors)
# Distinguished from pathogen-encoded targets to identify pharmacokinetic boosters vs curative agents
HOST_METABOLIC_TARGETS: Set[str] = {
    "CYP3A5", "CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2", "SLCO1B1", "ABCB11", "ABCB1", "HMGCR"
}
HOST_RECEPTOR_TARGETS: Set[str] = {
    "ADRA1D", "CACNA1S", "THRB", "PGR", "GABRA1", "ALOX5", "HDAC7", "CHRM1", "HTR2A"
}


class TherapeuticEvidenceRetriever:
    """Retrieves and structures therapeutic evidence for diseases identified by symptom similarity."""

    def __init__(
        self,
        candidates_path: Path = CANDIDATES_PATH,
        targets_path: Path = TARGETS_PATH,
        fda_path: Path = FDA_TIMELINE_PATH,
        indications_path: Path = INDICATIONS_PATH,
    ):
        if not candidates_path.exists():
            raise FileNotFoundError(f"Candidate rankings not found at {candidates_path}")
        if not targets_path.exists():
            raise FileNotFoundError(f"Target rankings not found at {targets_path}")

        logger.info("Loading therapeutic candidate, target, and indication datasets...")
        self.df_candidates = pd.read_csv(candidates_path)
        self.df_targets = pd.read_csv(targets_path)
        self.df_fda = pd.read_csv(fda_path) if fda_path.exists() else pd.DataFrame()
        self.df_indications = pd.read_csv(indications_path) if indications_path.exists() else pd.DataFrame()

        # Build disease index
        self.available_diseases = set(self.df_candidates["local_disease"].str.strip().unique())
        self.available_target_diseases = set(self.df_targets["local_disease"].str.strip().unique())

        # Build indication lookup index: molecule_chembl_id -> list of {indication, max_phase_for_ind}
        self.indication_lookup: Dict[str, List[Dict[str, Any]]] = {}
        if not self.df_indications.empty and "molecule_chembl_id" in self.df_indications.columns:
            for _, r in self.df_indications.iterrows():
                mid = str(r["molecule_chembl_id"]).strip()
                ind_term = str(r.get("indication", "")).strip().lower()
                phase_ind = float(r.get("max_phase_for_ind", 0.0)) if pd.notna(r.get("max_phase_for_ind")) else 0.0
                if mid not in self.indication_lookup:
                    self.indication_lookup[mid] = []
                self.indication_lookup[mid].append({
                    "indication": ind_term,
                    "max_phase_for_ind": phase_ind,
                })

    @property
    def candidate_diseases(self) -> Set[str]:
        return {d.lower() for d in self.available_diseases}

    def normalize_disease_name(self, disease_name: str) -> str:
        """Maps varying disease names to the canonical key used in therapeutic rankings."""
        clean = str(disease_name).strip().lower()
        if clean in DISEASE_NAME_NORMALIZATION:
            return DISEASE_NAME_NORMALIZATION[clean]
        return str(disease_name).strip().title()

    def _get_indication_evidence(
        self, molecule_chembl_id: str, canonical_disease: str
    ) -> Tuple[bool, Optional[float], str]:
        """Checks whether the molecule has verified indication-specific records for this disease."""
        records = self.indication_lookup.get(molecule_chembl_id, [])
        if not records:
            return False, None, "No documented indication records in repository."

        disease_clean = canonical_disease.lower().strip()
        matching_phases: List[float] = []
        matching_terms: List[str] = []

        for rec in records:
            ind = rec["indication"]
            if disease_clean in ind or ind in disease_clean:
                matching_phases.append(rec["max_phase_for_ind"])
                matching_terms.append(ind)
            elif disease_clean == "aids" and ("hiv" in ind or "acquired immunodeficiency" in ind):
                matching_phases.append(rec["max_phase_for_ind"])
                matching_terms.append(ind)
            elif disease_clean == "diabetes" and "diabetes" in ind:
                matching_phases.append(rec["max_phase_for_ind"])
                matching_terms.append(ind)
            elif disease_clean == "hypertension" and "hypertension" in ind:
                matching_phases.append(rec["max_phase_for_ind"])
                matching_terms.append(ind)
            elif disease_clean == "bronchial asthma" and "asthma" in ind:
                matching_phases.append(rec["max_phase_for_ind"])
                matching_terms.append(ind)

        if matching_phases:
            best_phase = max(matching_phases)
            term_str = ", ".join(sorted(list(set(matching_terms))))
            return True, best_phase, f"Verified indication: {term_str} (Max Phase {best_phase})"

        return False, None, f"Documented for other indications ({len(records)}), but none match '{canonical_disease}'."

    def _get_indication_record_details(
        self, molecule_chembl_id: str, canonical_disease: str
    ) -> Dict[str, Any]:
        """Provides provenance, recorded clinical phase, and registry limitations for indication records."""
        has_direct, best_phase, explanation = self._get_indication_evidence(molecule_chembl_id, canonical_disease)
        records = self.indication_lookup.get(molecule_chembl_id, [])
        disease_clean = canonical_disease.lower().strip()
        matched_terms = []
        for rec in records:
            ind = rec["indication"]
            if (
                disease_clean in ind
                or ind in disease_clean
                or (disease_clean == "aids" and ("hiv" in ind or "acquired immunodeficiency" in ind))
                or (disease_clean == "diabetes" and "diabetes" in ind)
                or (disease_clean == "hypertension" and "hypertension" in ind)
                or (disease_clean == "bronchial asthma" and "asthma" in ind)
            ):
                matched_terms.append(ind)

        if has_direct:
            return {
                "source_database": "ChEMBL Indication Registry (chembl_drug_indications.csv)",
                "has_direct_indication": True,
                "recorded_indication_terms": sorted(list(set(matched_terms))),
                "recorded_max_phase": best_phase,
                "registry_status": "Static Historical Archive (Active trial recruitment & outcome unverified)",
                "active_trial_verified": False,
                "data_limitations": (
                    "Indication records establish that clinical trials reached the recorded phase historically. "
                    "Static registry records do not establish whether trials are actively ongoing, completed with positive results, "
                    "or discontinued, nor do they guarantee pipeline continuation for newly emerging syndromes."
                ),
            }
        else:
            return {
                "source_database": "ChEMBL Indication Registry",
                "has_direct_indication": False,
                "recorded_indication_terms": [],
                "recorded_max_phase": None,
                "registry_status": "No disease-specific indication documented in repository",
                "active_trial_verified": False,
                "data_limitations": f"No documented clinical trials or approvals exist for '{canonical_disease}'.",
            }

    def retrieve_evidence_for_disease(
        self,
        disease_name: str,
        top_targets: int = 5,
        top_candidates: int = 5,
    ) -> Dict[str, Any]:
        """Retrieves structured targets, candidates, bioactivity, and clinical phase evidence."""
        canonical_disease = self.normalize_disease_name(disease_name)

        has_candidates = canonical_disease in self.available_diseases
        has_targets = canonical_disease in self.available_target_diseases

        if not has_candidates and not has_targets:
            return {
                "disease_name": canonical_disease,
                "status": "NO_EVIDENCE_FOUND",
                "message": f"No therapeutic candidates or targets found for '{canonical_disease}' in repository datasets.",
                "missing_reasons": [f"No records found for '{canonical_disease}' in Open Targets or ChEMBL."],
                "target_count": 0,
                "candidate_count": 0,
                "top_targets": [],
                "top_candidates": [],
                "max_clinical_phase": None,
                "regulatory_maturity": "Unknown (No Data)",
            }

        # 1. Biological Targets with Host vs Pathogen Classification
        sub_targets = self.df_targets[self.df_targets["local_disease"].str.strip() == canonical_disease].copy()
        sub_targets = sub_targets.sort_values(by="association_score", ascending=False).head(top_targets)
        
        target_list: List[Dict[str, Any]] = []
        for _, r in sub_targets.iterrows():
            tsym = str(r["target_symbol"]).strip()
            if tsym in HOST_METABOLIC_TARGETS:
                ttype = "Host Drug-Metabolizing Enzyme / Transporter"
                tclass = "Host Pharmacokinetic Target"
            elif tsym in HOST_RECEPTOR_TARGETS:
                ttype = "Host Physiological Receptor / Channel"
                tclass = "Host Physiological / Symptomatic Target"
            else:
                ttype = "Host Genomic Factor (Open Targets)"
                tclass = "Host Genetic Association Target"

            target_list.append({
                "target_symbol": tsym,
                "target_name": r.get("target_name", ""),
                "target_id": r.get("target_id", ""),
                "association_score": round(float(r["association_score"]), 4),
                "rank_within_disease": int(r.get("target_rank_within_disease", 1)),
                "target_type": ttype,
                "target_class": tclass,
                "is_pathogen_target": False,
                "target_biological_context": (
                    f"Human host target '{tsym}' genomically linked to '{canonical_disease}' in Open Targets. "
                    "Reflects host biology, drug metabolism, or tissue response, NOT a pathogen-encoded target."
                ),
            })

        # 2. Chemical Candidates with Rigorous Indication & Surrogate Validation
        sub_cands = self.df_candidates[self.df_candidates["local_disease"].str.strip() == canonical_disease].copy()
        raw_cands: List[Dict[str, Any]] = []

        for _, r in sub_cands.iterrows():
            mid = str(r["molecule_chembl_id"]).strip()
            cname = str(r.get("pref_name", mid)).strip() if pd.notna(r.get("pref_name")) else mid
            global_phase = float(r["max_phase"]) if pd.notna(r["max_phase"]) else 0.0
            cps = round(float(r["composite_priority_score"]), 4)

            # Check known toxicant / surrogate registry
            is_surrogate = False
            surrogate_reason = ""
            if mid in KNOWN_NON_THERAPEUTIC_OR_SURROGATES:
                is_surrogate = True
                surrogate_reason = KNOWN_NON_THERAPEUTIC_OR_SURROGATES[mid]
            elif cname.upper() in KNOWN_NON_THERAPEUTIC_OR_SURROGATES:
                is_surrogate = True
                surrogate_reason = KNOWN_NON_THERAPEUTIC_OR_SURROGATES[cname.upper()]
            elif canonical_disease.upper() == "AIDS" and "MEDROXYPROGESTERONE" in cname.upper():
                is_surrogate = True
                surrogate_reason = "Progestin steroid hormone for contraception; non-antiretroviral PGR ligand."

            # Check indication-level clinical evidence
            has_direct_ind, ind_phase, ind_explanation = self._get_indication_evidence(mid, canonical_disease)
            rec_details = self._get_indication_record_details(mid, canonical_disease)
            tsym = str(r["target_symbol"]).strip()

            if tsym in HOST_METABOLIC_TARGETS:
                cand_target_type = "Host Drug-Metabolizing Enzyme / Transporter"
                cand_target_relevance = "Host Pharmacokinetic Target (Modulates drug exposure; non-pathogen)"
            elif tsym in HOST_RECEPTOR_TARGETS:
                cand_target_type = "Host Physiological Receptor / Channel"
                cand_target_relevance = "Host Receptor Target (Modulates host signalling; non-pathogen)"
            else:
                cand_target_type = "Host Genomic Factor (Open Targets)"
                cand_target_relevance = "Host Genetic Association Target (Non-pathogen)"

            if mid == "CHEMBL163" or cname.upper() == "RITONAVIR":
                bio_role = "Pharmacokinetic Enhancer / CYP3A Booster"
                comp_suitability = "Adjuvant / Pharmacokinetic Enhancer (Non-Curative Monotherapy)"
                cand_relevance_desc = (
                    "Ritonavir is an FDA-approved HIV protease inhibitor and potent CYP3A inhibitor utilized "
                    "as a pharmacokinetic booster to enhance plasma levels of co-administered partner therapeutics. "
                    "In malaria, historical Phase 1 trials evaluated Ritonavir strictly as a pharmacokinetic adjuvant "
                    "to increase partner antimalarial drug exposure, NOT as an independent etiological/curative antiparasitic agent."
                )
                suitability_warn = (
                    "BIOLOGICAL SUITABILITY WARNING: Ritonavir does not target Plasmodium parasites or possess standalone "
                    "curative efficacy for malaria. It was evaluated in Phase 1 exclusively as an adjuvant PK booster. "
                    "Assuming monotherapy efficacy for an emerging syndrome carries critical biological risk."
                )
            elif is_surrogate:
                bio_role = "Disqualified Non-Therapeutic Surrogate / Toxicant"
                comp_suitability = "Disqualified Toxicant (Unsuitable for Human Therapeutics)"
                cand_relevance_desc = f"Disqualified: {surrogate_reason}"
                suitability_warn = f"Disqualified: {surrogate_reason}"
            elif has_direct_ind and ind_phase is not None and ind_phase >= 4.0:
                bio_role = "Approved Indication Therapeutic"
                comp_suitability = f"FDA-Approved Drug Indicated for {canonical_disease} (Direct Evidence)"
                cand_relevance_desc = f"Direct Indication Evidence: {ind_explanation}"
                suitability_warn = None
            elif has_direct_ind and ind_phase is not None and ind_phase >= 1.0:
                bio_role = f"Investigational Clinical Candidate (Phase {int(ind_phase)})"
                comp_suitability = f"Investigational Candidate (Documented Phase {int(ind_phase)} trials in {canonical_disease})"
                cand_relevance_desc = f"Clinical Trial Evidence: {ind_explanation}"
                suitability_warn = "Trial status indeterminate: Static registry record does not establish current trial recruitment or completion."
            else:
                bio_role = "Exploratory In-Vitro Lead (Indirect Target Hit)"
                comp_suitability = "Preclinical Target Lead (Indirect Target Association Only)"
                cand_relevance_desc = (
                    f"Indirect Target Hit: Molecule exhibits in-vitro potency against '{tsym}', "
                    f"but has NO documented clinical trials or approvals for '{canonical_disease}'. "
                    "Global testing in unrelated diseases cannot bypass early-stage development."
                )
                suitability_warn = "Indirect target hit: No clinical trials exist for this indication."

            if is_surrogate:
                evidence_tier = "Tier 4: Non-Therapeutic Surrogate / Toxicant"
                evidence_cat = "Disqualified Surrogate / Toxicant (Not a Viable Treatment)"
                tier_prio = 4
                disqualified = True
                relevance_note = f"Disqualified: {surrogate_reason}"
                effective_phase = None
            elif has_direct_ind and ind_phase is not None and ind_phase >= 4.0:
                evidence_tier = "Tier 1: Indication-Specific Approved Drug"
                evidence_cat = "Approved Drug (Verified Indication Treatment)"
                tier_prio = 1
                disqualified = False
                relevance_note = f"Direct Indication Evidence: {ind_explanation}"
                effective_phase = ind_phase
            elif has_direct_ind and ind_phase is not None and ind_phase >= 1.0:
                evidence_tier = "Tier 2: Indication-Specific Investigational Candidate"
                evidence_cat = f"Investigational Clinical Candidate (Phase {int(ind_phase)} for {canonical_disease})"
                tier_prio = 2
                disqualified = False
                relevance_note = f"Clinical Trial Evidence: {ind_explanation}"
                effective_phase = ind_phase
            else:
                evidence_tier = "Tier 3: Indirect Mechanistic Target Association"
                if global_phase >= 4.0:
                    evidence_cat = f"Exploratory Target Lead (Approved in Other Diseases: Global Phase {int(global_phase)})"
                elif global_phase >= 1.0:
                    evidence_cat = f"Exploratory Target Lead (Investigational Phase {int(global_phase)} in Other Diseases)"
                else:
                    evidence_cat = "Preclinical / In-Vitro Screened Lead"
                tier_prio = 3
                disqualified = False
                relevance_note = (
                    f"Indirect Target Hit: Molecule exhibits in-vitro potency against target '{r['target_symbol']}', "
                    f"but has NO documented clinical trials or approvals for '{canonical_disease}'. "
                    f"Global testing in unrelated diseases cannot bypass early-stage development."
                )
                effective_phase = None

            raw_cands.append({
                "compound_name": cname,
                "chembl_id": mid,
                "target_symbol": r["target_symbol"],
                "composite_priority_score": cps,
                "clinical_phase": global_phase,
                "indication_phase": effective_phase,
                "has_direct_indication": has_direct_ind,
                "evidence_tier": evidence_tier,
                "evidence_category": evidence_cat,
                "therapeutic_relevance": relevance_note,
                "biological_role": bio_role,
                "compound_suitability": comp_suitability,
                "target_type": cand_target_type,
                "target_relevance": cand_target_relevance,
                "clinical_context": cand_relevance_desc,
                "suitability_warning": suitability_warn,
                "indication_record_details": rec_details,
                "is_negative_control_surrogate": is_surrogate,
                "disqualified_from_repurposing": (is_surrogate or not has_direct_ind),
                "is_disqualified": is_surrogate,
                "median_pchembl": round(float(r["median_pchembl"]), 2) if pd.notna(r.get("median_pchembl")) else None,
                "molecular_weight": round(float(r["molecular_weight"]), 1) if pd.notna(r.get("molecular_weight")) else None,
                "drug_likeness_score": round(float(r.get("score_druglike", 0.5)), 2),
                "tier_priority": tier_prio,
                "score_breakdown": {
                    "target_association_contrib": round(float(r.get("score_target_assoc", 0.0)) * 0.35, 4),
                    "potency_contrib": round(float(r.get("score_potency", 0.0)) * 0.35, 4),
                    "druglikeness_contrib": round(float(r.get("score_druglike", 0.0)) * 0.15, 4),
                    "clinical_evidence_contrib": round(float(r.get("score_clinical_evidence", 0.0)) * 0.15, 4),
                },
            })

        # Multi-factor candidate ranking: Prioritize verified indication evidence over indirect target hits,
        # push non-therapeutic surrogates/toxicants to the bottom, and rank by CPS within tier
        raw_cands.sort(key=lambda x: (x["tier_priority"], -x["composite_priority_score"]))

        # Assign final rank
        cand_list = []
        for idx, c in enumerate(raw_cands[:top_candidates], 1):
            c["rank"] = idx
            cand_list.append(c)

        # Scientific pipeline maturity classification
        tier1_count = sum(1 for c in cand_list if c["evidence_tier"].startswith("Tier 1"))
        tier2_count = sum(1 for c in cand_list if c["evidence_tier"].startswith("Tier 2"))

        if tier1_count > 0:
            maturity = "Established Indication-Specific Approved Therapies Exist (Direct Indication Evidence)"
        elif tier2_count > 0:
            maturity = "Active Clinical Trials Documented for this Indication"
        elif len(cand_list) > 0:
            maturity = "Preclinical / In-Vitro Screen Stage (Only Indirect Target Associations; No Verified Indication Compounds)"
        else:
            maturity = "No Chemical Candidates Ranked (Target-Only Information)"

        missing_reasons = []
        if tier1_count == 0 and tier2_count == 0:
            missing_reasons.append(
                f"No verified indication-specific clinical candidates exist for '{canonical_disease}' in repository. "
                "Retrieved molecules are indirect target screen hits or chemical surrogates."
            )

        return {
            "disease_name": canonical_disease,
            "status": "EVIDENCE_RETRIEVED",
            "message": f"Retrieved {len(target_list)} targets and {len(cand_list)} candidates (Tier 1: {tier1_count}, Tier 2: {tier2_count}).",
            "missing_reasons": missing_reasons,
            "target_count": len(target_list),
            "candidate_count": len(cand_list),
            "top_targets": target_list,
            "top_candidates": cand_list,
            "has_verified_indication_treatment": (tier1_count > 0),
            "regulatory_maturity": maturity,
            "indication_status_disclaimer": (
                "Static indication records from ChEMBL reflect historical literature and trial records. "
                "They do not establish whether clinical development is currently active, terminated, or completed, "
                "nor do they guarantee therapeutic efficacy for the emerging syndrome."
            ),
        }

    def retrieve_evidence_for_similarity_matches(
        self,
        similarity_matches: List[Dict[str, Any]],
        top_candidates_per_disease: int = 3,
    ) -> List[Dict[str, Any]]:
        """Batches evidence retrieval across multiple ranked similarity matches."""
        enriched: List[Dict[str, Any]] = []
        for match in similarity_matches:
            dis = match["disease"]
            ev = self.retrieve_evidence_for_disease(dis, top_candidates=top_candidates_per_disease)
            # Combine similarity metrics with therapeutic evidence
            combined = {
                "similarity_context": match,
                "therapeutic_evidence": ev,
            }
            enriched.append(combined)
        return enriched
