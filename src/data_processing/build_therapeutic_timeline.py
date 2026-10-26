#!/usr/bin/env python3
"""
TheraLens: FDA Drug Approval Timeline → ChEMBL Molecule → Drug Indication → Disease Pipeline

This script builds a reproducible historical therapeutic-development timeline connecting
FDA-approved drugs to ChEMBL molecules, drug indications, and disease terms.

Definition:
The FDA approval year is defined as the "historical therapeutic solution milestone"
(the first qualifying regulatory approval associated with a therapeutic intervention
for the disease/indication).
"""

import json
import logging
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
try:
    import requests
except ImportError:
    requests = None

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
DATA_PROCESSED_DIR = Path("data/processed")
CACHE_DB_PATH = DATA_PROCESSED_DIR / ".chembl_cache.sqlite"

INPUT_FDA_TIMELINE = DATA_PROCESSED_DIR / "fda_drug_approval_timeline.csv"
OUTPUT_MOLECULE_MAPPING = DATA_PROCESSED_DIR / "fda_chembl_molecule_mapping.csv"
OUTPUT_INDICATIONS = DATA_PROCESSED_DIR / "chembl_drug_indications.csv"
OUTPUT_AUDIT = DATA_PROCESSED_DIR / "fda_chembl_match_audit.csv"
OUTPUT_TIMELINE = DATA_PROCESSED_DIR / "therapeutic_development_timeline.csv"

# Common drug salt/ester suffixes to strip when finding base parent molecules
SALT_SUFFIXES = {
    "SODIUM", "POTASSIUM", "HYDROCHLORIDE", "SULFATE", "ACETATE", "PHOSPHATE",
    "CITRATE", "MESYLATE", "MALEATE", "TARTRATE", "SUCCINATE", "FUMARATE",
    "BROMIDE", "CHLORIDE", "CALCIUM", "DIPROPIONATE", "PALMITATE", "DECANOATE",
    "VALERATE", "HCL", "HYCLATE", "BESYLATE", "DISODIUM", "MONOHYDRATE",
    "DIHYDRATE", "TRIHYDRATE", "MEGLUMINE", "ZINC", "TROMETHAMINE", "NITRATE",
    "LACTATE", "GLUCONATE", "BENZOATE", "PROPIONATE", "BUTYRATE", "ISOBUTYRATE",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# -----------------------------------------------------------------------------
# Persistent Disk Caching (SQLite)
# -----------------------------------------------------------------------------
class DiskCache:
    """Thread-safe SQLite persistent cache for HTTP requests."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, key: str) -> Optional[Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM api_cache WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logging.debug(f"Cache get error for key {key}: {e}")
        return None

    def set(self, key: str, value: Any) -> None:
        try:
            val_str = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (key, value) VALUES (?, ?)",
                    (key, val_str),
                )
                conn.commit()
        except Exception as e:
            logging.debug(f"Cache set error for key {key}: {e}")


# -----------------------------------------------------------------------------
# ChEMBL API Client
# -----------------------------------------------------------------------------
class ChEMBLClient:
    """Client for querying the official ChEMBL REST API with rate limiting & caching."""

    def __init__(self, cache: DiskCache, max_retries: int = 3, timeout: int = 15):
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for live ChEMBL REST API queries. "
                "Please install it using 'pip install requests'."
            )
        self.cache = cache
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "TheraLens-Pipeline/1.0 (Applied ML Project)",
                "Accept": "application/json",
            }
        )

    def _fetch_url(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Sort params to get deterministic cache key
        param_str = "&".join(f"{k}={params[k]}" for k in sorted(params.keys()))
        cache_key = f"{url}?{param_str}"

        cached_res = self.cache.get(cache_key)
        if cached_res is not None:
            return cached_res

        for attempt in range(1, 4):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    self.cache.set(cache_key, data)
                    return data
                elif resp.status_code in (429, 502, 503, 504):
                    time.sleep(attempt * 1.5)
                else:
                    logging.warning(f"ChEMBL API HTTP {resp.status_code} for {url} with params {params}")
                    return None
            except Exception as exc:
                logging.debug(f"Attempt {attempt} failed for {url}: {exc}")
                time.sleep(attempt * 1.0)

        return None

    def search_molecule_pref_name_exact(self, query: str) -> List[Dict[str, Any]]:
        data = self._fetch_url(f"{CHEMBL_BASE_URL}/molecule.json", {"pref_name__iexact": query})
        return data.get("molecules", []) if data else []

    def search_molecule_synonym_exact(self, query: str) -> List[Dict[str, Any]]:
        data = self._fetch_url(
            f"{CHEMBL_BASE_URL}/molecule.json",
            {"molecule_synonyms__molecule_synonym__iexact": query},
        )
        return data.get("molecules", []) if data else []

    def search_molecule_query(self, query: str) -> List[Dict[str, Any]]:
        data = self._fetch_url(f"{CHEMBL_BASE_URL}/molecule/search.json", {"q": query})
        return data.get("molecules", []) if data else []

    def get_drug_indications(self, molecule_chembl_id: str) -> List[Dict[str, Any]]:
        """Fetch all drug indication records for a given molecule ID, supporting pagination."""
        all_indications = []
        offset = 0
        limit = 100

        while True:
            data = self._fetch_url(
                f"{CHEMBL_BASE_URL}/drug_indication.json",
                {"molecule_chembl_id": molecule_chembl_id, "limit": limit, "offset": offset},
            )
            if not data:
                break

            indications = data.get("drug_indications", [])
            all_indications.extend(indications)

            meta = data.get("page_meta", {})
            if not meta.get("next") or len(indications) < limit:
                break

            offset += limit

        return all_indications


# -----------------------------------------------------------------------------
# Data Processing Helpers
# -----------------------------------------------------------------------------
def normalize_active_ingredient(raw_ingredient: str) -> str:
    """Normalize active ingredient string for matching."""
    if not isinstance(raw_ingredient, str) or not raw_ingredient.strip():
        return ""
    # Lowercase
    s = raw_ingredient.strip().lower()
    # Replace multiple whitespaces with single space
    s = re.sub(r"\s+", " ", s)
    # Strip obvious outer punctuation
    s = s.strip(".,;-()[]{}")
    return s


def strip_salt_suffixes(ingredient_str: str) -> str:
    """Remove common salt/ester words to find the parent chemical entity."""
    words = ingredient_str.upper().split()
    cleaned_words = [w for w in words if w not in SALT_SUFFIXES]
    return " ".join(cleaned_words)


# -----------------------------------------------------------------------------
# Matching Logic
# -----------------------------------------------------------------------------
def match_active_ingredient(
    raw_ing: str, client: ChEMBLClient
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Matches a single raw FDA active ingredient to ChEMBL molecule(s).
    Returns (mapping_record, audit_record).
    """
    norm_ing = normalize_active_ingredient(raw_ing)
    ing_upper = raw_ing.strip().upper()

    mapping_rec = {
        "fda_active_ingredient": raw_ing,
        "normalized_active_ingredient": norm_ing,
        "chembl_molecule_id": None,
        "chembl_pref_name": None,
        "match_method": "none",
        "match_status": "no_match",
        "candidate_count": 0,
    }

    audit_rec = {
        "fda_active_ingredient": raw_ing,
        "normalized_active_ingredient": norm_ing,
        "match_status": "no_match",
        "match_method": "none",
        "candidate_count": 0,
        "selected_chembl_id": None,
        "selected_pref_name": None,
        "confidence_score": 0.0,
        "audit_reason": "No matching ChEMBL molecule found",
    }

    if not norm_ing:
        audit_rec["audit_reason"] = "Empty or null active ingredient"
        return mapping_rec, audit_rec

    # --- Attempt 1: Exact pref_name match ---
    hits = client.search_molecule_pref_name_exact(ing_upper)
    if len(hits) == 1:
        m = hits[0]
        cid = m.get("molecule_chembl_id")
        pname = m.get("pref_name") or ing_upper
        mapping_rec.update(
            {
                "chembl_molecule_id": cid,
                "chembl_pref_name": pname,
                "match_method": "exact_pref_name",
                "match_status": "matched",
                "candidate_count": 1,
            }
        )
        audit_rec.update(
            {
                "match_status": "matched",
                "match_method": "exact_pref_name",
                "candidate_count": 1,
                "selected_chembl_id": cid,
                "selected_pref_name": pname,
                "confidence_score": 1.0,
                "audit_reason": "Exact case-insensitive match on ChEMBL pref_name",
            }
        )
        return mapping_rec, audit_rec
    elif len(hits) > 1:
        cids = [h.get("molecule_chembl_id") for h in hits]
        mapping_rec.update(
            {
                "match_method": "exact_pref_name",
                "match_status": "multiple_candidates",
                "candidate_count": len(hits),
            }
        )
        audit_rec.update(
            {
                "match_status": "multiple_candidates",
                "match_method": "exact_pref_name",
                "candidate_count": len(hits),
                "confidence_score": 0.0,
                "audit_reason": f"Multiple ChEMBL candidates found: {';'.join(cids)}",
            }
        )
        return mapping_rec, audit_rec

    # --- Attempt 2: Exact synonym match ---
    hits = client.search_molecule_synonym_exact(ing_upper)
    if len(hits) == 1:
        m = hits[0]
        cid = m.get("molecule_chembl_id")
        pname = m.get("pref_name") or ing_upper
        mapping_rec.update(
            {
                "chembl_molecule_id": cid,
                "chembl_pref_name": pname,
                "match_method": "exact_synonym",
                "match_status": "matched",
                "candidate_count": 1,
            }
        )
        audit_rec.update(
            {
                "match_status": "matched",
                "match_method": "exact_synonym",
                "candidate_count": 1,
                "selected_chembl_id": cid,
                "selected_pref_name": pname,
                "confidence_score": 0.95,
                "audit_reason": "Exact case-insensitive match on ChEMBL molecule synonym",
            }
        )
        return mapping_rec, audit_rec
    elif len(hits) > 1:
        u_ids = set(h.get("molecule_chembl_id") for h in hits if h.get("molecule_chembl_id"))
        if len(u_ids) == 1:
            cid = list(u_ids)[0]
            m = hits[0]
            pname = m.get("pref_name") or ing_upper
            mapping_rec.update(
                {
                    "chembl_molecule_id": cid,
                    "chembl_pref_name": pname,
                    "match_method": "exact_synonym_dedup",
                    "match_status": "matched",
                    "candidate_count": 1,
                }
            )
            audit_rec.update(
                {
                    "match_status": "matched",
                    "match_method": "exact_synonym_dedup",
                    "candidate_count": 1,
                    "selected_chembl_id": cid,
                    "selected_pref_name": pname,
                    "confidence_score": 0.95,
                    "audit_reason": "Multiple synonym records mapped to single ChEMBL ID",
                }
            )
            return mapping_rec, audit_rec
        else:
            mapping_rec.update(
                {
                    "match_method": "exact_synonym",
                    "match_status": "multiple_candidates",
                    "candidate_count": len(u_ids),
                }
            )
            audit_rec.update(
                {
                    "match_status": "multiple_candidates",
                    "match_method": "exact_synonym",
                    "candidate_count": len(u_ids),
                    "confidence_score": 0.0,
                    "audit_reason": f"Multiple distinct ChEMBL IDs for synonym: {';'.join(sorted(u_ids))}",
                }
            )
            return mapping_rec, audit_rec

    # --- Attempt 3: Salt Stripping for Single Active Ingredients ---
    if ";" not in ing_upper and "/" not in ing_upper:
        base_name = strip_salt_suffixes(ing_upper)
        if base_name and base_name != ing_upper:
            base_hits = client.search_molecule_pref_name_exact(base_name)
            if len(base_hits) == 1:
                m = base_hits[0]
                cid = m.get("molecule_chembl_id")
                pname = m.get("pref_name") or base_name
                mapping_rec.update(
                    {
                        "chembl_molecule_id": cid,
                        "chembl_pref_name": pname,
                        "match_method": "salt_stripped_base_pref_name",
                        "match_status": "matched",
                        "candidate_count": 1,
                    }
                )
                audit_rec.update(
                    {
                        "match_status": "matched",
                        "match_method": "salt_stripped_base_pref_name",
                        "candidate_count": 1,
                        "selected_chembl_id": cid,
                        "selected_pref_name": pname,
                        "confidence_score": 0.85,
                        "audit_reason": f"Matched parent molecule after stripping salt suffix: {base_name}",
                    }
                )
                return mapping_rec, audit_rec

            base_syn_hits = client.search_molecule_synonym_exact(base_name)
            if len(base_syn_hits) == 1:
                m = base_syn_hits[0]
                cid = m.get("molecule_chembl_id")
                pname = m.get("pref_name") or base_name
                mapping_rec.update(
                    {
                        "chembl_molecule_id": cid,
                        "chembl_pref_name": pname,
                        "match_method": "salt_stripped_base_synonym",
                        "match_status": "matched",
                        "candidate_count": 1,
                    }
                )
                audit_rec.update(
                    {
                        "match_status": "matched",
                        "match_method": "salt_stripped_base_synonym",
                        "candidate_count": 1,
                        "selected_chembl_id": cid,
                        "selected_pref_name": pname,
                        "confidence_score": 0.85,
                        "audit_reason": f"Matched parent synonym after stripping salt suffix: {base_name}",
                    }
                )
                return mapping_rec, audit_rec

    # --- Attempt 4: Multi-Component Active Ingredient Splitting ---
    if ";" in ing_upper or "/" in ing_upper:
        parts = [p.strip() for p in ing_upper.replace("/", ";").split(";") if p.strip()]
        matched_cids = []
        matched_pnames = []
        missing_parts = []

        for p in parts:
            p_clean = re.sub(r"\(.*?\)", "", p).strip()
            # Try component pref_name
            p_hits = client.search_molecule_pref_name_exact(p_clean)
            if len(p_hits) == 1:
                matched_cids.append(p_hits[0].get("molecule_chembl_id"))
                matched_pnames.append(p_hits[0].get("pref_name") or p_clean)
                continue

            # Try component synonym
            p_syns = client.search_molecule_synonym_exact(p_clean)
            if len(p_syns) == 1:
                matched_cids.append(p_syns[0].get("molecule_chembl_id"))
                matched_pnames.append(p_syns[0].get("pref_name") or p_clean)
                continue

            # Try component salt stripping
            p_base = strip_salt_suffixes(p_clean)
            if p_base and p_base != p_clean:
                pb_hits = client.search_molecule_pref_name_exact(p_base)
                if len(pb_hits) == 1:
                    matched_cids.append(pb_hits[0].get("molecule_chembl_id"))
                    matched_pnames.append(pb_hits[0].get("pref_name") or p_base)
                    continue

            missing_parts.append(p)

        if len(matched_cids) == len(parts) and len(parts) > 0:
            cid_str = ";".join(matched_cids)
            pname_str = ";".join(matched_pnames)
            mapping_rec.update(
                {
                    "chembl_molecule_id": cid_str,
                    "chembl_pref_name": pname_str,
                    "match_method": "multi_component_exact",
                    "match_status": "matched",
                    "candidate_count": len(parts),
                }
            )
            audit_rec.update(
                {
                    "match_status": "matched",
                    "match_method": "multi_component_exact",
                    "candidate_count": len(parts),
                    "selected_chembl_id": cid_str,
                    "selected_pref_name": pname_str,
                    "confidence_score": 0.80,
                    "audit_reason": f"All {len(parts)} combination components matched individual ChEMBL molecules",
                }
            )
            return mapping_rec, audit_rec
        elif len(matched_cids) > 0:
            cid_str = ";".join(matched_cids)
            pname_str = ";".join(matched_pnames)
            mapping_rec.update(
                {
                    "chembl_molecule_id": cid_str,
                    "chembl_pref_name": pname_str,
                    "match_method": "multi_component_partial",
                    "match_status": "matched",
                    "candidate_count": len(matched_cids),
                }
            )
            audit_rec.update(
                {
                    "match_status": "matched",
                    "match_method": "multi_component_partial",
                    "candidate_count": len(matched_cids),
                    "selected_chembl_id": cid_str,
                    "selected_pref_name": pname_str,
                    "confidence_score": 0.70,
                    "audit_reason": f"Partial combination match: {len(matched_cids)}/{len(parts)} components matched. Missing: {';'.join(missing_parts)}",
                }
            )
            return mapping_rec, audit_rec

    # --- Attempt 5: ChEMBL Search API Fallback ---
    search_hits = client.search_molecule_query(ing_upper)
    if search_hits:
        # Check exact pref_name in search hits
        exact_pref_hits = [
            h for h in search_hits
            if (h.get("pref_name") or "").strip().upper() == ing_upper
        ]
        if len(exact_pref_hits) == 1:
            m = exact_pref_hits[0]
            cid = m.get("molecule_chembl_id")
            pname = m.get("pref_name") or ing_upper
            mapping_rec.update(
                {
                    "chembl_molecule_id": cid,
                    "chembl_pref_name": pname,
                    "match_method": "search_exact_pref_name",
                    "match_status": "matched",
                    "candidate_count": 1,
                }
            )
            audit_rec.update(
                {
                    "match_status": "matched",
                    "match_method": "search_exact_pref_name",
                    "candidate_count": 1,
                    "selected_chembl_id": cid,
                    "selected_pref_name": pname,
                    "confidence_score": 0.75,
                    "audit_reason": "Matched exact pref_name via ChEMBL search endpoint",
                }
            )
            return mapping_rec, audit_rec

    return mapping_rec, audit_rec


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def run_pipeline() -> None:
    logging.info("Starting TheraLens FDA Timeline -> ChEMBL -> Indication -> Disease Pipeline")

    if not INPUT_FDA_TIMELINE.exists():
        logging.error(f"Input file not found: {INPUT_FDA_TIMELINE}")
        sys.exit(1)

    # Step 1: Read FDA Drug Approval Timeline
    fda_df = pd.read_csv(INPUT_FDA_TIMELINE)
    total_fda_records = len(fda_df)
    logging.info(f"Loaded {total_fda_records} FDA drug approval records from {INPUT_FDA_TIMELINE}")

    if "active_ingredient" not in fda_df.columns:
        logging.error("Missing required column 'active_ingredient' in FDA dataset.")
        sys.exit(1)

    # Extract unique raw active ingredients
    unique_ingredients = fda_df["active_ingredient"].dropna().unique().tolist()
    logging.info(f"Found {len(unique_ingredients)} unique active ingredients in FDA dataset")

    # Step 2 & 3: Match Active Ingredients to ChEMBL
    cache = DiskCache(CACHE_DB_PATH)
    client = ChEMBLClient(cache=cache)

    mapping_records = []
    audit_records = []

    logging.info("Matching active ingredients against ChEMBL API...")
    start_time = time.time()

    # Use ThreadPoolExecutor for fast API matching with caching
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ing = {
            executor.submit(match_active_ingredient, ing, client): ing
            for ing in unique_ingredients
        }

        completed = 0
        for future in as_completed(future_to_ing):
            completed += 1
            if completed % 250 == 0 or completed == len(unique_ingredients):
                logging.info(f"Matched {completed}/{len(unique_ingredients)} ingredients...")
            m_rec, a_rec = future.result()
            mapping_records.append(m_rec)
            audit_records.append(a_rec)

    match_elapsed = time.time() - start_time
    logging.info(f"Ingredient matching completed in {match_elapsed:.1f} seconds")

    mapping_df = pd.DataFrame(mapping_records)
    audit_df = pd.DataFrame(audit_records)

    # Attach sample fda_drug_name to mapping records
    sample_drug_map = fda_df.groupby("active_ingredient")["drug_name"].first().to_dict()
    mapping_df["fda_drug_name"] = mapping_df["fda_active_ingredient"].map(sample_drug_map)

    # Sort dataframes for consistency
    mapping_df.sort_values("fda_active_ingredient", inplace=True)
    audit_df.sort_values("fda_active_ingredient", inplace=True)

    # Save mapping and audit CSVs
    mapping_df.to_csv(OUTPUT_MOLECULE_MAPPING, index=False)
    audit_df.to_csv(OUTPUT_AUDIT, index=False)
    logging.info(f"Saved molecule mapping to {OUTPUT_MOLECULE_MAPPING}")
    logging.info(f"Saved match audit log to {OUTPUT_AUDIT}")

    # Extract all distinct matched ChEMBL molecule IDs
    matched_chembl_ids = set()
    for cid_str in mapping_df["chembl_molecule_id"].dropna():
        for cid in str(cid_str).split(";"):
            cid = cid.strip()
            if cid:
                matched_chembl_ids.add(cid)

    logging.info(f"Unique ChEMBL molecule IDs matched across ingredients: {len(matched_chembl_ids)}")

    # Step 4: Retrieve ChEMBL Drug Indications
    logging.info("Retrieving drug indication records from ChEMBL for matched molecules...")
    indication_records = []
    start_ind_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_cid = {
            executor.submit(client.get_drug_indications, cid): cid
            for cid in matched_chembl_ids
        }

        completed_ind = 0
        for future in as_completed(future_to_cid):
            completed_ind += 1
            if completed_ind % 100 == 0 or completed_ind == len(matched_chembl_ids):
                logging.info(f"Fetched indications for {completed_ind}/{len(matched_chembl_ids)} molecules...")
            ind_list = future.result()
            for ind in ind_list:
                # Standardize indication description field
                efo_term = ind.get("efo_term")
                mesh_heading = ind.get("mesh_heading")
                ind_text = efo_term or mesh_heading or "Unspecified Indication"

                indication_records.append({
                    "molecule_chembl_id": ind.get("molecule_chembl_id"),
                    "drugind_id": ind.get("drugind_id"),
                    "indication": ind_text,
                    "efo_id": ind.get("efo_id"),
                    "efo_term": efo_term,
                    "mesh_id": ind.get("mesh_id"),
                    "mesh_heading": mesh_heading,
                    "max_phase_for_ind": ind.get("max_phase_for_ind"),
                })

    ind_elapsed = time.time() - start_ind_time
    logging.info(f"Indication retrieval completed in {ind_elapsed:.1f} seconds")

    indications_df = pd.DataFrame(indication_records)
    if indications_df.empty:
        logging.warning("No indication records retrieved from ChEMBL!")
        indications_df = pd.DataFrame(columns=[
            "molecule_chembl_id", "drugind_id", "indication", "efo_id",
            "efo_term", "mesh_id", "mesh_heading", "max_phase_for_ind"
        ])

    indications_df.to_csv(OUTPUT_INDICATIONS, index=False)
    logging.info(f"Saved {len(indications_df)} indication records to {OUTPUT_INDICATIONS}")

    # Step 5 & 6: Join FDA Timeline with Indication Data to build Final Timeline
    logging.info("Joining FDA timeline with ChEMBL molecule indications...")

    # Build ingredient -> list of ChEMBL molecule IDs map
    ing_to_cids = {}
    ing_to_pnames = {}
    for _, row in mapping_df.iterrows():
        ing = row["fda_active_ingredient"]
        cid_val = row["chembl_molecule_id"]
        pname_val = row["chembl_pref_name"]
        if pd.notna(cid_val):
            cids = [c.strip() for c in str(cid_val).split(";") if c.strip()]
            pnames = [p.strip() for p in str(pname_val).split(";") if p.strip()] if pd.notna(pname_val) else []
            ing_to_cids[ing] = cids
            ing_to_pnames[ing] = dict(zip(cids, pnames)) if len(pnames) == len(cids) else {}

    # Explode FDA dataset by matched ChEMBL IDs
    fda_exploded_rows = []
    for _, fda_row in fda_df.iterrows():
        ing = fda_row["active_ingredient"]
        cids = ing_to_cids.get(ing, [])
        pname_map = ing_to_pnames.get(ing, {})

        if not cids:
            # Unmatched FDA record
            rec = fda_row.to_dict()
            rec["chembl_molecule_id"] = None
            rec["chembl_pref_name"] = None
            fda_exploded_rows.append(rec)
        else:
            for cid in cids:
                rec = fda_row.to_dict()
                rec["chembl_molecule_id"] = cid
                rec["chembl_pref_name"] = pname_map.get(cid)
                fda_exploded_rows.append(rec)

    fda_exploded_df = pd.DataFrame(fda_exploded_rows)

    # Merge exploded FDA dataset with Indications dataset
    merged_df = pd.merge(
        fda_exploded_df,
        indications_df,
        left_on="chembl_molecule_id",
        right_on="molecule_chembl_id",
        how="inner"  # Retain FDA approvals with active indications
    )

    # Derive standardized disease_name
    merged_df["disease_name"] = merged_df["efo_term"].fillna(merged_df["mesh_heading"])

    # Define requested columns for therapeutic_development_timeline.csv
    timeline_cols = [
        "disease_name",
        "efo_id",
        "efo_term",
        "mesh_id",
        "mesh_heading",
        "chembl_molecule_id",
        "chembl_pref_name",
        "drug_name",
        "active_ingredient",
        "appl_no",
        "product_no",
        "first_approval_date",
        "first_approval_year",
        "max_phase_for_ind",
    ]

    # Filter columns that exist in merged_df
    final_cols = [c for c in timeline_cols if c in merged_df.columns]
    timeline_df = merged_df[final_cols].drop_duplicates()

    # Save final therapeutic development timeline
    timeline_df.to_csv(OUTPUT_TIMELINE, index=False)
    logging.info(f"Saved {len(timeline_df)} final timeline records to {OUTPUT_TIMELINE}")

    # -------------------------------------------------------------------------
    # Step 8: Comprehensive Data Quality & Audit Report
    # -------------------------------------------------------------------------
    successful_matches = (mapping_df["match_status"] == "matched").sum()
    multiple_matches = (mapping_df["match_status"] == "multiple_candidates").sum()
    unmatched = (mapping_df["match_status"] == "no_match").sum()
    match_rate = (successful_matches / len(unique_ingredients)) * 100 if unique_ingredients else 0.0

    missing_approval_years = fda_df["first_approval_year"].isna().sum()
    duplicate_fda_records = fda_df.duplicated(subset=["appl_no", "product_no"]).sum()

    missing_disease_ids = (
        timeline_df["efo_id"].isna() & timeline_df["mesh_id"].isna()
    ).sum() if not timeline_df.empty else 0

    report = f"""
===============================================================================
THERALENS DATA PIPELINE — VALIDATION & DATA QUALITY REPORT
===============================================================================

1. FDA TIMELINE RECORDS PROCESSED:
   - Total FDA Records Processed:     {total_fda_records}
   - Unique Active Ingredients:       {len(unique_ingredients)}
   - Duplicate Approval Records:      {duplicate_fda_records}
   - Missing First Approval Years:    {missing_approval_years}

2. CHEMBL MOLECULE MATCHING RESULTS:
   - Successful ChEMBL Matches:       {successful_matches} ({match_rate:.1f}%)
   - Multiple ChEMBL Candidates:      {multiple_matches}
   - Unmatched Active Ingredients:    {unmatched}
   - Total Unique ChEMBL Molecule IDs: {len(matched_chembl_ids)}

3. CHEMBL DRUG INDICATIONS DATASET:
   - Total Indication Records:        {len(indications_df)}
   - Unique Indication Molecules:     {indications_df['molecule_chembl_id'].nunique() if not indications_df.empty else 0}

4. FINAL THERAPEUTIC DEVELOPMENT TIMELINE:
   - Total Disease-Indication Records: {len(timeline_df)}
   - Unique Disease/Indication Terms:  {timeline_df['disease_name'].nunique() if not timeline_df.empty else 0}
   - Missing Disease Identifiers:     {missing_disease_ids}

5. PROCESSED OUTPUT FILES GENERATED:
   - {OUTPUT_MOLECULE_MAPPING}
   - {OUTPUT_INDICATIONS}
   - {OUTPUT_AUDIT}
   - {OUTPUT_TIMELINE}

===============================================================================
DEFINITION: First approval year represents the 'historical therapeutic solution milestone'.
===============================================================================
"""
    print(report)
    with open(DATA_PROCESSED_DIR / "pipeline_quality_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    run_pipeline()
