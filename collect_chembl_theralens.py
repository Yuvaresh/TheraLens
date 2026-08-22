import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
INPUT = Path("data/raw/therapeutic_associations.csv")
OUTPUT_DIR = Path("data/raw")

# We use the strongest Open Targets relationships first.
# Ten targets per disease keeps this student project dataset manageable.
TOP_TARGETS_PER_DISEASE = 10

# Keep only measured activity records with a pChEMBL value where possible.
# 100 activities per target is enough for the first project dataset.
MAX_ACTIVITIES_PER_TARGET = 100

session = requests.Session()
session.headers.update({
    "User-Agent": "TheraLens-Student-Project/1.0",
    "Accept": "application/json",
})

def get_json(url, params=None):
    r = session.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r.json()

assoc = pd.read_csv(INPUT)

required = {
    "local_disease",
    "target_id",
    "target_symbol",
    "target_name",
    "association_score",
}
missing = required - set(assoc.columns)
if missing:
    raise ValueError(f"Missing columns in therapeutic_associations.csv: {sorted(missing)}")

# Select top Open Targets targets per disease.
top = (
    assoc.dropna(subset=["target_symbol", "association_score"])
    .sort_values(
        ["local_disease", "association_score"],
        ascending=[True, False]
    )
    .groupby("local_disease", as_index=False, group_keys=False)
    .head(TOP_TARGETS_PER_DISEASE)
    .drop_duplicates(subset=["target_symbol"])
    .reset_index(drop=True)
)

top.to_csv(OUTPUT_DIR / "chembl_target_candidates.csv", index=False)

target_rows = []
activity_rows = []
molecule_ids = set()

print(f"Selected {len(top)} unique target symbols from Open Targets.")

# Step 1: map target symbols to ChEMBL target IDs.
for i, row in top.iterrows():
    symbol = str(row["target_symbol"]).strip()

    try:
        data = get_json(
            f"{BASE}/target/search.json",
            params={"q": symbol, "limit": 20},
        )

        hits = data.get("targets", [])
        if not hits and isinstance(data.get("page_meta"), dict):
            # Some ChEMBL versions expose search results under different keys.
            hits = data.get("targets", [])

        # Save all candidates for later review.
        for rank, hit in enumerate(hits[:10], 1):
            target_rows.append({
                "target_symbol": symbol,
                "rank": rank,
                "chembl_target_id": hit.get("target_chembl_id"),
                "pref_name": hit.get("pref_name"),
                "target_type": hit.get("target_type"),
                "organism": hit.get("organism"),
            })

        print(f"[target {i+1}/{len(top)}] {symbol}: {len(hits)} candidates")

    except Exception as exc:
        target_rows.append({
            "target_symbol": symbol,
            "rank": None,
            "chembl_target_id": None,
            "pref_name": f"ERROR: {exc}",
            "target_type": None,
            "organism": None,
        })
        print(f"[target {i+1}/{len(top)}] ERROR {symbol}: {exc}")

    time.sleep(0.2)

target_map = pd.DataFrame(target_rows)
target_map.to_csv(OUTPUT_DIR / "chembl_target_mapping_candidates.csv", index=False)

# Step 2: use the first candidate for each symbol as a provisional mapping.
# We keep the candidate file so mappings can be manually reviewed later.
mapped = (
    target_map.dropna(subset=["chembl_target_id"])
    .sort_values(["target_symbol", "rank"])
    .drop_duplicates("target_symbol")
)

for i, row in mapped.iterrows():
    symbol = row["target_symbol"]
    chembl_id = row["chembl_target_id"]

    try:
        data = get_json(
            f"{BASE}/activity.json",
            params={
                "target_chembl_id": chembl_id,
                "pchembl_value__isnull": "false",
                "order_by": "-pchembl_value",
                "limit": MAX_ACTIVITIES_PER_TARGET,
                "only": ",".join([
                    "activity_id",
                    "molecule_chembl_id",
                    "target_chembl_id",
                    "assay_chembl_id",
                    "standard_type",
                    "standard_relation",
                    "standard_value",
                    "standard_units",
                    "pchembl_value",
                ]),
            },
        )

        activities = data.get("activities", [])

        for a in activities:
            molecule_id = a.get("molecule_chembl_id")
            if molecule_id:
                molecule_ids.add(molecule_id)

            activity_rows.append({
                "target_symbol": symbol,
                "chembl_target_id": chembl_id,
                "activity_id": a.get("activity_id"),
                "molecule_chembl_id": molecule_id,
                "assay_chembl_id": a.get("assay_chembl_id"),
                "standard_type": a.get("standard_type"),
                "standard_relation": a.get("standard_relation"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "pchembl_value": a.get("pchembl_value"),
            })

        print(
            f"[activity {i+1}/{len(mapped)}] {symbol}: "
            f"{len(activities)} records"
        )

    except Exception as exc:
        print(f"[activity] ERROR {symbol} / {chembl_id}: {exc}")

    time.sleep(0.2)

activities_df = pd.DataFrame(activity_rows)
activities_df.to_csv(OUTPUT_DIR / "chembl_activity.csv", index=False)

# Step 3: retrieve molecule properties for the unique compounds.
molecule_rows = []

for i, molecule_id in enumerate(sorted(molecule_ids), 1):
    try:
        data = get_json(f"{BASE}/molecule/{quote(molecule_id)}.json")
        props = data.get("molecule_properties") or {}

        molecule_rows.append({
            "molecule_chembl_id": data.get("molecule_chembl_id"),
            "pref_name": data.get("pref_name"),
            "molecule_type": data.get("molecule_type"),
            "max_phase": data.get("max_phase"),
            "molecular_formula": data.get("molecular_formula"),
            "molecular_weight": props.get("full_mwt"),
            "alogp": props.get("alogp"),
            "psa": props.get("psa"),
            "hba": props.get("hba"),
            "hbd": props.get("hbd"),
            "rotatable_bond": props.get("rtb"),
        })

        if i % 25 == 0 or i == len(molecule_ids):
            print(f"[molecule {i}/{len(molecule_ids)}]")

    except Exception as exc:
        print(f"[molecule] ERROR {molecule_id}: {exc}")

    time.sleep(0.1)

molecules_df = pd.DataFrame(molecule_rows)
molecules_df.to_csv(OUTPUT_DIR / "chembl_molecules.csv", index=False)

print("\nChEMBL extraction complete.")
print("Created:")
print("  data/raw/chembl_target_candidates.csv")
print("  data/raw/chembl_target_mapping_candidates.csv")
print("  data/raw/chembl_activity.csv")
print("  data/raw/chembl_molecules.csv")
print(f"Activities: {len(activities_df)}")
print(f"Unique molecules: {len(molecule_ids)}")
