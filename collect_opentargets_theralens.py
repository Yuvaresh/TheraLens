import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
INPUT = Path("data/raw/symptom_disease.csv")
OUTPUT_DIR = Path("data/raw")

SEARCH_QUERY = """
query SearchDisease($queryString: String!) {
  search(queryString: $queryString, entityNames: ["disease"]) {
    hits {
      id
      name
    }
  }
}
"""

ASSOCIATION_QUERY = """
query AssociatedTargets($efoId: String!, $index: Int!, $size: Int!) {
  disease(efoId: $efoId) {
    id
    name
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
      }
    }
  }
}
"""

session = requests.Session()
session.headers.update({"Content-Type": "application/json", "User-Agent": "TheraLens-Student-Project/1.0"})

def post(query, variables):
    r = session.post(API_URL, json={"query": query, "variables": variables}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload["data"]

df = pd.read_csv(INPUT)
disease_names = sorted(df["prognosis"].dropna().astype(str).str.strip().unique())

# Phase 1: map local disease names to Open Targets IDs.
mapping_rows = []

for i, disease in enumerate(disease_names, 1):
    try:
        data = post(SEARCH_QUERY, {"queryString": disease})
        hits = data.get("search", {}).get("hits", [])
        if hits:
            # Keep all candidates so ambiguous names can be reviewed later.
            for rank, hit in enumerate(hits[:5], 1):
                mapping_rows.append({
                    "local_disease": disease,
                    "rank": rank,
                    "open_targets_id": hit.get("id"),
                    "open_targets_name": hit.get("name"),
                })
        else:
            mapping_rows.append({
                "local_disease": disease,
                "rank": None,
                "open_targets_id": None,
                "open_targets_name": None,
            })
    except Exception as exc:
        mapping_rows.append({
            "local_disease": disease,
            "rank": None,
            "open_targets_id": None,
            "open_targets_name": f"ERROR: {exc}",
        })
    print(f"[search {i}/{len(disease_names)}] {disease}")
    time.sleep(0.25)

mapping = pd.DataFrame(mapping_rows)
mapping.to_csv(OUTPUT_DIR / "opentargets_disease_mapping_candidates.csv", index=False)

# Use the top search hit initially. This is a candidate mapping, not a final
# scientific assertion; ambiguous names should be reviewed before modeling.
top = (
    mapping.dropna(subset=["open_targets_id"])
    .sort_values(["local_disease", "rank"])
    .drop_duplicates("local_disease")
)

association_rows = []

for i, row in enumerate(top.iterrows(), 1):
    local_disease = row[1]["local_disease"]
    efo_id = row[1]["open_targets_id"]

    try:
        data = post(
            ASSOCIATION_QUERY,
            {"efoId": efo_id, "index": 0, "size": 100},
        )
        disease_obj = data.get("disease")

        if not disease_obj:
            print(f"[association] No disease object for {local_disease} ({efo_id})")
            continue

        assoc = disease_obj.get("associatedTargets", {})
        for item in assoc.get("rows", []):
            target = item.get("target") or {}
            association_rows.append({
                "local_disease": local_disease,
                "disease_id": disease_obj.get("id"),
                "open_targets_disease_name": disease_obj.get("name"),
                "target_id": target.get("id"),
                "target_symbol": target.get("approvedSymbol"),
                "target_name": target.get("approvedName"),
                "association_score": item.get("score"),
            })

        print(
            f"[association {i}/{len(top)}] {local_disease}: "
            f"{len(assoc.get('rows', []))} targets"
        )
    except Exception as exc:
        print(f"[association] ERROR {local_disease}: {exc}")

    time.sleep(0.35)

associations = pd.DataFrame(association_rows)
associations.to_csv(OUTPUT_DIR / "therapeutic_associations.csv", index=False)

print("\nDone.")
print("Created:")
print("  data/raw/opentargets_disease_mapping_candidates.csv")
print("  data/raw/therapeutic_associations.csv")
print(f"Diseases in source: {len(disease_names)}")
print(f"Diseases with candidate mappings: {top['local_disease'].nunique()}")
print(f"Association rows collected: {len(associations)}")
