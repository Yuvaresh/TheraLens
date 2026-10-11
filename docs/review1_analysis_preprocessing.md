# 4. Analysis & Preprocessing of Dataset

## 4.1 Dataset Overview

The **TheraLens** project aims to establish a machine learning framework for analyzing multi-domain biomedical data—spanning clinical symptoms, global epidemiological disease incidence, target-disease association scores, molecular bioactivities, and historical regulatory approval timelines—to forecast future therapeutic solution milestones.

To support this objective, a heterogeneous collection of twelve datasets was acquired, audited, preprocessed, and integrated. Table 4.1 summarizes the complete dataset inventory calculated directly from the project's repository files.

### Table 4.1: TheraLens Master Dataset Inventory

| Dataset Name | Source | Rows | Columns | Unique Diseases | Important Fields | Purpose in TheraLens |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Disease-Symptom Dataset** | Public Clinical Symptom Repository | 4,961 | 133 | 41 | 132 symptom binary flags, `prognosis` | Symptom-based disease identification & clinical baseline feature matrix |
| **WHO Measles Dataset** | World Health Organization (WHO) | 1,692 | 9 | N/A | `country_iso3`, `year`, `value_numeric` | Long-term global viral outbreak dynamics tracking |
| **WHO Yellow Fever Dataset** | World Health Organization (WHO) | 874 | 9 | N/A | `country_iso3`, `year`, `value_numeric` | Flavivirus incidence monitoring and endemic region analysis |
| **WHO Cholera 2012 Dataset** | World Health Organization (WHO) | 19 | 70 | 1 | Country, Weeks 1–52 reported cases | High-resolution weekly outbreak dynamics (2012) |
| **WHO Cholera 2013 Dataset** | World Health Organization (WHO) | 18 | 70 | 1 | Country, Weeks 1–52 reported cases | High-resolution weekly outbreak dynamics (2013) |
| **WHO Cholera 2014 Dataset** | World Health Organization (WHO) | 18 | 70 | 1 | Country, Weeks 1–52 reported cases | High-resolution weekly outbreak dynamics (2014) |
| **Open Targets Therapeutic Associations** | Open Targets Platform | 3,636 | 7 | 38 | `local_disease`, `target_symbol`, `association_score` | Disease-to-biological target evidence scoring |
| **ChEMBL Bioactivity** | EMBL-EBI ChEMBL DB | 8,023 | 10 | N/A | `molecule_chembl_id`, `chembl_target_id`, `pchembl_value` | Potency quantification and target-compound binding strength |
| **ChEMBL Molecular Properties** | EMBL-EBI ChEMBL DB | 6,319 | 11 | N/A | `molecular_weight`, `alogp`, `psa`, `hba`, `hbd` | Physicochemical characterization and Lipinski rule screening |
| **FDA Drug Approval Timeline** | FDA Drugs@FDA Source Tables | 12,243 | 12 | N/A | `appl_no`, `drug_name`, `active_ingredient`, `first_approval_year` | Ground-truth historical therapeutic approval milestone year |
| **ChEMBL Drug Indications** | EMBL-EBI ChEMBL REST API | 30,663 | 8 | 2,190 | `molecule_chembl_id`, `indication`, `efo_id`, `mesh_id` | Standardized ontology mapping from drugs to clinical diseases |
| **Therapeutic Development Timeline** | TheraLens Integration Pipeline | 555,323 | 14 | 2,378 | `disease_name`, `chembl_molecule_id`, `first_approval_year` | Master target dataset linking historical solution year to disease-drug pairs |

---

## 4.2 Disease and Symptom Dataset Analysis

The `symptom_disease.csv` dataset contains clinical presentation profiles across **41 distinct disease categories** with **132 binary symptom indicators**.

### Key Statistical Findings:
- **Total Records**: 4,961 rows and 133 columns.
- **Disease Categories**: 41 classes (balanced at exactly 121 observations per class).
- **Missing Values**: 0 missing entries (100% complete feature matrix).
- **Duplicate Records**: 4,657 rows (93.87%) are exact duplicate feature vectors generated via synthetic repetition.
- **Unique Feature Patterns**: 304 distinct clinical symptom combinations represent the underlying feature manifold.

![Disease Class Distribution](../outputs/data_analysis/disease_distribution.png)
*Figure 4.1: Distribution of records across the 41 disease classes in the symptom-disease dataset.*

![Top 20 Frequent Symptoms](../outputs/data_analysis/symptom_frequency.png)
*Figure 4.2: Top 20 most frequent symptom features observed in the clinical dataset.*

![Duplicate vs Unique Records](../outputs/data_analysis/symptom_duplicates.png)
*Figure 4.3: Duplication audit revealing 4,657 redundant synthetic rows vs 304 unique clinical patterns.*

---

## 4.3 Epidemiological Dataset Analysis

Longitudinal surveillance data from the World Health Organization (WHO) were analyzed across three major infectious diseases: Measles, Yellow Fever, and Cholera.

### Summary Statistics:
1. **Measles (`measles_who.csv`)**: 1,692 annual records across 48 countries spanning 1974 to 2024, accounting for a cumulative total of 29,416,642 reported cases.
2. **Yellow Fever (`yellow_fever_who.csv`)**: 874 annual records across 47 endemic countries spanning 1974 to 2024, accounting for 35,987 reported cases.
3. **Cholera (`cholera_2012_who.csv` to `cholera_2014_who.csv`)**: High-frequency weekly reporting across 19 countries. Originally formatted in wide format (52 weekly columns per country), these datasets were transformed into normalized long-format representations (`country`, `year`, `week`, `cases`) and exported to `data/processed/cholera_2012_long.csv` (988 rows), `cholera_2013_long.csv` (936 rows), and `cholera_2014_long.csv` (936 rows).

![Measles Outbreak Trend](../outputs/data_analysis/measles_trend.png)
*Figure 4.4: Longitudinal global measles incidence trend (1974–2024).*

![Cholera Weekly Dynamics](../outputs/data_analysis/cholera_trends.png)
*Figure 4.5: Normalized weekly cholera outbreak dynamics across epidemiological weeks (2012–2014).*

---

## 4.4 Therapeutic Association Analysis

The `therapeutic_associations.csv` dataset derived from the Open Targets Platform establishes evidence-backed links between human diseases and biological drug targets.

### Key Metrics:
- **Total Records**: 3,636 associations.
- **Unique Diseases**: 38 disease categories.
- **Unique Biological Targets**: 2,277 target gene symbols.
- **Association Score Range**: Minimum = 0.0015, Maximum = 0.9250, Mean = 0.2973, Median = 0.3013.
- **Missing Values & Duplicates**: 0 missing cells, 0 duplicate associations.

![Association Score Distribution](../outputs/data_analysis/target_association_distribution.png)
*Figure 4.6: Distribution of disease-target association scores from the Open Targets Platform.*

---

## 4.5 Compound and Bioactivity Analysis

Pharmacological potency and physicochemical property profiles were evaluated using `chembl_activity.csv` and `chembl_molecules.csv`.

### Bioactivity Statistics (`chembl_activity.csv`):
- **Total Measured Bioactivity Records**: 8,023 records.
- **Unique Candidate Compounds**: 6,319 molecules (`molecule_chembl_id`).
- **Unique Biological Targets**: 166 targets (`chembl_target_id`).
- **pChEMBL Potency Values**: Mean pChEMBL = 6.42 (-log10 M affinity), representing nanomolar to micromolar binding strengths.

### Molecular Property Profiling (`chembl_molecules.csv`):
- **Total Characterized Compounds**: 6,319 molecules.
- **Molecular Weight (MW)**: Mean = 428.5 Da (84.2% conform to Lipinski Rule of 5 threshold <= 500 Da).
- **Lipophilicity (ALogP)**: Mean = 3.12 (89.6% conform to Lipinski threshold LogP <= 5.0).
- **Polar Surface Area (PSA)**: Mean = 92.4 Å² (91.1% exhibit high oral bioavailability potential <= 140 Å²).

![pChEMBL Distribution](../outputs/data_analysis/chembl_pchembl_distribution.png)
*Figure 4.7: Bioactivity affinity distribution (pChEMBL values).*

![Molecular Weight Distribution](../outputs/data_analysis/molecular_weight_distribution.png)
*Figure 4.8: Molecular weight distribution against the Lipinski Rule of 5 baseline.*

---

## 4.6 FDA Therapeutic Development Analysis

The `fda_drug_approval_timeline.csv` dataset provides the historical benchmark for regulatory approvals.

### Timeline Highlights:
- **Total Approval Records**: 12,243 entries.
- **Unique Active Ingredients**: 3,142 normalized active ingredient entities.
- **Unique Trade Names**: 5,643 drug products.
- **Milestone Span**: Earliest recorded approval year = 1939, Latest = 2026.
- **Missing Milestone Years**: 749 historical records lacking explicit pre-1938 first approval year metadata.

![FDA Approval Timeline](../outputs/data_analysis/fda_approval_timeline.png)
*Figure 4.9: Annual FDA first approval solution milestone volume over time (1939–2024).*

---

## 4.7 Data Integration

An automated entity-resolution pipeline matched FDA active ingredients to ChEMBL chemical structure entities, fetched disease indication ontologies, and constructed the master historical timeline.

### Integration Metrics:
1. **FDA-to-ChEMBL Match Rate**: **97.4%** (3,059 of 3,142 active ingredients successfully resolved).
2. **Unique ChEMBL Molecules Matched**: 2,589 distinct molecule entities.
3. **ChEMBL Indication Records**: 30,663 clinical indication records retrieved via REST API.
4. **Master Timeline Records**: **555,323 disease-drug historical solution milestone pairs** spanning 2,378 standardized disease ontology terms.

![Data Integration Funnel](../outputs/data_analysis/fda_chembl_integration_funnel.png)
*Figure 4.10: Entity resolution funnel connecting FDA active ingredients to disease indications.*

---

## 4.8 Data Preprocessing

Table 4.2 documents the exact preprocessing operations executed across all datasets.

### Table 4.2: Summary of Executed Data Preprocessing Operations

| Dataset | Before Preprocessing | Executed Preprocessing Operation | After Preprocessing |
| :--- | :--- | :--- | :--- |
| **Disease-Symptom** | 4,961 raw records | Duplicate pattern detection & exact record deduplication | 304 unique clinical symptom vectors |
| **WHO Cholera (2012)** | 19 countries x 52 weekly columns (wide) | Melted wide matrix to long time-series format (`country`, `year`, `week`, `cases`) | 988 long-format weekly records |
| **WHO Cholera (2013)** | 18 countries x 52 weekly columns (wide) | Melted wide matrix to long time-series format (`country`, `year`, `week`, `cases`) | 936 long-format weekly records |
| **WHO Cholera (2014)** | 18 countries x 52 weekly columns (wide) | Melted wide matrix to long time-series format (`country`, `year`, `week`, `cases`) | 936 long-format weekly records |
| **FDA Approval Timeline** | 12,243 raw entries | Lowercase normalization, space trimming, & active ingredient extraction | 3,142 unique active ingredients |
| **FDA-ChEMBL Mapping** | 3,142 active ingredients | Multi-stage API matching (exact, synonym, salt stripping, combination split) | 3,059 matched ingredients (97.4%) |
| **ChEMBL Indications** | 2,589 matched molecules | REST API pagination & EFO/MeSH ontology mapping | 30,663 indication records |
| **Master Timeline** | 12,243 FDA x 30,663 indications | Exploded entity inner join on matched `chembl_molecule_id` | 555,323 master timeline records |

---

## 4.9 Data Quality Findings

Table 4.3 summarizes the quality audit findings and corrective actions.

### Table 4.3: Master Data Quality Audit Summary

| Dataset | Missing Values | Duplicates | Identified Quality Issue | Corrective Action Taken |
| :--- | :---: | :---: | :--- | :--- |
| **Disease-Symptom** | 0 (0.0%) | 4,657 (93.9%) | Synthetic row repetition | Deduplicated to 304 unique clinical patterns |
| **WHO Measles** | 3,384 nulls | 0 (0.0%) | Unreported country-years | Filtered complete country-year time series |
| **WHO Yellow Fever** | 1,748 nulls | 0 (0.0%) | Sparse endemic country reporting | Aggregated annual national totals |
| **WHO Cholera (2012-14)** | 152 null cells | 0 (0.0%) | Wide format unsuited for time-series ML | Melted to long format with zero-filled non-report weeks |
| **Open Targets** | 0 (0.0%) | 0 (0.0%) | None | Extracted top 10 scoring targets per disease |
| **ChEMBL Activity** | 0 (0.0%) | 0 (0.0%) | Heterogeneous bioactivity units | Filtered measured pChEMBL log-affinity values |
| **ChEMBL Molecules** | 20,850 nulls | 0 (0.0%) | Missing properties for biologicals | Retained raw property flags without artificial imputation |
| **FDA Timeline** | 749 nulls | 0 (0.0%) | Unrecorded pre-1938 approval dates | Excluded missing year entries from time series milestone evaluation |
| **FDA-ChEMBL Mapping** | 83 nulls | 0 (0.0%) | Obsolete trade names & electrolytes | Audited in `fda_chembl_match_audit.csv` without silent deletion |

---

## 4.10 Summary

All twelve project datasets have been comprehensively analyzed, quality-audited, transformed, and integrated into a unified data architecture. 

### Key Technical Deliverables Accomplished:
1. Created high-resolution long-format epidemiological datasets (`cholera_2012_long.csv`, `cholera_2013_long.csv`, `cholera_2014_long.csv`).
2. Generated 16 academic publication-grade visualizations in `outputs/data_analysis/`.
3. Created summary inventory, quality audit, and preprocessing CSVs (`dataset_inventory.csv`, `data_quality_summary.csv`, `preprocessing_summary.csv`).
4. Documented the historical solution milestone concept (*first FDA approval year*) without claiming clinical cure guarantees or executing premature model training.
