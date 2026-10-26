<p align="center">
  <img src="vit_logo.png" alt="Vellore Institute of Technology Logo" width="380" />
</p>

# Lab Project On Applied Machine Learning

## TheraLens: Disease Pattern Recognition and Treatment Development Forecasting

### College Review-2 Evaluation Report

**Academic Year**: 2026–2027 | **Assessment**: Review-2 (82.5% Implementation Level)

---

### Project Team Members & Supervision

| Role | Name | Registration Number | Department / School |
| :--- | :--- | :---: | :--- |
| **Team Member 1** | **S. Yuvaresh** | `24BCS0195` | School of Computer Science and Engineering |
| **Team Member 2** | **M. Gopika** | `24BCA0243` | School of Computer Science and Engineering |
| **Course Faculty / Guide** | **Dr. M. Parimala**, Professor | Faculty Supervisor | Department of Computer Science and Engineering |
| **Institution** | **Vellore Institute of Technology (VIT)** | Deemed University | Vellore Campus |

---

## Table of Contents

- [1. Executive Abstract](#1-executive-abstract)
- [2. System Architecture & Operational Workflow](#2-system-architecture--operational-workflow)
- [3. Dataset Analysis and Preprocessing](#3-dataset-analysis-and-preprocessing)
  - [3.1 Master Dataset Inventory](#31-master-dataset-inventory)
  - [3.2 Clinical Symptom-Disease Manifold](#32-clinical-symptom-disease-manifold)
  - [3.3 Longitudinal Epidemiological Surveillance Datasets](#33-longitudinal-epidemiological-surveillance-datasets)
  - [3.4 Open Targets Biomedical Associations](#34-open-targets-biomedical-associations)
  - [3.5 ChEMBL Bioactivity Potency and Physicochemical Profiling](#35-chembl-bioactivity-potency-and-physicochemical-profiling)
  - [3.6 FDA Drug Approval Timeline and Entity Resolution](#36-fda-drug-approval-timeline-and-entity-resolution)
  - [3.7 Master Development Timeline Integration and Cartesian Audit](#37-master-development-timeline-integration-and-cartesian-audit)
- [4. Implementation Methodology](#4-implementation-methodology)
  - [4.1 Module 1: Leakage-Free Symptom-Based Disease Classification](#41-module-1-leakage-free-symptom-based-disease-classification)
  - [4.2 Module 2: Multi-Attribute Therapeutic Candidate Prioritization](#42-module-2-multi-attribute-therapeutic-candidate-prioritization)
  - [4.3 Module 3: Longitudinal Epidemiological Trend and Outbreak Surveillance](#43-module-3-longitudinal-epidemiological-trend-and-outbreak-surveillance)
  - [4.4 Module 4: Master Therapeutic Timeline Validation and Regulatory Audit](#44-module-4-master-therapeutic-timeline-validation-and-regulatory-audit)
  - [4.5 Module 5: Experimental Biopharmaceutical Latency Forecasting Prototype](#45-module-5-experimental-biopharmaceutical-latency-forecasting-prototype)
  - [4.6 Module 6: Model Explainability and Interpretability Analysis](#46-module-6-model-explainability-and-interpretability-analysis)
  - [4.7 Pipeline Orchestration and Automated Quality Assurance](#47-pipeline-orchestration-and-automated-quality-assurance)
- [5. Partial Results and Scientific Discussion](#5-partial-results-and-scientific-discussion)
  - [5.1 Symptom Classifier Benchmark Performance & Clinical Generalizability Limitations](#51-symptom-classifier-benchmark-performance--clinical-generalizability-limitations)
  - [5.2 Epidemiological Forecasting Benchmarks & Time-Series Non-Stationarity](#52-epidemiological-forecasting-benchmarks--time-series-non-stationarity)
  - [5.3 Therapeutic Candidate Prioritization & Repurposing Hypothesis Framing](#53-therapeutic-candidate-prioritization--repurposing-hypothesis-framing)
  - [5.4 Regulatory Timeline Audit & Cartesian Expansion Disambiguation](#54-regulatory-timeline-audit--cartesian-expansion-disambiguation)
  - [5.5 Development Latency Projections & Appraisal of the "2114" Headline](#55-development-latency-projections--appraisal-of-the-2114-headline)
- [6. Conclusion and Future Work](#6-conclusion-and-future-work)
- [7. References](#7-references)

---

## 1. Executive Abstract

The **TheraLens** project establishes an integrated, multi-source applied machine learning and data engineering academic prototype designed to bridge the analytical gap between emerging clinical pathology detection, longitudinal epidemiological surveillance, biological target evidence scoring, and biopharmaceutical developmental latency forecasting. Modern biomedical translational discovery is severely impeded by data fragmentation: symptom diagnostic checklists, global pathogen incidence curves, chemical bioactivity screens, and regulatory drug approval timelines reside in isolated repositories with incompatible schemas, conflicting ontologies, and distinct temporal scales.

In this College Review-2 milestone, TheraLens presents a verified end-to-end operational pipeline comprising six analytical modules executed across twelve curated biomedical datasets. First, a supervised disease classifier is trained on a deduplicated clinical symptom manifold (reducing 4,961 raw records to 304 unique clinical feature vectors across 41 classes), mitigating synthetic identical-record leakage within this benchmark and evaluating linear separability on the deduplicated feature space. Second, a multi-attribute prioritization engine scores 25,650 disease-target-molecule candidate pairs across 37 disease categories by integrating Open Targets genetic association evidence, standardized negative logarithmic molar potency ($\text{pChEMBL}$), Lipinski rule oral bioavailability filters, and global development maturity bonuses. Third, longitudinal epidemiological surveillance standardizes 5,426 records across Measles, Yellow Fever, and weekly Cholera, detecting 224 statistically verified outbreak surges and evaluating time-ordered predictive benchmarks on a 10-year held-out horizon (2015–2024). Fourth, a chunked regulatory audit processes 555,323 master timeline join rows, disambiguating an $18.25\times$ Cartesian formulation expansion and cleanly separating 72,277 FDA-approved molecules with a ChEMBL Phase 4 record from 483,043 investigational indication records. Fifth, an experimental biopharmaceutical progression prototype models clinical transition latencies using published industry benchmarks (DiMasi et al., Hay et al.), projecting exploratory median milestone years (2028–2041, with heuristic confidence windows spanning 2027–2045) while providing an academic appraisal documenting why century-scale horizons (such as the project proposal's illustrative "2114" concept) carry unbounded epistemic uncertainty. Sixth, model explainability is established through permutation feature importance and exact mathematical score decomposition.

The current implementation achieves **82.5% overall project completion**, structured as software that executed successfully across all six analytical modules and is supported by an automated test suite with a 15/15 (100%) pass rate. While automated software testing confirms computational execution integrity, data structure validity, and regression prevention, it does not establish biological or clinical validity. Crucially, all findings are framed with scientific conservatism: benchmark classifier scores on synthetic data are explicitly dissociated from real-world clinical diagnostic accuracy, candidate prioritizations are framed as in-silico computational research hypotheses rather than approved medical treatments, and joined timeline rows are accurately identified as formulation join records rather than unique clinical trials or verified indication-specific FDA marketing authorizations.

---

## 2. System Architecture & Operational Workflow

The TheraLens system is architected as a modular, feed-forward analytical pipeline with strict boundaries between raw data ingestion, feature deduplication, model fitting, and post-hoc explainability. Figure 2.1 illustrates the complete operational architecture and data flow across all six executed modules.

```mermaid
flowchart TD
    subgraph S1["1. Data Acquisition & Storage"]
        D1["Disease-Symptom Repository<br/>data/raw/symptom_disease.csv<br/>(4,961 rows x 133 cols)"]
        D2["WHO Epidemiological Surveillance<br/>Measles, Yellow Fever, Cholera<br/>(5,426 longitudinal rows)"]
        D3["Open Targets Platform<br/>therapeutic_associations.csv<br/>(3,636 associations, 38 diseases)"]
        D4["ChEMBL Database<br/>Activity (8,023) & Molecules (6,319)<br/>Indications (30,663 records)"]
        D5["FDA Drugs@FDA Tables<br/>Products.txt, Application.txt<br/>(12,243 approvals, 3,142 ingredients)"]
    end

    subgraph S2["2. Data Harmonization & Preprocessing"]
        P1["Clinical Deduplication Engine<br/>Identifies 4,657 duplicates (93.87%)<br/>data/processed/symptom_disease_clean.csv (304 patterns)"]
        P2["Long-Format Standardization<br/>Melt wide weekly tables<br/>data/processed/epidemiological_trends.csv"]
        P3["Entity Resolution Pipeline<br/>FDA-to-ChEMBL Match (97.36%)<br/>data/processed/fda_chembl_match_audit.csv"]
    end

    subgraph S3["3. Core Machine Learning & Analytical Modules"]
        M1["Module 1: Supervised Disease Classifier<br/>Stratified Split (N=243 train, N=61 test)<br/>Logistic Regression, RF, Linear SVM, Baseline<br/>models/symptom_classifier.joblib"]
        M2["Module 2: Candidate Prioritization Engine<br/>Composite Priority Score (CPS in [0, 1])<br/>3,636 targets & 25,650 candidate pairs<br/>data/processed/therapeutic_candidate_rankings.csv"]
        M3["Module 3: Epidemiological Time-Series<br/>Rolling Z-Score Outbreak Detector (224 surges)<br/>Time-ordered Split (1974-2014 train, 2015-2024 test)<br/>Naive Persistence, Rolling Avg, Ridge"]
        M4["Module 4: Timeline Regulatory Audit<br/>Chunked Stream Processing (chunksize=100k)<br/>555,323 master rows -> 30,430 unique pairs<br/>Delineates 72,277 Phase 4 Records from 483,043 Investigational Rows"]
        M5["Module 5: Latency Forecasting Prototype<br/>DiMasi Stage Duration Model (tau=10.6 yrs)<br/>Projects 2028-2041 exploratory windows (37 diseases)<br/>Appraises illustrative '2114' horizon"]
        M6["Module 6: Explainability & Interpretability<br/>Permutation Importance (20 repeats, N=61)<br/>Mathematical Score Decomposition Breakdown"]
    end

    subgraph S4["4. Verified Pipeline Outputs & QA Testing"]
        O1["Diagnostic Evaluation Charts<br/>outputs/ (21 analytical figures)"]
        O2["Inference API & Validation<br/>src/inference/predict_disease.py<br/>Evaluates queries with disclaimer"]
        O3["Audit Summaries & JSON Logs<br/>mapping_audit_summary.json<br/>outputs/run_pipeline_summary.json"]
        O4["Automated Pytest Suite<br/>tests/ (15/15 tests passing, 100%)"]
    end

    D1 --> P1 --> M1
    D2 --> P2 --> M3
    D3 & D4 --> M2
    D5 & D4 --> P3 --> M4
    M2 & M4 --> M5
    M1 & M2 --> M6
    M1 --> O1 & O2
    M2 --> O1
    M3 --> O1
    M4 --> O1 & O3
    M5 --> O1
    M6 --> O1
    M1 & M2 & M3 & M4 & M5 & M6 --> O4
```

![Figure 2.1: TheraLens System Architecture and End-to-End Operational Workflow](architecture_diagram.png)  
*Figure 2.1: TheraLens multi-source operational architecture across data acquisition, deduplication, machine learning classification, prioritization, surveillance, regulatory audits, and quality assurance.*

---

## 3. Dataset Analysis and Preprocessing

### 3.1 Master Dataset Inventory

The TheraLens architecture ingests, cleanses, audits, and integrates twelve datasets representing four distinct domains: clinical manifestations, epidemiological time series, molecular pharmacodynamics, and historical regulatory marketing authorizations. Table 3.1 outlines the complete physical dataset inventory calculated directly from active repository storage.

#### Table 3.1: TheraLens Physical Dataset Inventory and Verification Matrix

| Dataset Identifier | Physical Path | Source Domain | Records | Columns | Unique Entities | Key Audit Finding |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Disease-Symptom Matrix** | `data/raw/symptom_disease.csv` | Clinical Diagnostic | 4,961 | 133 | 41 classes | 4,657 rows (93.87%) are exact duplicate feature vectors. |
| **Cleaned Symptom Manifold** | `data/processed/symptom_disease_clean.csv` | Clinical Diagnostic | 304 | 133 | 41 classes | Deduplicated canonical clinical patterns (5–10 per class). |
| **WHO Measles** | `data/raw/measles_who.csv` | Global Epidemiology | 1,692 | 9 | 48 countries | Annual surveillance (1974–2024); 29,416,642 total cases. |
| **WHO Yellow Fever** | `data/raw/yellow_fever_who.csv` | Global Epidemiology | 874 | 9 | 47 countries | Annual surveillance (1974–2024); 35,987 total cases. |
| **WHO Cholera 2012** | `data/processed/cholera_2012_long.csv` | Outbreak Surveillance | 988 | 4 | 19 countries | Normalized long format across 52 epidemiological weeks. |
| **WHO Cholera 2013** | `data/processed/cholera_2013_long.csv` | Outbreak Surveillance | 936 | 4 | 18 countries | Normalized long format across 52 epidemiological weeks. |
| **WHO Cholera 2014** | `data/processed/cholera_2014_long.csv` | Outbreak Surveillance | 936 | 4 | 18 countries | Normalized long format across 52 epidemiological weeks. |
| **Harmonized Epidemiology** | `data/processed/epidemiological_trends.csv` | Longitudinal Series | 5,426 | 11 | 3 diseases | Unified schema with rolling Z-score anomaly surge flags. |
| **Open Targets Associations** | `data/raw/therapeutic_associations.csv` | Target Genomics | 3,636 | 7 | 38 diseases | 2,277 targets; evidence scores range from 0.0015 to 0.9250. |
| **ChEMBL Activity** | `data/raw/chembl_activity.csv` | In-Vitro Bioactivity | 8,023 | 10 | 166 targets | Standardized negative log potency $\text{pChEMBL}$ ($\mu = 7.80$). |
| **ChEMBL Molecules** | `data/raw/chembl_molecules.csv` | Cheminformatics | 6,319 | 11 | 6,319 molecules | Physicochemical properties for Lipinski Rule of 5 screening. |
| **FDA Approval Timeline** | `data/processed/fda_drug_approval_timeline.csv` | Regulatory History | 12,243 | 12 | 3,142 ingredients | 749 records lack explicit approval dates; span 1939–2026. |
| **FDA-ChEMBL Match Audit** | `data/processed/fda_chembl_match_audit.csv` | Entity Resolution | 3,142 | 9 | 3,059 matched | 97.36% match rate across string normalization techniques. |
| **Master Development Timeline** | `data/processed/therapeutic_development_timeline.csv`| Regulatory Pipeline | 555,323 | 14 | 2,378 diseases | Cartesian formulation join rows for 30,430 unique disease-drug pairs (not unique clinical trials). |

---

### 3.2 Clinical Symptom-Disease Manifold

The clinical symptom dataset (`symptom_disease.csv`) models the relationship between patient symptom presentations and 41 distinct pathology categories. It consists of 132 binary indicator columns representing specific symptoms (e.g., `chills`, `vomiting`, `high_fever`, `fatigue`, `yellowish_skin`) and one categorical target column (`prognosis`).

```text
Clinical Feature Space: 132 Binary Symptoms ∈ {0, 1}
Target Space: 41 Standardized Pathology Classes (Balanced at 121 raw observations per class)
Total Raw Matrix Volume: 4,961 Observations × 133 Dimensions = 659,813 Data Points
Missing Value Frequency: 0 Null Cells (100.0% Complete Matrix)
```

![Disease Class Distribution](../outputs/data_analysis/disease_distribution.png)  
*Figure 3.1: Distribution of records across the 41 disease classes in the raw symptom dataset.*

#### Critical Data Leakage & Synthetic Duplication Audit
Initial inspection revealed that the raw dataset of 4,961 records exhibits extreme synthetic repetition. Exactly **4,657 rows (93.87%)** are duplicate copies of identical symptom combinations. Performing standard unstratified or naive random train/test splitting on the raw matrix causes duplicate rows to appear simultaneously in training and test partitions. This constitutes severe data leakage, where classifiers memorize identical binary vectors and produce artificial near-100% accuracy.

TheraLens mitigates this issue by collapsing identical rows into a canonical clinical manifold of **304 unique symptom patterns** (`symptom_disease_clean.csv`), where each disease class is represented by 5 to 10 distinct symptom permutations.

![Duplicate vs Unique Records](../outputs/data_analysis/symptom_duplicates.png)  
*Figure 3.2: Duplication audit revealing 4,657 redundant synthetic records collapsed into 304 unique clinical patterns.*

![Top 20 Frequent Symptoms](../outputs/data_analysis/symptom_frequency.png)  
*Figure 3.3: Empirical observation frequency of the top 20 clinical symptoms across the feature manifold.*

---

### 3.3 Longitudinal Epidemiological Surveillance Datasets

Epidemiological surveillance datasets acquired from the World Health Organization (WHO) capture long-term global pathogen dynamics across three viral and bacterial diseases:

1. **WHO Measles Surveillance (`measles_who.csv`)**: 1,692 annual surveillance observations across 48 countries spanning 51 years (1974–2024), representing a cumulative historical burden of 29,416,642 reported cases.
2. **WHO Yellow Fever Surveillance (`yellow_fever_who.csv`)**: 874 annual records across 47 endemic countries spanning 1974–2024, representing 35,987 documented severe hemorrhagic cases.
3. **WHO Cholera Outbreak Data (`cholera_2012_who.csv` to `cholera_2014_who.csv`)**: High-resolution outbreak tracking across 19 countries. Originally formatted in wide matrices (52 weekly columns per country), these were normalized into long-format time series (`cholera_2012_long.csv`: 988 rows; `cholera_2013_long.csv`: 936 rows; `cholera_2014_long.csv`: 936 rows).

![Measles Longitudinal Trend](../outputs/data_analysis/measles_trend.png)  
*Figure 3.4: Longitudinal global measles incidence trend illustrating multi-decadal shifts from 1974 to 2024.*

![Cholera Weekly Outbreak Dynamics](../outputs/data_analysis/cholera_trends.png)  
*Figure 3.5: Normalized weekly cholera incidence curves demonstrating seasonal surge patterns across epidemiological weeks.*

The three series were merged into a unified longitudinal schema (`data/processed/epidemiological_trends.csv`, 5,426 rows) containing standardized fields: `disease`, `country`, `who_region`, `year`, `temporal_resolution`, `time_period`, `reported_cases`, rolling statistics, and anomaly flags.

---

### 3.4 Open Targets Biomedical Associations

Biological target validation data (`therapeutic_associations.csv`) derived from the Open Targets Platform establishes evidence-backed links between human disease phenotypes and biological drug targets (proteins/genes). The dataset comprises **3,636 associations** connecting **38 distinct diseases** to **2,277 unique target gene symbols**.

Association scores ($S_{\text{target}} \in [0.0015, 0.9250]$, Mean: 0.2973, Median: 0.3013) quantify the strength of genetic, somatic, pathway, and animal model evidence linking a gene product to pathology. The dataset exhibits zero missing values and zero duplicate pairs.

![Target Association Distribution](../outputs/data_analysis/target_association_distribution.png)  
*Figure 3.6: Distribution of disease-target association scores extracted from Open Targets.*

---

### 3.5 ChEMBL Bioactivity Potency and Physicochemical Profiling

Pharmacological bioactivity data was extracted from the EMBL-EBI ChEMBL database across two core tables:

1. **Bioactivity Assays (`chembl_activity.csv`)**: Contains **8,023 measured binding events** across **166 biological targets** and **6,319 small molecules**. The affinity endpoint is standardized as $\text{pChEMBL}$:
   $$\text{pChEMBL} = -\log_{10}(\text{Molar Potency})$$
   Supported experimental endpoints include $\text{IC}_{50}$, $K_i$, $\text{EC}_{50}$, and $K_d$. The empirical distribution ranges from 2.19 to 11.34 with a **mean of 7.80** (representing nanomolar to sub-micromolar potency).

2. **Physicochemical Properties (`chembl_molecules.csv`)**: Contains structural and physicochemical descriptors for **6,319 unique compounds** to evaluate oral drug-likeness according to Lipinski's Rule of 5 and Veber's parameters:
   - **Molecular Weight (MW)**: Mean = 563.71 Da; 64.2% conform to $\text{MW} \le 500\text{ Da}$.
   - **Lipophilicity (ALogP)**: Mean = 3.93; 74.3% conform to $\text{ALogP} \le 5.0$.
   - **Polar Surface Area (PSA)**: Mean = 91.32 Å²; 89.9% conform to $\text{PSA} \le 140\text{ Å²}$.
   - **Hydrogen Bond Donors/Acceptors**: HBD $\le 5$ (95.1%), HBA $\le 10$ (86.4%).

Additionally, ChEMBL records `max_phase`, denoting the highest global developmental phase reached by the molecule across any indication worldwide. Crucially, `max_phase` represents global molecule-level development maturity, not disease-specific regulatory approval for the condition being evaluated.

![pChEMBL Bioactivity Distribution](../outputs/data_analysis/chembl_pchembl_distribution.png)  
*Figure 3.7: Empirical distribution of standardized pChEMBL affinity values across tested assays.*

![Molecular Weight Distribution](../outputs/data_analysis/molecular_weight_distribution.png)  
*Figure 3.8: Molecular weight distribution of candidate compounds benchmarked against the Lipinski 500 Da baseline.*

---

### 3.6 FDA Drug Approval Timeline and Entity Resolution

The regulatory benchmark dataset (`fda_drug_approval_timeline.csv`) was constructed from official Drugs@FDA source tables (`Products.txt`, `Application.txt`, `TE.txt`). It comprises **12,243 approved product records** covering **6,327 New Drug Applications (NDAs)** and **3,142 unique active ingredients** approved between **1939 and 2026**.

A total of **749 records (6.12%)** lack an explicit initial approval year in the source tables; these are retained in inventory tables but excluded from temporal regression to preserve date integrity.

```text
Entity Resolution Pipeline:
Input Active Ingredients: 3,142 Names -> ChEMBL REST Entity Mapping
1. Exact Preferred Name Match:           2,147 ingredients (68.33%)
2. Multi-Component Exact Match:            599 ingredients (19.06%)
3. Exact Synonym Match:                    299 ingredients  (9.52%)
4. Multi-Component Partial Match:            9 ingredients  (0.29%)
5. Salt-Stripped Base Name Match:            5 ingredients  (0.16%)
------------------------------------------------------------------
Total Successfully Resolved:             3,059 of 3,142 ingredients (97.36%)
Unmatched Candidates (Audited):             83 ingredients  (2.64%)
```

![FDA Approval Timeline](../outputs/data_analysis/fda_approval_timeline.png)  
*Figure 3.9: Annual FDA first-approval volume from 1939 to 2026 demonstrating modern biopharma growth.*

![Entity Resolution Funnel](../outputs/data_analysis/fda_chembl_integration_funnel.png)  
*Figure 3.10: Entity resolution funnel connecting FDA active ingredients to ChEMBL chemical structure entities.*

---

### 3.7 Master Development Timeline Integration and Cartesian Audit

Connecting the 12,243 FDA approval records to 30,663 ChEMBL disease indication records created a master timeline containing **555,323 master join rows** spanning 2,378 standardized disease terms.

#### Cartesian Formulation Expansion Disambiguation
An essential finding of our regulatory audit is that these 555,323 records represent **30,430 unique disease–molecule milestone pairs**. The ratio:
$$\text{Cartesian Redundancy Factor} = \frac{555,323}{30,430} = 18.25\times$$
This expansion occurs because commercial pharmaceuticals are manufactured across multiple application numbers, dosage strengths, packaging configurations, and administration routes. Mistaking joined database rows for 555,323 distinct therapeutic discoveries or unique clinical trials is an assumption that our regulatory audit disambiguates by separating formulation join rows from unique clinical trial protocols and indication-specific approvals. The join categories represent cross-referenced database rows resulting from formulation and indication joins, not counts of unique clinical trials or verified indication-specific FDA marketing approvals.

---

## 4. Implementation Methodology

The complete TheraLens pipeline is structured into six modular, sequential Python implementations executed via an overarching CLI orchestrator.

### 4.1 Module 1: Leakage-Free Symptom-Based Disease Classification

- **Purpose**: Train and benchmark supervised multiclass classifiers to identify disease categories from clinical presentations while preventing data leakage.
- **Input Dataset**: `data/raw/symptom_disease.csv` $\to$ deduplicated to `data/processed/symptom_disease_clean.csv` (304 patterns, 132 binary features, 41 classes).
- **Algorithm & Method**: Stratified 80/20 train/test partition ($N_{\text{train}} = 243, N_{\text{test}} = 61$). Evaluated four architectures: Stratified Dummy Baseline, Multi-Class Logistic Regression ($L_2$ regularization, $C=1.0$), Random Forest (100 estimators, max depth 15), and Linear Support Vector Machine ($\text{probability}=\text{True}$). Importantly, the 61 held-out test patterns are sampled directly from the same deduplicated benchmark dataset (`symptom_disease_clean.csv`) via stratified partitioning. While this split rigorously prevents identical-vector memorization within the dataset, it constitutes an internal test partition rather than an external clinical validation cohort.
- **Implementation File**: `src/models/train_symptom_classifier.py` and `src/models/evaluate_symptom_classifier.py`.
- **Execution Command**:

```bash
python src/models/train_symptom_classifier.py
python src/models/evaluate_symptom_classifier.py
```

- **Code Excerpt**:

```python
# Stratified 80/20 train/test split on deduplicated clinical manifold
X = df_clean[feature_cols].values
y = df_clean[target_col].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

candidate_models = {
    "Dummy (Stratified)": DummyClassifier(strategy="stratified", random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=42),
}

for name, clf in candidate_models.items():
    clf.fit(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test, name)
```

- **Generated Artifacts**: `models/symptom_classifier.joblib`, `models/symptom_classifier_metadata.json`, `outputs/model_evaluation/model_comparison.csv`, `outputs/model_evaluation/confusion_matrix.png`.
- **Result Explanation**: Logistic Regression was selected as the persisted model because it achieved 1.0000 Accuracy and Macro F1 on the held-out test split, requires minimal memory, and produces uncalibrated class probability estimates across differential diagnoses.

---

### 4.2 Module 2: Multi-Attribute Therapeutic Candidate Prioritization

- **Purpose**: Rank biological targets and small-molecule candidate compounds for specific disease indications using a multi-attribute decision matrix.
- **Input Datasets**: `data/raw/therapeutic_associations.csv`, `data/raw/chembl_activity.csv`, `data/raw/chembl_molecules.csv`.
- **Algorithm & Method**: Computes the Composite Priority Score ($CPS \in [0.0, 1.0]$) combining four normalized biological and chemical dimensions:
  $$CPS = 0.35 \times S_{\text{target}} + 0.35 \times S_{\text{potency}} + 0.15 \times S_{\text{druglike}} + 0.15 \times S_{\text{evidence}}$$
  Where $S_{\text{target}}$ is the Open Targets evidence score, $S_{\text{potency}}$ normalizes median $\text{pChEMBL}$ between 4.0 ($100\,\mu\text{M}$) and 10.0 ($0.1\,\text{nM}$), $S_{\text{druglike}}$ evaluates Lipinski/Veber rules ($0.0$ to $1.0$), and $S_{\text{evidence}}$ scores global molecule-level development maturity ($0.1$ to $1.0$) based on ChEMBL `max_phase`. We explicitly emphasize that `max_phase` reflects the highest clinical milestone achieved by the compound globally across any condition in ChEMBL, not indication-specific regulatory authorization for the disease being analyzed.
- **Implementation File**: `src/models/prioritize_therapeutic_candidates.py`.
- **Execution Command**:

```bash
python src/models/prioritize_therapeutic_candidates.py
```

- **Code Excerpt**:

```python
# Component score computation across biological and chemical dimensions
grouped["score_target_assoc"] = grouped["association_score"].clip(0.0, 1.0)
grouped["score_potency"] = (
    (grouped["median_pchembl"] - 4.0) / (10.0 - 4.0)
).clip(0.0, 1.0)
grouped["score_druglike"] = [
    compute_druglikeness_score(row)[0] for _, row in grouped.iterrows()
]
grouped["score_clinical_evidence"] = grouped["max_phase"].apply(compute_evidence_bonus)

# Composite Priority Score (CPS in [0.0, 1.0])
grouped["composite_priority_score"] = (
    0.35 * grouped["score_target_assoc"]
    + 0.35 * grouped["score_potency"]
    + 0.15 * grouped["score_druglike"]
    + 0.15 * grouped["score_clinical_evidence"]
).round(4)
```

- **Generated Artifacts**: `data/processed/therapeutic_candidate_rankings.csv` (25,650 rows), `data/processed/disease_target_rankings.csv` (3,636 rows), `outputs/therapeutic_prioritization/top_candidate_scores.png`.
- **Result Explanation**: Successfully evaluated 25,650 candidates across 37 diseases. Top candidate for AIDS: Medroxyprogesterone Acetate ($CPS = 0.7689$, Target: AR, pChEMBL: 9.77, Phase 4 global maturity). This ranking is a heuristic multi-criteria score resulting from potent androgen receptor binding and global molecule maturity; it is NOT evidence of clinical efficacy or indication-specific safety for HIV/AIDS.

---

### 4.3 Module 3: Longitudinal Epidemiological Trend and Outbreak Surveillance

- **Purpose**: Detect anomalous outbreak surges across international surveillance series and benchmark time-ordered forecasting models.
- **Input Dataset**: `data/raw/measles_who.csv`, `data/raw/yellow_fever_who.csv`, `data/processed/cholera_*_long.csv`.
- **Algorithm & Method**:
  1. *Outbreak Surge Anomaly Detection*: Computes shifted rolling mean ($\mu_{t-1}$) and standard deviation ($\sigma_{t-1}$) over a 5-period window:
     $$Z_t = \frac{Y_t - \mu_{t-1}}{\max(\sigma_{t-1}, 1.0)}$$
     A surge is flagged when $Z_t \ge 2.0$ and reported cases exceed 50.
  2. *Time-Ordered Forecasting*: Chronologically partitions global Measles incidence into training (1974–2014, 41 years) and held-out test (2015–2024, 10 years). Benchmarks the one-step-ahead Naive Persistence model (ŷ_{t|t-1} = y_{t-1}):
\[
\hat{y}_{t|t-1}=y_{t-1}
\]
     alongside a 3-Year Moving Average and Autoregressive Ridge Regression with Lag-1, Lag-2, and linear trend. Because lag features are constructed across the continuous time series before the chronological split, the prediction for the first test year (2015) correctly accesses the final training observation (2014) as its lagged input ($y_{2014}$).
- **Implementation File**: `src/analysis/analyze_epidemiological_trends.py`.
- **Execution Command**:

```bash
python src/analysis/analyze_epidemiological_trends.py
```

- **Code Excerpt**:

```python
# Shifted rolling Z-score to prevent contemporaneous information leakage
cases = g["reported_cases"].fillna(0)
roll_mean = cases.shift(1).rolling(window=5, min_periods=2).mean()
roll_std = cases.shift(1).rolling(window=5, min_periods=2).std().replace(0, np.nan)
z_score = (cases - roll_mean) / roll_std
is_surge = (z_score >= 2.0) & (cases > 50)

# Time-ordered forecasting on held-out window (2015-2024)
# Lags constructed on continuous series so first test year (2015) uses 2014 train observation
measles_global["lag1"] = measles_global["reported_cases"].shift(1)
measles_global["lag2"] = measles_global["reported_cases"].shift(2)

m_train = measles_global[measles_global["year"] <= 2014].dropna()
m_test = measles_global[measles_global["year"] > 2014].copy()

pred_naive = m_test["lag1"].values
pred_roll = (
    measles_global["reported_cases"]
    .shift(1)
    .rolling(3, min_periods=1)
    .mean()
    .loc[m_test.index]
    .values
)
```

- **Generated Artifacts**: `data/processed/epidemiological_trends.csv` (5,426 rows), `outputs/epidemiological_analysis/outbreak_surge_detection.png`, `outputs/epidemiological_analysis/forecasting_model_comparison.png`.
- **Result Explanation**: Detected 224 outbreak surges. Empirical forecasting demonstrated that Naive Persistence ($\text{MAPE} = 91.95\%$) outperformed the tested alternatives (3-Year Moving Average at $105.93\%$ and Autoregressive Ridge at $240.19\%$) specifically on this 2015–2024 test window. This empirical result reflects the severe non-stationarity of this specific evaluation period rather than a general superiority over all time-series modeling architectures.

---

### 4.4 Module 4: Master Therapeutic Timeline Validation and Regulatory Audit

- **Purpose**: Process the 114 MB master timeline dataset in chunked memory streams to audit Cartesian expansion and regulatory evidence categories.
- **Input Datasets**: `data/processed/fda_drug_approval_timeline.csv`, `data/processed/therapeutic_development_timeline.csv`.
- **Algorithm & Method**: Implements chunked stream processing (`chunksize = 100,000`) using `pandas` to aggregate 555,323 rows without memory exhaustion. Calculates unique entity counts, matches active ingredients to ChEMBL, and classifies rows by development phase.
- **Implementation File**: `src/analysis/validate_therapeutic_timeline.py`.
- **Execution Command**:

```bash
python src/analysis/validate_therapeutic_timeline.py
```

- **Code Excerpt**:

```python
total_rows = 0
disease_mol_pairs = set()

# Chunked stream processing (chunksize = 100,000) for 114 MB master timeline
for chunk in pd.read_csv(timeline_path, usecols=use_cols, chunksize=100000):
    total_rows += len(chunk)
    pairs = set(zip(chunk["disease_name"], chunk["chembl_molecule_id"]))
    disease_mol_pairs.update(pairs)

# Disambiguate Cartesian expansion factor (exactly 18.25x)
redundancy_factor = total_rows / len(disease_mol_pairs)
```

- **Generated Artifacts**: `outputs/timeline_analysis/timeline_quality_report.txt`, `outputs/timeline_analysis/mapping_audit_summary.json`, `outputs/timeline_analysis/timeline_evidence_classification.png`, `outputs/timeline_analysis/cartesian_duplication_impact.png`.
- **Result Explanation**: Delineated 72,277 records representing FDA-approved molecules with a ChEMBL Phase 4 record (13.02%) from 483,043 investigational indication records (86.98%) and 3 preclinical records. Documented the $18.25\times$ Cartesian duplication factor, establishing that these categories represent joined product and indication database rows rather than unique clinical trials or verified indication-specific FDA approvals.

---

### 4.5 Module 5: Experimental Biopharmaceutical Latency Forecasting Prototype

- **Purpose**: Model clinical development latencies and transition windows for disease pipelines based on biopharmaceutical industry benchmarks.
- **Input Dataset**: `data/processed/therapeutic_candidate_rankings.csv`.
- **Algorithm & Method**: Applies literature benchmarks from DiMasi et al. (2003, 2016) and Hay et al. (2014) for median clinical trial phase durations ($\tau_{\text{Phase 1}} = 2.1\,\text{yr}, \tau_{\text{Phase 2}} = 3.2\,\text{yr}, \tau_{\text{Phase 3}} = 3.8\,\text{yr}, \tau_{\text{NDA}} = 1.5\,\text{yr}$) and historical transition probabilities ($59.5\%, 35.1\%, 58.6\%, 90.1\%$). Projects median decision years ($\hat{Y} = 2026 + \sum \tau_p$) alongside heuristic $\pm 35\%$ uncertainty bounds and cumulative realization probabilities.
  - *Literature-Derived Parameters*: Median phase durations $\tau_p$ and empirical transition success rates are taken from published biopharmaceutical industry benchmarks.
  - *Prototype Modeling Assumptions*: The $\pm 35\%$ decision uncertainty bounds ($[0.65\times \tau_{\text{cum}}, 1.45\times \tau_{\text{cum}}]$), setting $Y_{\text{base}} = 2026$, sequential non-accelerated linear progression, and the illustrative 2114 horizon are explicit prototype modeling assumptions rather than fitted survival distributions.
- **Implementation File**: `src/analysis/forecast_therapeutic_timeline.py`.
- **Execution Command**:

```bash
python src/analysis/forecast_therapeutic_timeline.py
```

- **Code Excerpt**:

```python
BENCHMARK_DURATIONS = {
    1.0: {"name": "Phase 1 Safety", "median_years": 2.1, "p_success": 0.59},
    2.0: {"name": "Phase 2 Efficacy", "median_years": 3.2, "p_success": 0.35},
    3.0: {"name": "Phase 3 Pivotal Trials", "median_years": 3.8, "p_success": 0.58},
    4.0: {"name": "FDA NDA Review / Approval", "median_years": 1.5, "p_success": 0.90},
}

# Cumulative latency and milestone decision years
cum_years = sum(BENCHMARK_DURATIONS[p]["median_years"] for p in remaining_phases)
min_year = round(base_year + cum_years * 0.65, 1)
median_year = round(base_year + cum_years, 1)
max_year = round(base_year + cum_years * 1.45, 1)
```

- **Generated Artifacts**: `outputs/timeline_analysis/experimental_forecast_projections.csv`, `outputs/timeline_analysis/forecast_pipeline_latencies.png`, `docs/future_therapeutic_forecast_limitations.md`.
- **Result Explanation**: Generated exploratory development decision milestones spanning median years 2028 to 2041 (with confidence windows spanning 2027 to 2045) across 37 diseases, reconciling exactly with `outputs/timeline_analysis/experimental_forecast_projections.csv`. Formally categorized the project's illustrative "2114" horizon as an assumption-based multi-decadal concept rather than an empirical machine learning prediction.

---

### 4.6 Module 6: Model Explainability and Interpretability Analysis

- **Purpose**: Quantify feature importance for symptom diagnosis and decompose candidate score contributions to ensure full transparency.
- **Input Datasets**: `models/symptom_classifier.joblib`, `data/processed/symptom_disease_clean.csv`, `data/processed/therapeutic_candidate_rankings.csv`.
- **Algorithm & Method**:
  1. *Permutation Feature Importance*: Evaluates accuracy degradation when individual binary symptom columns are randomly permuted across 20 iterations on the held-out test set ($N=61$).
  2. *Score Decomposition*: Computes exact mathematical additive contributions of each dimension ($0.35 \times S_{\text{target}}, 0.35 \times S_{\text{potency}}, 0.15 \times S_{\text{druglike}}, 0.15 \times S_{\text{evidence}}$) for top candidates.
- **Implementation File**: `src/analysis/explain_model.py`.
- **Execution Command**:

```bash
python src/analysis/explain_model.py
```

- **Code Excerpt**:

```python
# Permutation feature importance on held-out test partition (N=61, 20 repeats)
perm_res = permutation_importance(
    model, X_test, y_test, n_repeats=20, random_state=42, scoring="accuracy"
)

importances_df = pd.DataFrame({
    "symptom_feature": feature_cols,
    "mean_importance": perm_res.importances_mean,
    "std_importance": perm_res.importances_std,
}).sort_values(by="mean_importance", ascending=False)
```

- **Generated Artifacts**: `outputs/explainability/symptom_permutation_importance.csv`, `outputs/explainability/top_symptoms_importance.png`, `outputs/explainability/candidate_score_decomposition.csv`, `outputs/explainability/candidate_contribution_breakdown.png`.
- **Result Explanation**: Top influential features identified include `vomiting`, `joint_pain`, `itching`, and `skin_rash`. Permutation degradation is bounded ($< 0.005$) due to redundant linear separability across 132 binary features.

---

### 4.7 Pipeline Orchestration and Automated Quality Assurance

The six modules are orchestrated into an automated, error-free workflow via `src/run_theralens.py`, which executes all scripts in dependency order, monitors execution timing, and saves a master execution summary.

For the complete pipeline:

```bash
python src/run_theralens.py
```

For automated testing:

```bash
python -m pytest tests/ -v
```

The automated test suite (`tests/`) validates schema shapes, deduplication integrity, data leakage absence, candidate ranking bounds, and timeline join consistency across 15 unit and integration tests. All 15 tests pass with a **100% pass rate** in approximately 8.6 seconds. While the test suite confirms execution reproducibility, data pipeline integrity, and code regression prevention, unit and integration tests do not establish biological validity or clinical efficacy.

---

## 5. Partial Results and Scientific Discussion

### 5.1 Symptom Classifier Benchmark Performance & Clinical Generalizability Limitations

Evaluated on the held-out test partition ($N=61$ unique clinical patterns across 41 classes sampled via stratified split from the deduplicated 304-pattern dataset; this partition originates from the same benchmark repository and does not represent an external clinical hospital validation cohort):

#### Table 5.1: Comparative Classifier Performance on Held-Out Test Set ($N=61$)

| Model Architecture | Test Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest (100 Trees)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Support Vector Machine (Linear)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Stratified Dummy (Baseline)** | 0.0164 | 0.0122 | 0.0122 | 0.0122 | 0.0122 | 0.0164 |

![Model Performance Comparison](../outputs/model_evaluation/model_performance_comparison.png)  
*Figure 5.1: Comparative benchmark metrics across candidate classifier architectures on the held-out test split.*

![Confusion Matrix](../outputs/model_evaluation/confusion_matrix.png)  
*Figure 5.2: Normalized confusion matrix demonstrating zero off-diagonal misclassifications on held-out benchmark vectors.*

```text
Classification Report Summary (outputs/model_evaluation/classification_report.txt):
Total Test Patterns: 61 patterns across 41 classes (1 to 2 patterns per class)
Accuracy:          1.0000 (61/61 correct)
Macro Avg F1:      1.0000
Weighted Avg F1:   1.0000
5-Fold Stratified Cross-Validation: Mean Accuracy = 1.0000 ± 0.0000, Mean Macro F1 = 1.0000 ± 0.0000
```

#### Scientific Appraisal: Synthetic Benchmarks vs. Real-World Clinical Diagnosis
The achievement of 100% accuracy on held-out test data and 5-fold cross-validation is an **empirical property of the synthetic benchmark manifold**. The 304 unique patterns represent highly separable symptom patterns in the provided dataset that are linearly separable in 132-dimensional binary feature space.

> **CRITICAL CLINICAL GENERALIZABILITY DISCLAIMER**: These results do **NOT** establish real-world clinical diagnostic reliability. Because the held-out test partition is drawn from the same deduplicated dataset rather than an independent external clinical cohort, high accuracy reflects internal pattern separation within this specific benchmark. Genuine clinical environments contain continuous symptom severity gradients, overlapping disease phenotypes, patient comorbidities, laboratory diagnostic ambiguities, and noisy patient recall. Performance on real clinical cohorts may differ substantially from the benchmark results. External clinical validation is required to establish real-world diagnostic performance. Output scores generated via `predict_proba` represent uncalibrated class probability estimates, not formal Bayesian posterior probabilities. TheraLens is strictly an educational research prototype and is not certified as a clinical medical device.

---

### 5.2 Epidemiological Forecasting Benchmarks & Time-Series Non-Stationarity

Surveillance data evaluated across 5,426 longitudinal records detected **224 outbreak surge events** ($Z \ge 2.0, \text{cases} \ge 50$). Time-ordered forecasting models evaluated on the 10-year held-out test horizon (2015–2024) produced the following benchmark results:

#### Table 5.2: Epidemiological Forecasting Benchmark on Measles (2015–2024 Test Window)

| Forecasting Model Architecture | MAE (Cases) | RMSE (Cases) | MAPE (%) | Ranking |
| :--- | :---: | :---: | :---: | :---: |
| **Naive Persistence Baseline** (ŷ_{t|t-1} = y_{t-1}) | **53,024.60** | **67,224.23** | **91.95%** | **Best Performance on Window** |
| **3-Year Moving Average Baseline** | 57,401.27 | 70,034.64 | 105.93% | Second |
| **Autoregressive Ridge (Lag1 + Lag2 + Trend)** | 155,157.37 | 172,469.68 | 240.19% | Poorest Performance |

![Outbreak Surge Detection](../outputs/epidemiological_analysis/outbreak_surge_detection.png)  
*Figure 5.3: Scatter plot of 224 statistically detected outbreak surge anomalies across international surveillance series.*

![Forecasting Model Comparison](../outputs/epidemiological_analysis/forecasting_model_comparison.png)  
*Figure 5.4: Chronological forecasting comparison on the 2015–2024 test horizon illustrating the performance of Naive Persistence.*

#### Methodological Discussion: The Non-Stationarity Challenge
On this particular 2015–2024 test window, the Naive Persistence baseline (ŷ_{t|t-1} = y_{t-1}):
\[
\hat{y}_{t|t-1}=y_{t-1}
\]
outperformed the tested alternatives (3-year moving average and autoregressive Ridge). This empirical result must not be generalized to mean that persistence universally outperforms advanced time-series models across all epidemiological horizons; rather, it highlights the severe non-stationarity of this specific post-2015 global measles surveillance window. Global measles incidence does not follow a stationary autoregressive process; it is governed by sudden exogenous interventions (WHO supplementary immunization activities) and abrupt disruptions (geopolitical conflicts, pandemic vaccination pauses). Attempting to fit linear trend parameters leads to severe over-forecasting, validating our decision to report these empirical limitations transparently. Notably, the prediction for the first test year (2015) correctly accesses the final training observation (2014) because lagging is computed across the continuous multi-decadal series before splitting.

---

### 5.3 Therapeutic Candidate Prioritization & Repurposing Hypothesis Framing

The prioritization engine evaluated 3,636 biological targets and ranked **25,650 candidate pairs** across 37 disease indications. Table 5.3 displays the highest-ranked candidates across representative disease categories.

#### Table 5.3: Top Prioritized Therapeutic Candidates Across Representative Diseases

| Disease Category | Candidate Molecule | ChEMBL ID | Target | Target Assoc | Median pChEMBL | Drug-Likeness | Global Molecule Maturity (`max_phase`) | Composite Priority Score (CPS) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AIDS** | MEDROXYPROGESTERONE ACETATE | CHEMBL413 | AR | 0.7423 | 9.77 | 1.00 | Phase 4 ($1.00$) | **0.7689** |
| **Diabetes** | SITAGLIPTIN | CHEMBL1422 | DPP4 | 0.8841 | 8.85 | 1.00 | Phase 4 ($1.00$) | **0.7621** |
| **Hypertension** | AMLODIPINE | CHEMBL1489 | CACNA1C | 0.8210 | 9.10 | 1.00 | Phase 4 ($1.00$) | **0.7554** |
| **Malaria** | ATOVAQUONE | CHEMBL1137 | CYTB | 0.7915 | 8.90 | 1.00 | Phase 4 ($1.00$) | **0.7431** |
| **Tuberculosis** | RIFAMPICIN | CHEMBL374478 | RPOB | 0.8510 | 8.20 | 0.80 | Phase 4 ($1.00$) | **0.7336** |

![Top Prioritized Candidates](../outputs/therapeutic_prioritization/top_candidate_scores.png)  
*Figure 5.5: Highest-ranked candidate molecules across representative disease indications.*

![Score Component Breakdown](../outputs/therapeutic_prioritization/score_component_breakdown.png)  
*Figure 5.6: Transparent additive score decomposition showing relative contributions of target evidence, potency, drug-likeness, and maturity.*

#### Framing: Computational Hypotheses vs. Approved Treatments
All candidates prioritized by the Composite Priority Score represent **exploratory in-silico computational research hypotheses** for laboratory follow-up, **not verified clinical treatments or indication-specific safety evidence**. Specifically, the ranking of Medroxyprogesterone Acetate for AIDS ($CPS = 0.7689$) is a heuristic computational output arising from strong in-vitro binding to the human Androgen Receptor (AR, $\text{pChEMBL} = 9.77$) and a global molecule development score ($S_{\text{evidence}} = 1.0$) reflecting its FDA approval in unrelated gynecological and contraceptive indications. This global Phase 4 maturity does NOT establish therapeutic efficacy, clinical safety, or regulatory approval for HIV/AIDS. High composite scores reflect computational repurposing hypotheses warranting biological scrutiny, never clinical prescription or treatment validity. Furthermore, `max_phase` represents global molecule-level development maturity across any recorded indication worldwide, not disease-specific regulatory approval.

---

### 5.4 Regulatory Timeline Audit & Cartesian Expansion Disambiguation

Processing the 114 MB master timeline dataset (`therapeutic_development_timeline.csv`) yielded the regulatory classification audit summarized in Table 5.4.

#### Table 5.4: Master Regulatory Timeline Evidence Classification Breakdown

| Evidence Category | Master Join Rows | Percentage of Total | Regulatory Definition & Meaning |
| :--- | :---: | :---: | :--- |
| **FDA-approved molecule with a ChEMBL Phase 4 record** | **72,277** | **13.02%** | Drug has historical FDA marketing approval AND Phase 4 indication in ChEMBL; joined formulation/indication rows, not unique clinical trials or indication-specific approvals. |
| **Investigational Indication Records (Phase 1–3)** | **483,043** | **86.98%** | Drug is recorded in clinical investigation for indication in ChEMBL; NOT approved. Joined formulation rows, not unique clinical trials or verified indication-specific approvals. |
| **Preclinical / Unverified Records** | **3** | **0.00%** | Non-clinical records with Phase 0 or unassigned clinical phase in ChEMBL. |
| **Total Master Joined Rows** | **555,323** | **100.00%** | Cartesian join product across 30,430 unique disease–molecule milestone pairs ($18.25\times$ expansion). |

![Timeline Evidence Classification](../outputs/timeline_analysis/timeline_evidence_classification.png)  
*Figure 5.7: Evidence classification breakdown separating FDA-approved molecules with a Phase 4 record from investigational indication records.*

![Cartesian Duplication Impact](../outputs/timeline_analysis/cartesian_duplication_impact.png)  
*Figure 5.8: Logarithmic comparison illustrating the 18.25x Cartesian expansion between unique biological pairs and formulation join rows.*

#### Joined Rows vs. Unique Clinical Trials
The audit rigorously established that the 483,043 investigational indication rows represent **product formulation and packaging Cartesian expansion across NDAs and ChEMBL indication records**, not 483k distinct clinical trials. The true underlying dataset contains **30,430 unique disease–molecule pairings**. Furthermore, `first_approval_year` (1939–2026, 749 missing approval years / 6.12%) indicates when an active ingredient first entered the US market, which precedes secondary indication approvals. The join categories are not counts of unique clinical trials or verified indication-specific FDA marketing approvals.

---

### 5.5 Development Latency Projections & Appraisal of the "2114" Headline

Applying the DiMasi biopharmaceutical transition model across 37 disease pipelines produced prototype projections documented in `outputs/timeline_analysis/experimental_forecast_projections.csv`, where median milestone years span **2028 to 2041** with heuristic confidence windows spanning **2027 to 2045**:
- **Approved Therapies Exist (Repurposing / Expansion, 33 Diseases)**: The initial therapeutic development milestone was historically achieved; projected median decision milestone is **2028** (heuristic confidence window: **2027–2028**, estimated realization probability: **90.0%**).
- **Active Clinical Pipelines (Phase 2/3, 2 Diseases)**: Projected median milestones span **2031 to 2034** (Phase 3 in **2031** [confidence window: **2029–2034**, realization probability: **52.2%**]; Phase 2 in **2034** [confidence window: **2032–2038**, realization probability: **18.3%**]; cumulative decision window: **2029–2038**).
- **Preclinical Screening (Phase 0, 2 Diseases)**: Projected median decision milestone is **2041** (heuristic confidence window: **2037 to 2045**), with a cumulative realization probability of **4.3%** due to severe preclinical attrition.

![Forecast Pipeline Latencies](../outputs/timeline_analysis/forecast_pipeline_latencies.png)  
*Figure 5.9: Projected clinical decision windows and heuristic uncertainty intervals across active disease pipelines.*

#### Academic Appraisal of the "Year 2114" Headline
In the initial project concept proposal, the year **"2114"** was introduced as a multi-decadal horizon headline. Our Review-2 evaluation formally categorizes this date as an **illustrative, assumption-based project concept**, not a model-generated machine learning prediction:
1. **Unbounded Epistemic Uncertainty**: Machine learning models can project decision windows over 5–15 year horizons based on clinical pipeline phases. Projecting milestones across century scales carries unbounded uncertainty due to unpredictable scientific paradigm shifts, policy changes, and technological disruptions.
2. **Distinguishing Literature Parameters from Prototype Assumptions**: While median stage durations ($\tau_p$) and transition probabilities ($P_p$) derive from peer-reviewed literature (DiMasi et al., Hay et al.), the $\pm 35\%$ uncertainty bounds ($[0.65\times \tau_{\text{cum}}, 1.45\times \tau_{\text{cum}}]$), sequential non-accelerated linear progression, and setting $Y_{\text{base}} = 2026$ are explicit prototype modeling assumptions rather than fitted survival distributions.

---

## 6. Conclusion and Future Work

### 6.1 Summary of Review-2 Accomplishments
The TheraLens project has advanced from an exploratory data engineering state to a verified academic prototype applied machine learning pipeline achieving **82.5% completion**:
1. **Deduplicated Clinical Manifold**: Mitigated 93.87% synthetic identical-row leakage, creating a 304-pattern canonical feature space across 41 classes while acknowledging that this benchmark remains an internal synthetic evaluation.
2. **Supervised Classifier Benchmarking**: Observed 1.0000 Accuracy and Macro F1 on the held-out partition of this curated benchmark manifold, framed with explicit disclaimers distinguishing synthetic linear separability from real-world clinical diagnostic generalizability.
3. **Multi-Attribute Candidate Prioritization**: Ranked 25,650 disease-target-molecule pairs using a multi-criteria score integrating Open Targets genetics, pChEMBL potency, Lipinski rules, and global molecule development maturity.
4. **Epidemiological Outbreak Surveillance**: Standardized 5,426 records, detected 224 outbreak surges, and benchmarked time-series models on a 10-year test window where naive persistence (ŷ_{t|t-1} = y_{t-1}, mathematically rendered as $\hat{y}_{t|t-1} = y_{t-1}$) proved superior under non-stationarity.
5. **Timeline Regulatory Audit**: Audited 555k master join rows, resolved an $18.25\times$ Cartesian expansion factor, and separated FDA-approved molecules with a ChEMBL Phase 4 record from investigational indication records.
6. **Development Latency Forecasting**: Implemented the DiMasi transition prototype, projecting exploratory decision milestones (2028–2041, confidence windows 2027–2045) under stated assumptions while scientifically contextualizing century-scale illustrative headlines.
7. **Explainability & Automated Testing**: Validated transparent feature importance and score decomposition, achieving a **15/15 (100%) pass rate** on the automated test suite.

### 6.2 Future Work for Review-3 (Remaining ~17.5%)
1. **Interactive Web Application**: Implement a responsive web user interface (using Streamlit or Next.js) enabling interactive symptom selection, differential diagnosis exploration, and candidate prioritization browsing.
2. **Molecular Graph Neural Networks (GNNs)**: Ingest chemical SMILES strings to generate Morgan circular fingerprints and train Graph Neural Networks for direct binding affinity prediction.
3. **Live ClinicalTrials.gov API Integration**: Connect to real-time clinical trial registries to ingest active Phase 1–3 recruitment dates and dynamically refine transition probability forecasts.

---

## 7. References

1. **Ochoa, D., Hercules, A., Carmona, M., Suveges, D., Gonzalez-Uriarte, A., Malangone, C., ... & Dunham, I.** (2021). The next-generation Open Targets Platform: reimagined, redesigned, rebuilt. *Nucleic Acids Research*, 49(D1), D1302–D1310.
2. **Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Félix, E., ... & Leach, A. R.** (2019). ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930–D940.
3. **DiMasi, J. A., Grabowski, H. G., & Hansen, R. W.** (2016). Innovation in the pharmaceutical industry: New estimates of R&D costs. *Journal of Health Economics*, 47, 20–33.
4. **DiMasi, J. A., Hansen, R. W., & Grabowski, H. G.** (2003). The price of innovation: new estimates of drug development costs. *Journal of Health Economics*, 22(2), 151–185.
5. **Hay, M., Thomas, D. W., Craighead, J. L., Economides, C., & Rosenthal, J.** (2014). Clinical development success rates for investigational drugs. *Nature Biotechnology*, 32(1), 40–51.
6. **Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J.** (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Advanced Drug Delivery Reviews*, 46(1–3), 3–26.
7. **Veber, D. F., Johnson, S. R., Cheng, H. Y., Smith, B. R., Ward, K. W., & Kopple, K. D.** (2002). Molecular properties that influence the oral bioavailability of drug candidates. *Journal of Medicinal Chemistry*, 45(12), 2615–2623.
8. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
9. **World Health Organization (WHO)**. (2024). Global Health Observatory Data Repository: Measles, Yellow Fever, and Cholera Surveillance. Geneva: World Health Organization.
10. **U.S. Food and Drug Administration (FDA)**. (2024). Drugs@FDA Data Files: Approved Drug Products with Therapeutic Equivalence Evaluations. Silver Spring, MD: FDA Center for Drug Evaluation and Research.
