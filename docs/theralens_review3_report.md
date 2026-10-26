<p align="center">
  <img src="vit_logo.png" alt="Vellore Institute of Technology Logo" width="320" />
</p>

# Lab Project Report On Applied Machine Learning

## TheraLens: Machine Learning-Based Forecasting of Emerging Diseases and Future Therapeutic Development

### College Review-3 Final Evaluation Report

**Academic Year**: 2026–2027 | **Assessment**: Review-3 (100% Completed Implementation Level)

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

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Architecture and UML Diagrams](#3-architecture-and-uml-diagrams)
4. [Analysis and Preprocessing of Dataset](#4-analysis-and-preprocessing-of-dataset)
5. [Implementation (100% Completed)](#5-implementation-100-completed)
6. [Results and Empirical Evaluation](#6-results-and-empirical-evaluation)
7. [Conclusion and Future Work](#7-conclusion-and-future-work)
8. [References](#8-references)

---

## 1. Abstract

Modern biomedical translational discovery is severely impeded by data fragmentation: clinical symptom diagnostic checklists, global pathogen incidence curves, chemical bioactivity screens, and regulatory drug approval timelines reside in isolated repositories with incompatible schemas, conflicting ontologies, and distinct temporal scales. The **TheraLens** project establishes an integrated, multi-source applied machine learning and data engineering academic framework that bridges these isolated domains into an interpretable computational intelligence pipeline. 

TheraLens unifies five heterogeneous biomedical data streams comprising twelve primary datasets: (1) 4,961 raw symptom presentations deduplicated into 304 unique clinical patterns across 41 disease classes; (2) 5,426 longitudinal epidemiological surveillance records from the World Health Organization covering Measles, Yellow Fever, and Cholera; (3) 3,636 genome-wide disease-target associations from the Open Targets Platform; (4) 8,023 curated in-vitro bioactivity records and 6,319 small-molecule property profiles from EMBL-EBI ChEMBL (curated to targets matching Open Targets from an initial broad query of 22,238 activity records and 8,983 candidate compounds); and (5) 12,243 FDA approval records from Drugs@FDA mapped with a 97.36% ingredient resolution rate to 555,323 integrated master timeline entries across 30,430 unique disease-molecule pairs.

The completed pipeline implements six modular subsystems: (1) a leakage-resistant multiclass symptom classifier where Multinomial Logistic Regression achieves 1.0000 Accuracy and Macro F1 on a held-out test set ($N=61$) and 5-fold cross-validation ($N=304$), with documented disclaimers that synthetic linear separability does not imply real-world clinical diagnostic infallibility; (2) a reusable inference CLI (`predict_disease.py`) providing candidate differential diagnoses, softmax confidence metrics, and medical disclaimers; (3) a transparent multi-attribute Candidate Prioritization Score ($CPS \in [0.0, 1.0]$) ranking 3,636 biological targets and 25,650 small-molecule candidates across 37 pathologies; (4) epidemiological outbreak surveillance detecting 224 surge anomalies ($Z \ge 2.0$) and evaluating time-ordered forecasting models where Naive Persistence ($\text{MAE} = 53,024.60, \text{MAPE} = 91.95\%$) outperforms Autoregressive Ridge ($\text{MAPE} = 240.19\%$) due to empirical non-stationarity; (5) a regulatory timeline audit delineating 72,277 Phase 4 indication records from 483,043 investigational trial records, resolving an $18.25\times$ Cartesian join expansion; (6) a literature-grounded biopharmaceutical milestone latency forecasting model (DiMasi et al., Hay et al.) projecting empirical decision windows between 2026 and 2042 across 37 diseases, while providing an academically rigorous demystification of the illustrative proposal headline "2114"; and (7) permutation feature importance over 20 repeats identifying top discriminating clinical indicators. The entire pipeline executes deterministically in ~37 seconds, validated by an expanded 53-test automated pytest suite achieving a 100% pass rate with zero warnings.

---

## 2. Introduction

### 2.1 Problem Background and Motivation
The biopharmaceutical drug discovery and development pipeline is notorious for its protracted timelines, astronomical expenditures, and high attrition rates. Bringing a single new molecular entity (NME) from early target discovery to commercial market approval typically spans **10 to 15 years** and requires capitalized expenditures ranging from **USD 1.3 billion to 2.6 billion** (DiMasi et al., 2003, 2016; Wouters et al., 2020). Concurrently, overall clinical transition success rates from Phase 1 safety trials to regulatory approval remain between **11% and 14%**, with approximately 86% to 89% of candidates failing during clinical development due to lack of efficacy or unanticipated human toxicity (Hay et al., 2014; Wong, Siah, & Lo, 2019).

Compounding this latency is the profound fragmentation of open biomedical knowledge. Although international health agencies, genomic consortia, and chemical databases have democratized access to petabytes of scientific observations, these repositories exist in isolated silos:
- *Clinical diagnostic data* capture patient-reported symptoms and phenotypic indicators.
- *Epidemiological surveillance data* track longitudinal pathogen incidence and outbreak geography.
- *Functional genomics resources* (e.g., Open Targets) score statistical associations between diseases and biological genes.
- *Chemoinformatics databases* (e.g., ChEMBL) catalog in-vitro binding assays ($IC_{50}, K_i$) and molecular physicochemical structures.
- *Regulatory databases* (e.g., FDA Drugs@FDA) record marketing authorization approvals and commercial drug applications.

No unified, open-source pipeline currently integrates patient symptom presentation, biological target selection, chemical candidate prioritization, outbreak surveillance, and regulatory milestone latency into an end-to-end applied machine learning workflow.

### 2.2 Objectives and Tripartite Computational Framework
TheraLens integrates machine-learning-based disease classification, biomedical evidence retrieval, therapeutic candidate prioritization, and exploratory development-timeline simulation to support research into emerging diseases and future therapeutic development. The system explicitly demarcates three complementary computational modalities:
1. **Machine Learning Component**: Supervised symptom-based disease classification, feature extraction across a 132-symptom vocabulary, and permutation feature importance mapping.
2. **Evidence-Based Information Retrieval Component**: Multi-metric open-world symptom similarity (Jaccard, Weighted Jaccard, Cosine), genomic disease-target association mapping (Open Targets), and chemoinformatic candidate prioritization ($CPS$) with strict 4-tier evidence stratification and toxicant exclusion.
3. **Exploratory Development-Timeline Simulation Component**: Dual-Milestone biopharmaceutical progression modeling (Horizon A: Direct Advancement vs. Horizon B: Clinical Re-Validation) grounded in empirical literature benchmarks (DiMasi et al., Hay et al.) and technological acceleration sensitivity scenarios (15% and 30%).

---

## 3. Architecture and UML Diagrams

### 3.1 Master System Architecture
TheraLens follows a modular, feed-forward pipeline architecture where each analytical stage operates independently while consuming standardized data contracts.

<p align="center">
  <img src="architecture_diagram.png" alt="TheraLens Master System Architecture Diagram" width="100%" />
</p>

### 3.2 Pipeline Data-Flow Sequence
```
[User / Doctor Presentation] -> [predict_disease.py] -> [Vectorize 132 Symptoms] -> [Logistic Regression]
                                                                                           │
[Predicted Pathology Class] <--------------------------------------------------------------+
       │
       ▼
[Open Targets Associations] ---> [Filter Targets (3,636)] ---> [Cross-Reference ChEMBL Assays (8,023)]
                                                                               │
[Prioritized Candidate List] <-- [Rank by Composite Priority Score (CPS)] <----+
       │
       ▼
[Longitudinal Surveillance] ---> [Detect Outbreak Surges (Z >= 2.0)] ---> [Chronological Forecast]
       │
       ▼
[Regulatory Timeline Audit] ---> [Differentiate Phase 4 from Trials] ---> [Model Pipeline Latency (2026-2042)]
```

---

## 4. Analysis and Preprocessing of Dataset

### 4.1 Master Dataset Inventory
TheraLens processes 12 primary biomedical datasets spanning five independent institutions:

| # | Dataset Identifier | Source Organization | Dimensions | Target / Primary Attributes | Role in Pipeline |
| :-: | :--- | :--- | :---: | :--- | :--- |
| 1 | `symptom_disease.csv` | Public Clinical Benchmark | 4,961 $\times$ 133 | 132 binary symptoms, `prognosis` | Multiclass disease classification manifold |
| 2 | `measles_who.csv` | World Health Organization | 1,692 $\times$ 9 | `country_iso3`, `year`, `value_numeric` | 51-year annual global viral surveillance |
| 3 | `yellow_fever_who.csv` | World Health Organization | 874 $\times$ 9 | `country_iso3`, `year`, `value_numeric` | 51-year annual flavivirus surveillance |
| 4 | `cholera_2012_who.csv` | World Health Organization | 19 $\times$ 70 (raw) / 988 $\times$ 4 (long) | Country, Weeks 1–52 case counts | Granular weekly bacterial outbreak tracking |
| 5 | `cholera_2013_who.csv` | World Health Organization | 18 $\times$ 70 (raw) / 936 $\times$ 4 (long) | Country, Weeks 1–52 case counts | Granular weekly bacterial outbreak tracking |
| 6 | `cholera_2014_who.csv` | World Health Organization | 18 $\times$ 70 (raw) / 936 $\times$ 4 (long) | Country, Weeks 1–52 case counts | Granular weekly bacterial outbreak tracking |
| 7 | `therapeutic_associations.csv` | Open Targets Platform | 3,636 $\times$ 7 | `local_disease`, `target_symbol`, score | Disease-to-gene genomic association evidence |
| 8 | `chembl_activity.csv` | EMBL-EBI ChEMBL Database | 8,023 $\times$ 10 | `molecule_chembl_id`, `pchembl_value` | Curated in-vitro bioactivity (22,238 initial query) |
| 9 | `chembl_molecules.csv` | EMBL-EBI ChEMBL Database | 6,319 $\times$ 11 | MW, ALogP, PSA, HBA, HBD, RotB | Curated small molecules (8,983 initial query) |
| 10 | `fda_drug_approval_timeline.csv` | FDA Drugs@FDA Archive | 12,243 $\times$ 12 | `appl_no`, `active_ingredient`, year | Earliest market authorization milestones |
| 11 | `chembl_drug_indications.csv` | EMBL-EBI ChEMBL API | 30,663 $\times$ 8 | Molecule, EFO term, `max_phase_for_ind` | Indication-specific development phase mapping |
| 12 | `therapeutic_development_timeline.csv` | TheraLens Integration | 555,323 $\times$ 14 | Disease, Target, Molecule, Phase, Year | Master historical developmental timeline |

*Dataset Reconciliation Note*:
During initial exploratory data acquisition, broad queries across ChEMBL retrieved 22,238 raw activity records and 8,983 candidate molecule entities. After filtering for valid numeric $p\text{ChEMBL}$ values and strict target symbol intersection against Open Targets genes, the active dataset committed and processed by the modeling pipeline contains exactly **8,023 bioactivity measurements** and **6,319 unique small molecules**. Merging these with the 3,636 Open Targets associations yields 31,753 raw cross-referenced rows, which aggregate into **25,650 unique (disease, target, molecule) candidate pairs** in `therapeutic_candidate_rankings.csv`. Likewise, standardizing Measles (1,692 rows), Yellow Fever (874 rows), and weekly Cholera (2,860 melted long rows) yields the **5,426 records** in `epidemiological_trends.csv`.

### 4.2 Data Hygiene, Duplication, and Leakage Remediation
A critical discovery during exploratory data analysis revealed severe synthetic repetition in `symptom_disease.csv`:
- **Raw Row Count**: 4,961 records.
- **Exact Duplicates Detected**: Exactly **4,657 rows (93.87%)** possessed 100% identical feature vectors to other rows.
- **Label Consistency**: Zero label conflicts were observed; every identical feature vector consistently mapped to the exact same disease class.
- **Remediation**: The dataset was collapsed to **304 unique clinical presentations** (`symptom_disease_clean.csv`). Each of the 41 classes is represented by 5 to 10 distinct symptom configurations.
- **Leakage-Free Partitioning**: Train/test splitting (80/20, $N_{\text{train}}=243, N_{\text{test}}=61$) was performed exclusively on the deduplicated manifold using stratified sampling. Cross-set pattern intersection confirmed **0 overlapping feature vectors** between partitions.

```
Raw Benchmark: 4,961 Rows ──[Deduplication Engine]──> 304 Unique Presentations
(93.87% Synthetic Duplicates)                         (Zero Overlap Guarantee)
                                                                 │
                                                      [Stratified 80/20 Split]
                                                                 │
                                                                 ▼
                                                      Train: 243 / Test: 61
                                                      (Zero Data Leakage)
```

### 4.3 Surveillance Harmonization & Null Semantics
WHO surveillance datasets presented structural formatting challenges:
1. **Wide-to-Long Transformation**: Cholera surveillance tables originally formatted with 52 discrete weekly case columns were melted into normalized long-format records (`Country`, `Year`, `Week`, `Reported_Cases`).
2. **Preservation of Missing Observations**: Epidemiological gaps were retained as `NaN` rather than imputed as 0. In surveillance records, an unrecorded observation represents missing administrative reporting, not confirmed pathogen absence. Blind zero-filling would severely distort variance and underestimate surge anomalies.

### 4.4 Regulatory Entity Resolution & Cartesian Expansion Disambiguation
Merging Drugs@FDA active ingredients with ChEMBL chemical entities required robust entity resolution:
1. **Pharmaceutical Salt Stripping**: Active ingredients were normalized by stripping common salt and ester suffixes (`SODIUM`, `HYDROCHLORIDE`, `SULFATE`, `TARTRATE`, `MALEATE`, etc.) to map formulations back to parent bioactive compounds.
2. **Match Rate**: Out of **3,142 unique FDA active pharmaceutical ingredients** evaluated, **3,059 were successfully resolved** against parent ChEMBL molecules, achieving an empirical match rate of **97.36%** (with 83 unmatched entities detailed in `fda_chembl_match_audit.csv`).
3. **Cartesian Expansion Audit**: Joining FDA approvals to ChEMBL indications produced **555,323 records**. Crucially, an audit revealed that these 555k rows represent formulation, packaging, and dosage expansions across **30,430 unique (disease, molecule, milestone) triplets**—an expansion factor of **$18.25\times$**. They do **not** represent 555,323 independent clinical trials.
4. **Approval Year Semantics**: The recorded `first_approval_year` represents the earliest United States marketing authorization for that active chemical ingredient, **not** an indication-specific approval for secondary exploratory pathologies.

---

## 5. Implementation (100% Completed)

### 5.1 Module 1: Leakage-Free Symptom-Based Disease Classification
- **Source Module**: `src/models/train_symptom_classifier.py`
- **Algorithms Evaluated**:
  1. *Dummy Classifier (Stratified)*: Naive baseline sampling class labels according to training empirical distribution.
  2. *Multinomial Logistic Regression*: Regularized linear model with $L_2$ penalty ($C=1.0$, max iterations = 1000).
  3. *Random Forest Classifier*: Ensembled forest of 100 decision trees (max depth = 15, random seed = 42).
  4. *Support Vector Machine (Linear)*: Linear margin maximizer with Platt probability calibration.
- **Cross-Validation**: In addition to held-out test evaluation, a 5-fold stratified cross-validation was conducted across the 304 unique presentations.
- **Scientific Caveat**: The classifier achieves 1.0000 Accuracy on held-out test data because canonical symptom vectors in this synthetic benchmark are linearly separable. This performance establishes algorithmic correctness on the benchmark, **not clinical diagnostic infallibility in human patients**.

### 5.2 Module 2: Model Persistence and CLI Inference Engine
- **Source Module**: `src/inference/predict_disease.py`
- **Persistence**: Best performing model pipeline serialized to `models/symptom_classifier.joblib` alongside feature names (`feature_names.json`, 132 features) and class labels (`class_labels.json`, 41 classes).
- **Interface Capabilities**:
  - Implements `DiseasePredictor` class supporting raw comma-separated symptom strings, Python lists, or dictionary flags.
  - Case-insensitive normalization, whitespace trimming, and partial keyword matching.
  - Computes uncalibrated softmax confidence metrics over top-$k$ differential candidates.
  - Automatically appends mandatory research and ethical disclaimers.
  - *Example Diagnostic Query*: Passing clinical symptoms (`"chills, vomiting, high_fever, sweating, headache"`) correctly predicts **Malaria** with an output score of **0.2794**. The engine explicitly labels this score as an *uncalibrated model decision margin* (softmax probability across 41 competing classes), clearly documenting that it does not represent an absolute clinical diagnostic probability.

### 5.3 Module 3: Multi-Attribute Candidate Prioritization Engine
- **Source Module**: `src/models/prioritize_therapeutic_candidates.py`
- **Mathematical Formulation**:
  The Composite Priority Score ($CPS$) ranks candidate molecules for a given pathology:

  $$CPS = 0.35 \cdot S_{\text{target}} + 0.35 \cdot S_{\text{potency}} + 0.15 \cdot S_{\text{druglike}} + 0.15 \cdot S_{\text{evidence}}$$

  1. **Biological Target Evidence ($S_{\text{target}}$)**: Bounded Open Targets association score:
     $$S_{\text{target}} = \text{clip}(\text{association\_score}, 0.0, 1.0)$$
  2. **In-Vitro Potency ($S_{\text{potency}}$)**: Normalized from median $p\text{ChEMBL} = -\log_{10}(\text{Molar Potency})$, clipping between $4.0$ ($100\,\mu\text{M}$) and $10.0$ ($0.1\,\text{nM}$):
     $$S_{\text{potency}} = \text{clip}\left(\frac{\text{median}(p\text{ChEMBL}) - 4.0}{10.0 - 4.0}, 0.0, 1.0\right)$$
  3. **Drug-Likeness ($S_{\text{druglike}}$)**: Proportion of five Lipinski and Veber rules satisfied:
     - Molecular Weight $\le 500\,\text{Da}$
     - Calculated $\text{ALogP} \in [-0.4, 5.0]$
     - Polar Surface Area ($\text{PSA}$) $\le 140\,\text{Å}^2$
     - Hydrogen Bond Acceptors ($\text{HBA}$) $\le 10$
     - Hydrogen Bond Donors ($\text{HBD}$) $\le 5$
     - *Missing Value Handling*: Compounds lacking property records are assigned a neutral baseline score of $0.50$ with an explicit audit flag (`has_complete_properties=False`).
  4. **Clinical Development Evidence ($S_{\text{evidence}}$)**: Stepwise bonus based on the compound's highest global development phase recorded in ChEMBL:
     - Phase 4 (Approved Molecule): $1.00$
     - Phase 3 (Advanced Clinical Testing): $0.80$
     - Phase 2 (Clinical Efficacy Testing): $0.60$
     - Phase 1 (Clinical Safety Testing): $0.40$
     - Preclinical / Phase 0: $0.20$
     - Unrecorded / Missing Phase: $0.10$

- **Negative Control Audit**: The system documents that heuristic scores do not guarantee clinical efficacy. For example, in AIDS, Medroxyprogesterone Acetate achieves rank #1 ($CPS=0.7689$) due to strong PGR target binding and Phase 4 status, despite lacking anti-retroviral efficacy against HIV. This case serves as a vital negative control proving that $CPS$ is an in-silico prioritization heuristic, not a treatment recommendation.

### 5.4 Module 4: Longitudinal Epidemiological Surveillance & Outbreak Forecasting
- **Source Module**: `src/analysis/analyze_epidemiological_trends.py`
- **Surge Anomaly Detection**:
  Calculates a shifted rolling Z-score per country and disease to prevent future data leakage:
  $$\mu_{t} = \frac{1}{k}\sum_{i=1}^{k} y_{t-i}, \quad \sigma_{t} = \sqrt{\frac{1}{k-1}\sum_{i=1}^{k} (y_{t-i} - \mu_t)^2}, \quad Z_t = \frac{y_t - \mu_t}{\sigma_t}$$
  An outbreak surge anomaly is flagged when $Z_t \ge 2.0$ and reported cases $y_t > 50$ (window $k=5$ years).
- **Time-Ordered Forecasting Evaluation**:
  Longitudinal global Measles incidence is partitioned chronologically (Train: 1974–2014, 41 years; Held-Out Test: 2015–2024, 10 years). Evaluates:
  1. *Naive Persistence Baseline*: $\hat{y}_t = y_{t-1}$
  2. *3-Year Moving Average*: $\hat{y}_t = \frac{1}{3}\sum_{i=1}^3 y_{t-i}$
  3. *Autoregressive Ridge Regression*: $\hat{y}_t = w_1 y_{t-1} + w_2 y_{t-2} + w_3 t + b$ ($\alpha=100.0$)

### 5.5 Module 5: Regulatory Timeline Validation & Latency Modeling
- **Source Modules**: `src/analysis/validate_therapeutic_timeline.py` & `src/analysis/forecast_therapeutic_timeline.py`
- **Evidence Stratification**:
  Streams the 114 MB master timeline in chunks of 100,000 rows to categorize records into:
  - *Verified Phase 4 Indications*: 72,277 records (13.02%)
  - *Investigational Clinical Trials (Phases 1–3)*: 483,043 records (86.98%)
  - *Preclinical Records*: 3 records (<0.01%)
- **Biopharmaceutical Transition Duration Model**:
  Rather than ungrounded year extrapolation, pipeline progression horizons are modeled using empirical trial duration distributions (DiMasi et al., 2003, 2016; Hay et al., 2014):
  $$\text{Pipeline Duration} = \sum_{p \in \text{Remaining Phases}} \tau_p$$
  - Phase 1 Safety: $\tau_1 = 2.1$ years, $P_1 = 0.59$
  - Phase 2 Efficacy: $\tau_2 = 3.2$ years, $P_2 = 0.35$
  - Phase 3 Pivotal: $\tau_3 = 3.8$ years, $P_3 = 0.58$
  - FDA NDA Review: $\tau_4 = 1.5$ years, $P_4 = 0.90$
  - Cumulative Phase 1 to Approval Probability: $\approx 11.3\%$
- **Uncertainty Bounds**: Generates median projected completion years and asymmetric uncertainty bounds ($[-35\%, +45\%]$ trial duration variance) across 37 diseases with active clinical candidates.

### 5.6 Module 6: Model Explainability and Interpretability
- **Source Module**: `src/analysis/explain_model.py`
- **Permutation Feature Importance**: Evaluates feature degradation on held-out test data across 20 independent shuffle repeats. Quantifies the empirical accuracy drop when each symptom column is randomly permuted.
- **Score Component Decomposition**: Decomposes candidate $CPS$ values into four transparent additive components: $0.35 S_{\text{target}} + 0.35 S_{\text{potency}} + 0.15 S_{\text{druglike}} + 0.15 S_{\text{evidence}}$.

### 5.7 Module 7: Automated Quality Assurance Test Suite
- **Source Directory**: `tests/`
- **Coverage**: **53 automated unit, integration, and end-to-end tests** across six test modules with 100% passing rate and zero warnings:
  1. `test_emerging_disease_end_to_end.py` (27 tests): Open-world symptom similarity metrics (Jaccard, Weighted Jaccard, Cosine), hallmark symptom identification, threshold gating, candidate evidence retrieval, Lindane toxicant disqualification, Ritonavir host PK adjuvant classification, and Dual-Milestone horizon accounting.
  2. `test_symptom_classifier.py` (5 tests): Raw schemas, deduplication verification, leakage-free train/test partition intersection, model serialization, predictor API.
  3. `test_therapeutic_prioritization.py` (4 tests): Input schemas, Lipinski rule logic, clinical bonus steps, candidate ranking output bounds.
  4. `test_epidemiological_analysis.py` (3 tests): Surveillance schemas, missing-value NaN preservation, synthetic surge detector logic.
  5. `test_timeline_validation.py` (3 tests): FDA timeline schemas, 95%+ matching threshold, evidence classification breakdown.
  6. `test_unit_and_pipeline_fixtures.py` (11 tests): Edge cases for symptom vectorization, feature length alignment, boundary testing for molecular rules, salt-stripping normalization, milestone projection formulas, zero-denominator MAPE stability, flat series surge immunity, and negative control verification.
- **Pipeline Execution Status**: Master orchestrator (`src/run_theralens.py`) executed with **zero fatal errors (exit code 0)** in 37.17 seconds under the documented Python 3.12.10 / scikit-learn 1.9.1 reference environment.

---

## 6. Results and Empirical Evaluation

### 6.1 Symptom Classifier Benchmark Performance
Table 1 summarizes model evaluation on the held-out test set ($N_{\text{test}}=61$ unique presentations) following training on $N_{\text{train}}=243$ unique presentations.

**Table 1: Held-Out Benchmark Performance Across Evaluated Models**

| Candidate Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Logistic Regression (Selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest (100 Trees)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Support Vector Machine (Linear)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Dummy Classifier (Stratified Baseline)** | **0.0164** | **0.0122** | **0.0122** | **0.0122** | **0.0122** | **0.0164** |

Additionally, 5-fold stratified cross-validation on the complete 304 unique patterns confirmed:
- Mean Accuracy: $1.0000 \pm 0.0000$
- Mean Macro F1-Score: $1.0000 \pm 0.0000$

The confusion matrix on the held-out test set is perfectly diagonal ($61/61$ correct predictions across 41 classes). Per-class precision, recall, and F1-score are all 1.00.

*Scientific Context on 100% Benchmark Accuracy*:
The observed 1.0000 Accuracy and Macro F1 on the held-out test set reflects the idealized, rule-based synthetic structure of the public clinical benchmark, where 41 disease profiles are completely linearly separable in a sparse 132-dimensional binary symptom hyperspace. Crucially, as documented in Section 7.2, **this does not establish real-world clinical diagnostic infallibility**. Real hospital patients present with ambiguous, overlapping, and multi-morbid symptoms, where true clinical classification accuracy would be substantially lower (typically 75%–85%). That is why TheraLens pairs this supervised classifier with an open-world multi-metric similarity engine to handle novel, unclassified syndromes.

### 6.2 Model Explainability & Permutation Importance
Table 2 details the top 10 most influential clinical symptom features identified by permutation feature importance across 20 shuffle repeats on the held-out test set.

**Table 2: Top 10 Influential Clinical Symptoms (Permutation Importance on Test Set)**

| Rank | Clinical Symptom Feature | Mean Accuracy Drop ($\Delta \text{Acc}$) | Standard Deviation ($\sigma$) |
| :-: | :--- | :---: | :---: |
| 1 | `vomiting` | **0.0352** | $\pm 0.0118$ |
| 2 | `chills` | **0.0328** | $\pm 0.0104$ |
| 3 | `fatigue` | **0.0279** | $\pm 0.0098$ |
| 4 | `high_fever` | **0.0262** | $\pm 0.0089$ |
| 5 | `headache` | **0.0246** | $\pm 0.0085$ |
| 6 | `abdominal_pain` | **0.0213** | $\pm 0.0076$ |
| 7 | `cough` | **0.0197** | $\pm 0.0068$ |
| 8 | `loss_of_appetite` | **0.0180** | $\pm 0.0062$ |
| 9 | `malaise` | **0.0164** | $\pm 0.0058$ |
| 10 | `sweating` | **0.0148** | $\pm 0.0051$ |

*Scientific Interpretation*: Features such as `vomiting`, `chills`, and `high_fever` serve as critical branching nodes in separating febrile infectious diseases (e.g., Malaria, Dengue, Typhoid) from chronic dermatological or metabolic disorders. Feature importance reflects statistical reliance in the model's decision function, **not biological causality or disease etiology**.

### 6.3 Epidemiological Outbreak Surveillance & Time-Ordered Forecasting
Surveillance harmonization produced 5,426 standardized records across Measles (1974–2024), Yellow Fever (1974–2024), and Cholera (2012–2014). The rolling Z-score detector flagged **224 anomalous outbreak surge events** ($Z \ge 2.0$, $y > 50$).

Table 3 compares time-ordered forecasting models on longitudinal global Measles incidence over the 10-year held-out test period (2015–2024).

**Table 3: Time-Ordered Forecasting Benchmark Comparison (Measles 2015–2024)**

| Forecasting Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) |
| :--- | :---: | :---: | :---: |
| **Naive Persistence Baseline ($\hat{y}_t = y_{t-1}$)** | **53,024.60** | **67,224.23** | **91.95%** |
| **3-Year Moving Average Baseline** | **57,401.27** | **70,034.64** | **105.93%** |
| **Autoregressive Ridge (Lag1 + Lag2 + Trend)** | **155,157.37** | **172,469.68** | **240.19%** |

*Scientific Discussion*: The Naive Persistence Baseline outperforms the Autoregressive Ridge model by nearly $3\times$ on MAE and MAPE. Longitudinal surveillance series exhibit severe non-stationarity driven by national vaccination campaign rollouts and post-2016 geopolitical disruptions. Autoregressive linear trend extrapolation overshoots historical turning points. Preserving this finding demonstrates scientific honesty: complex regression models should not be deployed when simple empirical baselines are superior.

### 6.4 Therapeutic Target & Candidate Prioritization Rankings
TheraLens successfully prioritized **3,636 biological targets** and **25,650 small-molecule candidates** across 37 diseases. Table 4 presents representative top-ranked candidates across major disease categories.

**Table 4: Representative Top-Ranked Therapeutic Candidates Across Selected Diseases**

| Disease Category | Candidate Molecule | ChEMBL ID | Target Symbol | Target Name | $S_{\text{target}}$ | $S_{\text{potency}}$ | $S_{\text{druglike}}$ | $S_{\text{evidence}}$ | Composite Score ($CPS$) |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Malaria** | Chloroquine | CHEMBL428 | `PF3D7_0709000` | CRT Transporter | 0.8800 | 0.9200 | 1.0000 | 1.0000 | **0.9300** |
| **Tuberculosis** | Isoniazid | CHEMBL64 | `inhA` | Enoyl-[acyl-carrier-protein] | 0.8500 | 0.8800 | 1.0000 | 1.0000 | **0.9055** |
| **Diabetes** | Metformin | CHEMBL1431 | `PRKAA1` | AMPK Subunit Alpha 1 | 0.7200 | 0.7500 | 1.0000 | 1.0000 | **0.8145** |
| **Hypertension** | Amlodipine | CHEMBL1489 | `CACNA1C` | Voltage Calcium Channel | 0.6800 | 0.8200 | 1.0000 | 1.0000 | **0.8250** |
| **AIDS (Negative Control)** | Medroxyprogesterone Acetate | CHEMBL717 | `PGR` | Progesterone Receptor | 0.3696 | 0.9700 | 1.0000 | 1.0000 | **0.7689** |

*Score Decomposition of AIDS Negative Control*:

$$CPS = 0.35 \cdot 0.3696 + 0.35 \cdot 0.9700 + 0.15 \cdot 1.0000 + 0.15 \cdot 1.0000 = 0.1294 + 0.3395 + 0.1500 + 0.1500 = \mathbf{0.7689}$$

Medroxyprogesterone Acetate achieves a high score because of strong in-vitro binding affinity ($p\text{ChEMBL} = 9.82$) to the Progesterone Receptor and FDA-approved status. However, it possesses no clinical anti-retroviral efficacy against HIV-1. This finding serves as empirical proof that $CPS$ is an in-silico screening heuristic that must never replace biomedical clinical judgment.

*Biological Target Context & Indication Evidence Stratification*:
Beyond generic chemical tractability ($CPS$), candidates are evaluated across three orthogonal dimensions:
1. **Target Relevance**: Distinguishing pathogen-directed curative targets from human host metabolic enzymes (e.g., Ritonavir targets host `CYP3A5` and `SLCO1B1` as a pharmacokinetic booster to enhance partner antimalarials, rather than acting as a standalone antiparasitic curative agent).
2. **Indication Evidence Tiers**: Enforcing strict hierarchical stratification (Tier 1: Approved; Tier 2: Investigational with indication trial history; Tier 3: Indirect target hit without indication trial history; Tier 4: Disqualified toxicant such as Lindane).
3. **Compound Suitability**: Segregating curative monotherapies, adjuvants/PK enhancers, and disqualified non-therapeutic chemicals.

### 6.5 Master Timeline Audit & Cartesian Join Disambiguation
Streaming audit of the 114 MB master timeline dataset revealed:
- **Total Master Timeline Rows**: 555,323 records.
- **Unique Disease-Molecule-Milestone Triplets**: 30,430 unique combinations.
- **Cartesian Duplication Factor**: **$18.25\times$** expansion resulting from multiple trade names, application numbers, dosage forms, and routes of administration per active chemical ingredient. Crucially, these rows represent product packaging and formulation combinations, **not 555,323 independent clinical trials**.
- **Evidence Stratification**:
  - Phase 4 Verified Indication Approvals: **72,277 records (13.02%)**
  - Investigational Clinical Trials (Phases 1–3): **483,043 records (86.98%)**
  - Preclinical Research Compounds: **3 records (<0.01%)**
- **Regulatory Caveat**: The recorded `first_approval_year` reflects the molecule's initial FDA marketing authorization, **not proof of regulatory approval for every associated secondary disease indication**.

### 6.6 Biopharmaceutical Milestone Latency Projections & Dual-Milestone Horizons
Modeling clinical progression latencies across 37 diseases with active candidates yielded realistic decision horizons between **2026 and 2042** (median latency: 2 to 7 years depending on current phase maturity). 

**Table 5: Milestone Projections for Representative Disease Pipelines**

| Evaluated Disease | Top Active Clinical Candidate | Current Max Phase | Projected Milestone Window | 80% Uncertainty Window | Estimated Phase-to-Approval Success Probability |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Malaria** | Chloroquine / Artemisinin Derivatives | Phase 4 | **2027.5** | 2027 – 2028 | 90.0% (Supplemental NDA) |
| **Dengue** | Investigational Antivirals | Phase 2 | **2034.5** | 2031 – 2038 | 18.3% (Phases 2 $\to$ 4) |
| **Chagas Disease** | Benznidazole Analogs | Phase 3 | **2031.3** | 2029 – 2034 | 52.2% (Phases 3 $\to$ 4) |
| **Emerging Arbovirus** | Screened Hit Compound | Preclinical | **2039.0** | 2034 – 2046 | 4.5% (Preclinical $\to$ Approval) |

*Dual-Milestone Horizon Accounting for Investigational Candidates*:
When evaluating investigational candidates with historical clinical trial records (e.g., Phase 1 indication records in ChEMBL), static databases record the highest reached development phase but cannot verify active recruitment, trial termination, or positive clinical efficacy. TheraLens formalizes this epistemic boundary through Dual-Milestone Scenario Modeling:
- **Horizon A (Direct Phase Advancement / Optimistic Continuation)**: Assumes historical Phase 1 safety is accepted by regulators, projecting progression directly through Phase 2 proof-of-concept, Phase 3, and FDA Review (8.5 years baseline; ~2034).
- **Horizon B (Clinical Re-Validation / Conservative In-Progress)**: Accounts for the requirement to repeat or validate Phase 1 safety and pharmacokinetic parameters in the emerging disease patient cohort before entering Phase 2 (10.6 years baseline; ~2037).
- **Scenario Sensitivity Bounds**: Neither horizon constitutes a deterministic forecast; both serve as literature-informed scenario simulations (DiMasi et al., Hay et al.) with explicit sensitivity bounds (+/- 30-35% dispersion) and hypothetical technology acceleration scenarios (15% and 30% duration reductions).

*Methodological Note on Projections*:
Projected decision windows represent heuristic latency projections based on literature phase duration distributions (DiMasi et al., Hay et al.), **not guaranteed approval dates or statistically calibrated confidence intervals**.

*Critical Demystification of the Year "2114" Headline*:
The year "2114" was introduced in the initial project proposal as an illustrative, symbolic horizon highlighting the century-scale quest for elusive therapies against intractable human diseases. Our empirical audit confirms that:
1. No supervised machine learning model can be trained on an arbitrary target year 88 years into the future without fabricating training labels.
2. Industry clinical transition models demonstrate that biopharmaceutical pipeline progression is bounded to 10–15 year operational horizons.
3. The year 2114 is retained strictly as an inspirational educational headline and is explicitly segregated from empirical machine learning predictions.

---

## 7. Conclusion and Future Work

### 7.1 Summary of Implemented Achievements (100% Completion)
The TheraLens project has reached 100% completion against all Applied Machine Learning Review-3 criteria:
1. **End-to-End Orchestration**: Seamless integration of clinical presentation, genomic associations, chemical bioactivity, epidemiological trends, and regulatory history into a unified pipeline executing in ~37 seconds.
2. **Methodological Rigor**: Eradicated 93.87% synthetic row duplication, prevented data leakage, and implemented stratified cross-validation.
3. **Transparent Heuristic Prioritization**: Formulated the Composite Priority Score ($CPS$), verified its mathematical bounds, and documented negative controls (Medroxyprogesterone Acetate) and host adjuvants (Ritonavir).
4. **Epidemiological Integrity**: Demonstrated why persistence baselines outperform complex autoregressive models under structural non-stationarity.
5. **Regulatory Disambiguation**: Resolved the $18.25\times$ Cartesian expansion in regulatory join data and separated Phase 4 approvals from investigational trials.
6. **Software Quality & Testing**: Achieved 100% passing rate across an expanded **53-test pytest suite** covering unit, fixture, and end-to-end integration workflows with zero warnings, coupled with native scikit-learn environment documentation (Python 3.12.10 / scikit-learn 1.9.1) and transparent version diagnostics.

### 7.2 Limitations
1. *Synthetic Clinical Benchmark*: The 304 unique symptom presentations are synthetic and linearly separable; real-world clinical presentations exhibit probabilistic ambiguity, variable severity, and multi-morbidity.
2. *In-Vitro vs In-Vivo Gap*: $CPS$ relies on in-vitro binding assays ($p\text{ChEMBL}$) and Lipinski compliance; it does not model in-vivo pharmacokinetics, ADMET toxicity, metabolic clearance, or blood-brain barrier penetration.
3. *Regulatory Indication Specificity*: FDA `first_approval_year` applies to the active ingredient's initial market authorization, not secondary label expansions.

### 7.3 Realistic Future Directions
- **Electronic Health Record (EHR) Validation**: Validate symptom classifiers against real-world de-identified hospital presentation datasets (e.g., MIMIC-IV).
- **Structure-Based Deep Learning**: Incorporate 3D molecular graph neural networks (GNNs) and AlphaFold-predicted target binding pocket coordinates.
- **Deep Epidemiological Transformers**: Implement Spatio-temporal Graph Neural Networks incorporating climate variables, human mobility, and local vaccination coverage.

---

## 8. References

1. DiMasi, J. A., Hansen, R. W., & Grabowski, H. G. (2003). The price of innovation: new estimates of drug development costs. *Journal of Health Economics*, 22(2), 151-185. [https://doi.org/10.1016/S0167-6296(02)00126-1](https://doi.org/10.1016/S0167-6296(02)00126-1)
2. DiMasi, J. A., Grabowski, H. G., & Hansen, R. W. (2016). Innovation in the pharmaceutical industry: New estimates of R&D costs. *Journal of Health Economics*, 47, 20-33. [https://doi.org/10.1016/j.jhealeco.2016.01.012](https://doi.org/10.1016/j.jhealeco.2016.01.012)
3. Hay, M., Thomas, D. W., Craighead, J. L., Economides, C., & Rosenthal, J. (2014). Clinical development success rates for investigational drugs. *Nature Biotechnology*, 32(1), 40-51. [https://doi.org/10.1038/nbt.2786](https://doi.org/10.1038/nbt.2786)
4. Lipinski, C. A., Lombardo, F., Dominy, B. W., & Feeney, P. J. (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Advanced Drug Delivery Reviews*, 46(1-3), 3-26. [https://doi.org/10.1016/S0169-409X(00)00129-0](https://doi.org/10.1016/S0169-409X(00)00129-0)
5. Ochoa, D., Hercules, A., Carmona, M., Suveges, D., Gonzalez-Uriarte, A., Malangone, C., ... & Ghoussaini, M. (2021). The next-generation Open Targets Platform: reimagined, redesigned, and upgraded. *Nucleic Acids Research*, 49(D1), D1302-D1310. [https://doi.org/10.1093/nar/gkaa1026](https://doi.org/10.1093/nar/gkaa1026)
6. Mendez, D., Gaulton, A., Bento, A. P., Chambers, J., De Veij, M., Félix, E., ... & Leach, A. R. (2019). ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930-D940. [https://doi.org/10.1093/nar/gky1075](https://doi.org/10.1093/nar/gky1075)
7. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
8. U.S. Food and Drug Administration (FDA). (2024). Drugs@FDA Data Files: Approved Drug Products with Therapeutic Equivalence Evaluations. [https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files](https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files)
9. Veber, D. F., Johnson, S. R., Cheng, H. Y., Smith, B. R., Ward, K. W., & Kopple, K. D. (2002). Molecular properties that influence the oral bioavailability of drug candidates. *Journal of Medicinal Chemistry*, 45(12), 2615-2623. [https://doi.org/10.1021/jm020017n](https://doi.org/10.1021/jm020017n)
10. Wong, C. H., Siah, K. W., & Lo, A. W. (2019). Estimation of clinical trial success rates and related parameters. *Biostatistics*, 20(2), 273-286. [https://doi.org/10.1093/biostatistics/kxx069](https://doi.org/10.1093/biostatistics/kxx069)
11. World Health Organization (WHO). (2024). Global and Regional Measles and Rubella Surveillance Data. [https://www.who.int/teams/immunization-vaccines-and-biologicals/immunization-analysis-and-insights/surveillance/surveillance-data-and-reports](https://www.who.int/teams/immunization-vaccines-and-biologicals/immunization-analysis-and-insights/surveillance/surveillance-data-and-reports)
12. Wouters, O. J., McKee, M., & Luyten, J. (2020). Estimated research and development investment needed to bring a new medicine to market, 2009-2018. *JAMA*, 323(9), 844-853. [https://doi.org/10.1001/jama.2020.1166](https://doi.org/10.1001/jama.2020.1166)
