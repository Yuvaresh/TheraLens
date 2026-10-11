# TheraLens: Applied Machine Learning Framework for Disease Classification, Therapeutic Prioritization, and Milestone Analysis

**College Review-2 Implementation & Technical Report**  
**Academic Year**: 2026–2027  
**Department**: Computer Science & Engineering / Applied Machine Learning  
**Project**: TheraLens  

---

## 1. Abstract

The discovery and development of therapeutic interventions against emerging and existing pathologies is an intricate, multi-decadal process characterized by high clinical attrition rates and fragmented biomedical evidence. The **TheraLens** project introduces a unified, modular Applied Machine Learning architecture designed to synthesize multi-domain biomedical data across four foundational pillars:
1. Clinical presentation profiles for symptom-based disease classification;
2. Global longitudinal epidemiological surveillance for outbreak surge detection;
3. Genome-scale target associations and in-vitro chemical bioactivity for therapeutic candidate prioritization; and
4. Historical regulatory drug approval timelines for biopharmaceutical progression milestone forecasting.

For the College Review-2 evaluation, the project transitioned from exploratory data ingestion to a functional, empirical machine learning implementation. Addressing critical data duplication in clinical records (93.87% synthetic duplication), a leakage-resistant stratified modeling pipeline was established on 304 unique clinical patterns, achieving 1.0000 Accuracy and 1.0000 Macro F1 across 41 disease classes with full model persistence. A transparent multi-criteria candidate prioritization engine ranked 3,636 biological targets and 25,650 candidate compounds by combining Open Targets genetic evidence, pChEMBL bioactivity potency, and Lipinski physicochemical drug-likeness. Longitudinal surveillance across 5,426 WHO records established outbreak surge detection and time-ordered forecasting benchmarks. Finally, rigorous regulatory auditing of 555,323 master timeline entries clarified the boundary between initial active-ingredient approval and indication-specific milestones, framing an experimental biopharmaceutical pipeline progression prototype. The entire framework is validated through an automated 15-test suite (100% pass rate) and orchestrated via a command-line interface.

---

## 2. System Architecture & UML Design

The implemented TheraLens architecture decouples data ingestion, preprocessing, predictive modeling, prioritization, and explainability into modular, loosely coupled subsystems.

### 2.1 Master Architecture Diagram (Component & Data Flow)

```mermaid
graph TD
    subgraph Data_Layer ["1. Biomedical Data Layer"]
        D1["symptom_disease.csv<br/>(4,961 raw rows / 41 classes)"]
        D2["WHO Surveillance<br/>(Measles, Yellow Fever, Cholera)"]
        D3["Open Targets<br/>(therapeutic_associations.csv)"]
        D4["ChEMBL Bioactivity & Molecules<br/>(chembl_activity.csv, chembl_molecules.csv)"]
        D5["FDA Drugs@FDA & Indications<br/>(fda_drug_approval_timeline.csv, indications)"]
    end

    subgraph Preprocessing_Layer ["2. Audit & Preprocessing Pipeline"]
        P1["Deduplication Engine<br/>(304 Unique Symptom Patterns)"]
        P2["Wide-to-Long Standardizer<br/>(Longitudinal Harmonization)"]
        P3["Entity Match Audit<br/>(FDA-ChEMBL Normalization)"]
    end

    subgraph ML_Modeling_Layer ["3. Machine Learning & Prioritization Engines"]
        M1["Symptom Classifier<br/>(Logistic Regression, RF, SVM, Dummy)<br/>src/models/train_symptom_classifier.py"]
        M2["Inference Predictor API<br/>src/inference/predict_disease.py"]
        M3["Candidate Prioritization Engine<br/>(CPS Multi-Criteria Scoring)<br/>src/models/prioritize_therapeutic_candidates.py"]
        M4["Epidemiological Surge & Forecast<br/>src/analysis/analyze_epidemiological_trends.py"]
        M5["Timeline Audit & Latency Forecast<br/>src/analysis/validate_therapeutic_timeline.py<br/>src/analysis/forecast_therapeutic_timeline.py"]
    end

    subgraph Explainability_Testing ["4. Explainability & Quality Assurance"]
        E1["Permutation Feature Importance<br/>src/analysis/explain_model.py"]
        E2["Transparent Score Decomposition"]
        T1["Automated Pytest Suite<br/>tests/ (15 Unit/Integration Tests)"]
    end

    subgraph Output_Layer ["5. Artifact & Persistence Layer"]
        O1["Persisted Pipelines<br/>models/symptom_classifier.joblib"]
        O2["Ranked Datasets<br/>data/processed/*.csv"]
        O3["Evaluation Figures & Reports<br/>outputs/ (20 Publication Charts)"]
    end

    D1 --> P1 --> M1 --> O1
    M1 --> M2
    M1 --> E1 --> O3
    D2 --> P2 --> M4 --> O2
    M4 --> O3
    D3 & D4 --> M3 --> O2
    M3 --> E2 --> O3
    D5 --> P3 --> M5 --> O2
    M5 --> O3
    M1 & M3 & M4 & M5 --> T1
```

### 2.2 Orchestration Flow (UML Sequence)

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / User
    participant CLI as src/run_theralens.py
    participant Clf as SymptomClassifier
    participant Prio as CandidatePrioritizer
    participant Epi as EpidemiologicalEngine
    participant Time as TimelineValidator
    participant Exp as ExplainabilityModule

    User->>CLI: Execute Pipeline (python src/run_theralens.py)
    CLI->>CLI: Validate Prerequisite Files
    CLI->>Clf: Train & Evaluate on Deduplicated Patterns
    Clf-->>CLI: Model Persisted (models/symptom_classifier.joblib)
    CLI->>Prio: Join Targets, Activity & Molecules (Compute CPS)
    Prio-->>CLI: Ranked Targets & Candidates Saved
    CLI->>Epi: Aggregate Time Series & Detect Outbreak Surges
    Epi-->>CLI: Trends Saved & Forecast Benchmarked
    CLI->>Time: Chunked Audit of Master Timeline
    Time-->>CLI: Regulatory Classification & Quality Report Saved
    CLI->>Exp: Compute Permutation Importance & Score Decomposition
    Exp-->>CLI: Interpretability Charts Generated
    CLI-->>User: Master Execution Dashboard & Summary JSON
```

---

## 3. Analysis and Preprocessing of Dataset

Table 3.1 details the preprocessing transformations executed across the twelve primary project datasets, establishing the input foundations for machine learning.

### Table 3.1: Preprocessing Operations & Data Quality Audits

| Dataset | Raw State | Preprocessing Operation | Clean State | Quality Issue Resolved |
| :--- | :--- | :--- | :--- | :--- |
| **Disease-Symptom** | 4,961 rows x 133 cols | Feature vector deduplication | 304 unique patterns | Filtered 4,657 synthetic duplicate rows (93.87%) to eliminate data leakage. |
| **WHO Measles** | 1,692 rows x 9 cols | Standardized indicator & numeric casting | 1,692 records | Resolved reporting nulls; preserved legitimate absence of reporting. |
| **WHO Yellow Fever** | 874 rows x 9 cols | Standardized indicator & numeric casting | 874 records | Handled sparse reporting across 47 endemic nations. |
| **WHO Cholera (2012-14)**| 55 rows x 70 cols (wide) | Melted weekly matrix into long format | 2,860 weekly records | Converted wide 52-week columns to normalized longitudinal time series. |
| **Open Targets** | 3,636 rows x 7 cols | Validated target symbols & association bounds | 3,636 associations | Ranked targets within 38 diseases (scores: 0.0015 to 0.9250). |
| **ChEMBL Bioactivity** | 8,023 rows x 10 cols | Verified measured pChEMBL logarithmic affinity | 8,023 potency records| Standardized heterogeneous assay units (IC50, Ki, EC50) into logarithmic pChEMBL. |
| **ChEMBL Molecules** | 6,319 rows x 11 cols | Evaluated MW, AlogP, PSA, HBA, HBD | 6,319 characterized molecules | Imputed missing properties with neutral baseline and audit flag. |
| **FDA Approval Timeline**| 12,243 rows x 12 cols | Normalized active ingredients; extracted year | 3,142 unique active ingredients | Flagged 749 records lacking explicit first approval dates in source Drugs@FDA tables. |
| **FDA-ChEMBL Match** | 3,142 ingredients | Multi-stage dictionary & salt-stripping match | 3,059 matched ingredients | Achieved 97.36% resolution rate; audited 83 unmatched entities. |
| **Master Timeline** | 555,323 rows x 14 cols | Chunked entity join with Phase 4 auditing | 30,430 unique disease-drug pairs | Resolved $18.25\times$ Cartesian duplication from multi-dosage products. |

---

## 4. Implementation

### 4.1 System Modules & Source Structure
The Review-2 codebase is organized into modular Python packages:
- `src/models/train_symptom_classifier.py`: Deduplication, stratified splitting, model benchmarking, and artifact persistence.
- `src/models/evaluate_symptom_classifier.py`: Standalone verification of persisted model pipelines.
- `src/inference/predict_disease.py`: Reusable inference interface with clinical research disclaimers.
- `src/models/prioritize_therapeutic_candidates.py`: Multi-criteria composite priority scoring engine.
- `src/analysis/analyze_epidemiological_trends.py`: Longitudinal surveillance aggregation, rolling Z-score surge detection, and time-ordered forecasting.
- `src/analysis/validate_therapeutic_timeline.py`: Chunked regulatory timeline auditing and evidence classification.
- `src/analysis/forecast_therapeutic_timeline.py`: Experimental biopharmaceutical progression latency model.
- `src/analysis/explain_model.py`: Permutation feature importance and candidate contribution breakdown.
- `src/run_theralens.py`: End-to-end command-line orchestrator.
- `tests/`: Automated unit and integration testing suite.

### 4.2 Module 1: Symptom-Based Disease Classification
- **Algorithms Evaluated**: DummyClassifier (Stratified Baseline), Logistic Regression (L2 regularization, $C=1.0$), Random Forest Classifier ($N=100$ estimators, max depth 15), and Linear Support Vector Machine.
- **Leakage Prevention**: Evaluated strictly on the deduplicated 304 unique clinical presentations using stratified 80/20 train/test splitting ($N_{\text{train}}=243, N_{\text{test}}=61$) with fixed random seed ($42$). Because minimum pattern frequency per class is 5, every single class is represented in both training and testing.
- **Model Persistence**: Serialized best pipeline to `models/symptom_classifier.joblib` alongside feature names (`models/feature_names.json`) and target classes (`models/class_labels.json`).

### 4.3 Module 2: Therapeutic Candidate Prioritization
- **Scoring Formula**:
  $$\text{CPS} = 0.35 \times S_{\text{target}} + 0.35 \times S_{\text{potency}} + 0.15 \times S_{\text{druglike}} + 0.15 \times S_{\text{evidence}}$$
  Where:
  1. $S_{\text{target}} = \text{Open Targets association score} \in [0.0, 1.0]$.
  2. $S_{\text{potency}} = \text{clip}\left(\frac{\text{median}(p\text{ChEMBL}) - 4.0}{10.0 - 4.0}, 0.0, 1.0\right)$ where $\text{pChEMBL} = -\log_{10}(\text{Molar Concentration})$ represents standardized negative logarithmic molar potency across supported activity endpoints ($\text{IC}_{50}$, $K_i$, $\text{EC}_{50}$, $K_d$).
  3. $S_{\text{druglike}} = \frac{\sum \text{passed Lipinski / Veber rules}}{5.0} \in [0.0, 1.0]$ (neutral $0.5$ if missing).
  4. $S_{\text{evidence}} = \text{Global drug-development maturity score} \in [0.10, 1.00]$ based on the molecule's highest worldwide recorded clinical phase (`max_phase` in ChEMBL across any indication); does not establish approval or clinical efficacy for the specific disease being queried.
- **Outputs**: Generates `data/processed/disease_target_rankings.csv` (3,636 targets) and `data/processed/therapeutic_candidate_rankings.csv` (25,650 candidates).

### 4.4 Module 3: Epidemiological Trend Analysis
- **Surge Detection**: Identifies outbreak spikes using shifted rolling Z-score:
  $$Z_t = \frac{Y_t - \mu_{t-1..t-k}}{\sigma_{t-1..t-k}}$$
  An anomaly is flagged when $Z_t \ge 2.0$ and reported cases exceed 50.
- **Temporal Forecasting**: Chronologically partitioned Measles surveillance into training (1974–2014, 41 years) and held-out test (2015–2024, 10 years). Compares Naive Persistence ($\hat{y}_t = y_{t-1}$), 3-Year Moving Average, and Autoregressive Ridge Regression.

### 4.5 Module 4: Therapeutic Timeline Validation & Forecasting
- **Scientific Audit**: Classifies records into Verified Phase 4 Milestones, Investigational Indication Records (Phase 1–3), and Preclinical Associations (joined rows reflect formulation expansion, not unique clinical trials).
- **Progression Latency Modeling**: Implements benchmark transition latencies (DiMasi et al.): Phase 1 (2.1 yrs), Phase 2 (3.2 yrs), Phase 3 (3.8 yrs), FDA NDA Review (1.5 yrs). Literature-derived median durations are coupled with prototype modeling assumptions (heuristic $\pm 35\%$ uncertainty intervals and sequential progression) to project empirical milestone windows (2028–2042) for 37 diseases.

### 4.6 Module 5: Model Explainability
- Evaluates permutation feature importance on held-out test data across 20 repeats.
- Decomposes exact mathematical component contributions for top candidate molecules.

---

## 5. Empirical Results

### 5.1 Symptom Classifier Benchmark Performance
Evaluated on the held-out test set ($N=61$ unique patterns across 41 classes):

| Model Architecture | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Support Vector Machine** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Dummy (Stratified Baseline)** | 0.0164 | 0.0122 | 0.0122 | 0.0122 | 0.0122 |

![Model Comparison](../outputs/model_evaluation/model_performance_comparison.png)  
*Figure 5.1: Comparative benchmark evaluation on the held-out test set.*

![Confusion Matrix](../outputs/model_evaluation/confusion_matrix.png)  
*Figure 5.2: Confusion matrix showing 100% correct classification on held-out test patterns.*

### 5.2 Therapeutic Candidate Prioritization Results
Top prioritized molecules identified across representative disease categories:
- **AIDS**: Medroxyprogesterone Acetate ($\text{CPS} = 0.7689$, Target: AR, pChEMBL: 9.77)
- **Diabetes**: Sitagliptin ($\text{CPS} = 0.7621$, Target: DPP4, pChEMBL: 8.85)
- **Hypertension**: Amlodipine ($\text{CPS} = 0.7554$, Target: CACNA1C, pChEMBL: 9.10)
- **Malaria**: Atovaquone ($\text{CPS} = 0.7431$, Target: CYTB, pChEMBL: 8.90)

![Top Candidates](../outputs/therapeutic_prioritization/top_candidate_scores.png)  
*Figure 5.3: Highest-ranked candidate molecules across representative disease categories.*

![Score Breakdown](../outputs/therapeutic_prioritization/score_component_breakdown.png)  
*Figure 5.4: Transparent multi-attribute scoring decomposition for top candidates.*

### 5.3 Epidemiological Outbreak & Forecasting Results
- **Outbreak Surges Detected**: 224 statistically verified surge anomalies across 5,426 records.
- **Forecasting Performance (2015–2024 Test Horizon)**:
  - Naive Persistence: $\text{MAE} = 53,024.60, \text{RMSE} = 67,224.23, \text{MAPE} = 91.95\%$
  - 3-Year Moving Average: $\text{MAE} = 57,401.27, \text{RMSE} = 70,034.64, \text{MAPE} = 105.93\%$
  - Autoregressive Ridge: $\text{MAE} = 155,157.37, \text{RMSE} = 172,469.68, \text{MAPE} = 240.19\%$

![Forecasting Comparison](../outputs/epidemiological_analysis/forecasting_model_comparison.png)  
*Figure 5.5: Time-ordered forecasting comparison on the held-out test horizon.*

### 5.4 Timeline Audit & Regulatory Findings
- Total Timeline Rows: 555,323
- Unique Disease-Molecule Pairs: 30,430
- Verified Phase 4 Milestone Rows: 72,277 (13.02%)
- Investigational Indication Records: 483,043 (86.98%)
- Preclinical / Unverified Records: 3 (0.00%)

![Timeline Classification](../outputs/timeline_analysis/timeline_evidence_classification.png)  
*Figure 5.6: Evidence classification audit separating Phase 4 approvals from investigational indication records.*

---

## 6. References

1. **Ochoa, D., et al.** (2021). The next-generation Open Targets Platform: reimagined, redesigned, rebuilt. *Nucleic Acids Research*, 49(D1), D1302–D1310.
2. **Mendez, D., et al.** (2019). ChEMBL: towards direct deposition of bioassay data. *Nucleic Acids Research*, 47(D1), D930–D940.
3. **DiMasi, J. A., Grabowski, H. G., & Hansen, R. W.** (2016). Innovation in the pharmaceutical industry: New estimates of R&D costs. *Journal of Health Economics*, 47, 20–33.
4. **Lipinski, C. A., et al.** (2001). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. *Advanced Drug Delivery Reviews*, 46(1–3), 3–26.
5. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
6. **World Health Organization (WHO)**. (2024). Global Health Observatory: Measles and Yellow Fever surveillance data.
