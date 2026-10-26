# TheraLens: University Review-3 Viva Voce & Technical Defense Guide

**Project Title:** TheraLens: Machine Learning-Based Forecasting of Emerging Diseases and Future Therapeutic Development  
**Academic Milestone:** VIT Applied Machine Learning — Review-3 (100% Final Evaluation)  
**Authors:** S. Yuvaresh (`24BCS0195`), M. Gopika (`24BCA0243`)  
**Faculty Supervisor:** Dr. M. Parimala, Professor, Department of Computer Science and Engineering  
**Academic Year:** 2026–2027  

---

## 1. 60-Second Elevator Pitch

> *"TheraLens is an Applied Machine Learning biomedical framework that bridges the gap between clinical presentations, epidemiological outbreaks, genomic drug targets, chemical bioactivity, and regulatory approval timelines. When a patient presents with symptoms, our leakage-free multiclass model predicts the disease; the system then queries Open Targets genomic evidence and ChEMBL in-vitro bioactivity to rank therapeutic small-molecule candidates using a transparent Composite Priority Score ($CPS$). Simultaneously, it tracks global longitudinal WHO surveillance curves to detect outbreak surges and models realistic biopharmaceutical progression timelines based on empirical clinical phase transition latencies. Crucially, TheraLens rejects black-box ungrounded extrapolations—such as the symbolic proposal headline '2114'—in favor of mathematically bounded, scientifically honest, and reproducible machine learning."*

---

## 2. Problem Statement & Core Objectives

### The Core Problem: The Siloed Biomedical Pipeline
Biopharmaceutical drug development takes **10–15 years**, costs **$1.3B–$2.6B**, and suffers from an **86%–89% clinical failure rate**. Despite massive open-access datasets, knowledge is trapped in disconnected departmental silos:
- Clinical diagnostic checklists (symptoms)
- WHO epidemiological incidence curves (cases over time)
- Open Targets functional genomics (gene-disease links)
- ChEMBL chemoinformatics (in-vitro binding assays and chemistry)
- Drugs@FDA regulatory archives (marketing approvals)

### The 4 Core Objectives
1. **Clinical Presentation Classification**: Predict pathology from presented binary symptoms on a clean, deduplicated clinical manifold with zero train/test leakage.
2. **Multi-Attribute Candidate Prioritization**: Formulate a transparent, multi-criteria mathematical score ($CPS$) ranking biological targets and small-molecule chemical candidates.
3. **Surveillance & Anomaly Forecasting**: Standardize longitudinal epidemiological data, detect outbreak surges via rolling shifted Z-scores, and evaluate chronological time-series forecasting.
4. **Regulatory Audit & Latency Modeling**: Delineate verified approvals from investigational trials across 555k master timeline records, audit Cartesian join expansions, and model realistic pipeline latencies based on industry transition benchmarks.

---

## 3. System Architecture & Data Flow

```
[Clinical Symptoms Input] ──> [predict_disease.py] ──> [Logistic Regression (41 Classes)]
                                                              │
[Predicted Pathology] <───────────────────────────────────────┘
       │
       ├─► [Open Targets Platform] ──► Biological Drug Targets (3,636 ranked)
       │                                     │
       ├─► [ChEMBL Database] ────────► Bioactivity Potency & Lipinski Properties
       │                                     │
       │                                     ▼
       │                          [Composite Priority Score (CPS)]
       │                          25,650 Candidates Ranked Across 37 Diseases
       │
       ├─► [WHO Surveillance] ───────► Rolling Shifted Z-Score (224 Surges Flagged)
       │                               Time-Ordered Forecasts (Persistence > Ridge)
       │
       └─► [FDA Drugs@FDA] ──────────► Master Timeline Audit (18.25x Cartesian Expansion)
                                       Biopharma Transition Latency Model (2026–2042)
```

---

## 4. Dataset Inventory & Preprocessing Summary

| Dataset Name | Source | Raw Shape | Clean / Processed Shape | Key Preprocessing Performed |
| :--- | :--- | :---: | :---: | :--- |
| **Symptom-Disease** | Public Benchmark | 4,961 $\times$ 133 | 304 $\times$ 133 | Collapsed 4,657 identical rows (93.87% duplicates) to 304 unique clinical patterns |
| **WHO Measles** | World Health Org | 10,724 $\times$ 4 | 5,426 records | Standardized annual long format; preserved missing reporting as NaN |
| **WHO Yellow Fever**| World Health Org | 5,200 $\times$ 4 | (combined) | Harmonized annual surveillance |
| **WHO Cholera** | World Health Org | 3 annual tables | (combined) | Melted wide format (Weeks 1–52 columns) into standardized long rows |
| **Therapeutic Assoc**| Open Targets | 3,636 $\times$ 6 | 3,636 $\times$ 7 | Filtered target symbols; ranked targets per disease by association score |
| **ChEMBL Activity** | EMBL-EBI | 8,023 $\times$ 10 | 25,650 aggregated | Filtered pChEMBL values; median potency calculated per target-molecule pair (from 22,238 raw query records) |
| **ChEMBL Molecules**| EMBL-EBI | 6,319 $\times$ 11 | 6,319 $\times$ 11 | Calculated Lipinski & Veber drug-likeness rules compliance (from 8,983 candidate compounds) |
| **FDA Approval Timeline**| Drugs@FDA | 12,243 $\times$ 6 | 12,243 $\times$ 6 | Normalized 3,142 active ingredients; stripped salt suffixes (3,059 matched to ChEMBL, 97.36%) |
| **Master Timeline** | Integrated | 555,323 $\times$ 11 | 30,430 unique triplets| Resolved 18.25x Cartesian expansion; separated Phase 4 from Phase 1–3 trials |

---

## 5. Mathematical Formulations & Algorithms

### 5.1 Composite Priority Score ($CPS$)
$$CPS = 0.35 \cdot S_{\text{target}} + 0.35 \cdot S_{\text{potency}} + 0.15 \cdot S_{\text{druglike}} + 0.15 \cdot S_{\text{evidence}}$$

- **Target Association Score ($S_{\text{target}}$)**:
  $$S_{\text{target}} = \text{clip}(\text{association\_score}, 0.0, 1.0)$$
- **Potency Score ($S_{\text{potency}}$)**:
  $$S_{\text{potency}} = \text{clip}\left(\frac{\text{median}(p\text{ChEMBL}) - 4.0}{10.0 - 4.0}, 0.0, 1.0\right)$$
  *(Maps $100\,\mu\text{M}$ [low potency] to $0.0$, and $0.1\,\text{nM}$ [high potency] to $1.0$)*
- **Drug-Likeness Score ($S_{\text{druglike}}$)**:
  Fraction of 5 Lipinski & Veber rules passed ($MW \le 500$, $-0.4 \le \text{ALogP} \le 5.0$, $PSA \le 140$, $HBA \le 10$, $HBD \le 5$). Missing data receives neutral $0.50$.
- **Clinical Advancement Score ($S_{\text{evidence}}$)**:
  Phase 4 = $1.00$, Phase 3 = $0.80$, Phase 2 = $0.60$, Phase 1 = $0.40$, Preclinical = $0.20$, Missing = $0.10$.

### 5.2 Outbreak Surge Anomaly (Rolling Shifted Z-Score)
To prevent lookahead data leakage, the rolling mean and standard deviation exclude the current evaluation year $t$:
$$\mu_{t} = \frac{1}{k}\sum_{i=1}^{k} y_{t-i}, \quad \sigma_{t} = \sqrt{\frac{1}{k-1}\sum_{i=1}^{k} (y_{t-i} - \mu_t)^2}, \quad Z_t = \frac{y_t - \mu_t}{\sigma_t}$$
An outbreak surge is flagged when $Z_t \ge 2.0$ and reported cases $y_t > 50$ (window $k=5$ years).

### 5.3 Biopharmaceutical Progression Model
$$\text{Projected Latency} = \sum_{p \in \text{Remaining Phases}} \tau_p$$
- $\tau_1 = 2.1\,\text{yr}$ (Phase 1, $P=0.59$)
- $\tau_2 = 3.2\,\text{yr}$ (Phase 2, $P=0.35$)
- $\tau_3 = 3.8\,\text{yr}$ (Phase 3, $P=0.58$)
- $\tau_4 = 1.5\,\text{yr}$ (FDA NDA Review, $P=0.90$)
- Cumulative Phase 1 to Approval Probability: $\prod P_p \approx 11.3\%$.

---

## 6. Empirical Results Summary

- **Symptom Classification ($N_{\text{test}}=61$)**:
  - Logistic Regression: Accuracy = **1.0000**, Macro F1 = **1.0000**
  - Random Forest: Accuracy = **1.0000**, Macro F1 = **1.0000**
  - Linear SVM: Accuracy = **1.0000**, Macro F1 = **1.0000**
  - Dummy Baseline: Accuracy = **0.0164**, Macro F1 = **0.0122**
  - 5-Fold Stratified CV on $N=304$: $1.0000 \pm 0.0000$
- **Epidemiological Forecasting (Measles 2015–2024, $N=10$ yrs)**:
  - Naive Persistence: $\text{MAE} = \mathbf{53,024.60}, \text{RMSE} = \mathbf{67,224.23}, \text{MAPE} = \mathbf{91.95\%}$
  - 3-Year Moving Average: $\text{MAE} = 57,401.27, \text{RMSE} = 70,034.64, \text{MAPE} = 105.93\%$
  - Autoregressive Ridge: $\text{MAE} = 155,157.37, \text{RMSE} = 172,469.68, \text{MAPE} = 240.19\%$
- **Surveillance Anomalies**: 224 outbreak surges detected across 5,426 records.
- **Candidate Prioritization**: 3,636 targets and 25,650 candidates ranked across 37 diseases.
- **Master Timeline Breakdown**: 555,323 rows ($18.25\times$ Cartesian join expansion) $\to$ 72,277 Phase 4 approved indication combinations vs 483,043 investigational trial records across 30,430 unique triplets.
- **Milestone Latency Horizons**: 2026 to 2042 across 37 diseases.
- **Automated Testing Suite**: **26 of 26 tests passed (100%)** in 5.87 seconds.

---

## 7. Tough Viva Questions & Defensible Answers

### Q1: "Your symptom classifier gets 100% accuracy. Is that not severe overfitting or data leakage?"
**Defensible Answer**:
> *"No, it is not data leakage, but it must be interpreted with strict scientific caution. In the raw dataset, 93.87% of rows were duplicates. If evaluated on the raw data, test samples would have identical duplicates in the train set—which WOULD be data leakage. We eliminated that by collapsing the data to 304 unique clinical presentations and using a stratified split with zero pattern intersection. The 100% accuracy occurs because canonical synthetic disease vectors in this benchmark are linearly separable. We explicitly state in our report that this 100% score applies only to this synthetic benchmark and does NOT establish diagnostic accuracy in real clinical practice, where symptoms are probabilistic, noisy, and comorbid."*

### Q2: "Why did your complex Autoregressive Ridge model perform so much worse than simple Naive Persistence in epidemiological forecasting?"
**Defensible Answer**:
> *"Because real-world global epidemiology is heavily non-stationary. Global measles incidence dropped precipitously from the 1980s through the 2000s due to the WHO Expanded Programme on Immunization, but then saw sharp resurgences after 2016 due to vaccine hesitancy and conflict disruptions. Autoregressive regression models fit on the long-term downward trend overshot actual resurgence spikes, resulting in a MAPE of 240%. Naive persistence ($\hat{y}_t = y_{t-1}$) adapts instantly to the most recent level, achieving an MAE of 53,024 and MAPE of 91.95%. Preserving this negative finding demonstrates scientific integrity: we do not conceal poor results to make complex ML look artificially superior."*

### Q3: "In your candidate rankings, AIDS ranked Medroxyprogesterone Acetate at #1. Is Medroxyprogesterone a cure for AIDS?"
**Defensible Answer**:
> *"Absolutely not, and that is one of our most important findings. Medroxyprogesterone Acetate is a synthetic progestin used in contraception and endometriosis. It achieves a high Composite Priority Score ($CPS=0.7689$) because: (1) Open Targets lists an association between AIDS and the Progesterone Receptor (PGR) due to literature co-occurrence in hormonal contraceptive HIV-risk studies; (2) Medroxyprogesterone has nanomolar binding potency ($p\text{ChEMBL}=9.82$) to PGR; (3) it has perfect Lipinski drug-likeness; and (4) it is an FDA-approved Phase 4 drug. Because all four components score high, the heuristic formula ranks it #1. This case serves as a vital negative control proving that $CPS$ is an in-silico screening heuristic to prioritize hypotheses for laboratory assays, NOT clinical proof of antiviral efficacy."*

### Q4: "What happened to the year '2114' headline from the initial project proposal?"
**Defensible Answer**:
> *"In the initial project concept, the year '2114' was introduced as an illustrative, symbolic headline representing the long-term century-scale quest for cures. In Review-2 and Review-3, we conducted a rigorous audit of this claim. To train a supervised machine learning regression model that predicts an exact future approval year like 2114, you would have to fabricate future training labels that do not exist in historical data. Furthermore, biopharmaceutical pipeline predictability is empirically bounded to 10–15 year horizons. We formally segregated the year 2114 as an inspirational educational headline and replaced it with a scientifically grounded transition latency model based on published DiMasi benchmarks, projecting realistic decision windows between 2026 and 2042."*

### Q5: "Your master timeline has 555,323 rows. Did you find 555,000 clinical trials?"
**Defensible Answer**:
> *"No. The 555,323 rows are an artifact of an $18.25\times$ Cartesian join expansion. In the Drugs@FDA database, a single drug ingredient (e.g., Metformin) has multiple application numbers, dosage strengths (500mg, 850mg, 1000mg), dosage forms (tablet, extended-release), and packaging configurations. When cross-referenced with multiple recorded indication terms in ChEMBL, each formulation line duplicates across each indication. Our audit proved that these 555k rows represent exactly 30,430 unique disease-molecule pairs. Treating each row as an independent clinical trial would be completely scientifically flawed."*

### Q6: "Why is the FDA `first_approval_year` not necessarily the approval year for the queried disease?"
**Defensible Answer**:
> *"In regulatory pharmacology, the FDA approves an active chemical ingredient for an initial commercial indication (e.g., Aspirin for pain/fever in 1899). Decades later, clinicians may explore Aspirin for cardiovascular prophylaxis or colorectal cancer prevention. The Drugs@FDA database records the active ingredient's earliest marketing authorization. If you associate Aspirin with colorectal cancer in the timeline, the 1899 approval year represents the molecule's market entry, NOT the year it was approved for oncology. Our timeline audit explicitly flags this distinction."*

### Q7: "What is the difference between biological association, computational prediction, and clinical causation in TheraLens?"
**Defensible Answer**:
> *"TheraLens maintains strict conceptual separation between these three tiers:
> 1. **Association**: Open Targets computes statistical co-occurrence and genomic linkage between a gene and a disease. An association does not prove the gene is a druggable therapeutic target.
> 2. **Prediction**: Our models compute mathematical predictions (e.g., classification probabilities, $CPS$ scores, and time-ordered persistence forecasts). These predictions quantify pattern similarity in historical training sets.
> 3. **Causation**: True therapeutic efficacy requires randomized controlled double-blind clinical trials demonstrating that modulating the target alters disease progression in humans without unacceptable toxicity. Computational models prioritize hypotheses; they never establish clinical causation."*

### Q8: "When running disease prediction for Malaria, why does the model return a confidence score of 0.2794 (27.9%) rather than 99%?"
**Defensible Answer**:
> *"The score 0.2794 is an uncalibrated softmax decision margin across 41 competing disease classes. Presenting symptoms like chills, vomiting, high fever, sweating, and headache are non-specific constitutional symptoms shared across dozens of tropical febrile illnesses (such as Dengue, Typhoid, and Hepatitis). The multinomial logistic regression model correctly identifies Malaria as the highest-ranked diagnosis (rank #1), but distributes probability mass across other plausible differential diagnoses. We explicitly treat this score as an uncalibrated ranking heuristic rather than a true clinical posterior probability, deliberately avoiding artificial overconfidence."*

### Q9: "How did you perform entity resolution between Drugs@FDA and ChEMBL, and what are the exact matching metrics?"
**Defensible Answer**:
> *"We audited all 3,142 unique FDA active pharmaceutical ingredients extracted from 12,243 approved product records. Through a 5-tier entity resolution pipeline (exact preferred name, combination splitting, synonym lookup, partial combination, and salt-stripping), we successfully matched 3,059 ingredients to ChEMBL parent molecules, achieving an empirical match rate of 97.36% (3,059 / 3,142). The remaining 83 unmatched entities (2.64%) consist of complex biologics, unmapped radiopharmaceuticals, and botanical extracts. Every match decision is documented with its match method in `fda_chembl_match_audit.csv`."*

---

## 8. Step-by-Step Viva Demonstration Script

When asked to demonstrate the working software during the viva, execute this 3-step sequence:

### Step 1: Execute the 26-Test Automated Suite (10 seconds)
```powershell
pytest -v
```
- **What to say**: *"First, we demonstrate verified software quality. Our test suite runs 26 automated unit and integration tests across 5 test modules in under 5 seconds, validating schema integrity, zero pattern leakage, molecular scoring boundaries, edge-case handling, and negative control verification with a 100% pass rate."*

### Step 2: Run Real-Time Symptom Inference (5 seconds)
```powershell
python src/inference/predict_disease.py --symptoms "chills,vomiting,high_fever,sweating,headache" --top_k 3
```
- **What to say**: *"Next, we demonstrate our real-time clinical inference interface. Given a presenting symptom string, the system normalizes the keywords, aligns them against our 132-feature manifold, and predicts Malaria as the top differential diagnosis. Notice that the top score is 0.2794; our interface explicitly flags this as an uncalibrated decision margin across 41 competing classes rather than overconfident clinical certainty, accompanied by an automated medical disclaimer."*

### Step 3: Run the Master Pipeline Orchestrator (37 seconds)
```powershell
python src/run_theralens.py
```
- **What to say**: *"Finally, we execute the master end-to-end orchestrator. In 37 seconds, it sequentially retrains the symptom classifier, prioritizes 25,650 candidates across 37 diseases using our multi-criteria CPS score, runs epidemiological surveillance over 5,426 WHO records, audits 555k master timeline rows, projects biopharmaceutical latency horizons, and calculates permutation feature importance, outputting all 20 production artifacts with full traceability and zero fatal errors."*
