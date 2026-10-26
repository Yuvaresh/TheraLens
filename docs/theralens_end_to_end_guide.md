# TheraLens: End-to-End Emerging Disease Research Workflow Guide

## Executive Summary

**TheraLens** is an open-world translational biomedical intelligence platform designed to address the central research challenge:
> **Given the clinical symptoms of a newly emerging disease, which existing reference diseases have similar symptom profiles, what therapeutic evidence (drug targets, small molecules, clinical phases) is associated with those diseases, and what can historical biopharmaceutical development evidence reasonably suggest about the possible future availability of a treatment?**

This guide documents the architecture, mathematical formulations, evidence linkages, scenario models, execution instructions, and scientific limitations of the unified end-to-end workflow.

---

## 1. System Architecture

The TheraLens research workflow bridges clinical observations, computational genomics, medicinal chemistry, and regulatory history into a unified pipeline:

```
[ Emerging Disease Symptom Input ]
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Symptom Normalization & Open-World Similarity Analysis   │
│    - Alias mapping, duplicate filtering, case normalization │
│    - Prevalence-aware filtering (excluding 0% noise)       │
│    - Hallmark symptom detection (>= 50% presentation rate)  │
│    - Jaccard Similarity: J(A,B) = |A ∩ B| / |A ∪ B|         │
│    - Frequency-Weighted Jaccard J_weighted(A,B) & Cosine    │
│    - Transparent Breakdown: Shared, Query-Only, Ref-Only    │
│    (src/models/symptom_similarity.py)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │ Top Reference Disease Matches
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Cross-Disease Therapeutic Evidence Linkage & Auditing    │
│    - Cluster-wide multi-disease evidence evaluation         │
│    - Indication record verification (chembl_drug_indications)│
│    - 4-Tier Candidate Taxonomy:                             │
│        * Tier 1: Indication-Specific Approved Drug          │
│        * Tier 2: Indication-Specific Investigational        │
│        * Tier 3: Indirect Mechanistic Target Screen Hit     │
│        * Tier 4: Disqualified Toxicant / Surrogate (Lindane)│
│    - Composite Priority Score (CPS) Multi-Attribute Ranking │
│    - Multi-factor sorting: Tier 1 > Tier 2 > Tier 3 > Tier 4│
│    (src/models/therapeutic_evidence_retriever.py)           │
└──────────────────────────────┬──────────────────────────────┘
                               │ Viable Leads & Phase Benchmarks
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Prospective Milestone Forecasting (Scenario Simulation)  │
│    - DiMasi / Hay standard biopharmaceutical benchmarks     │
│    - Indication maturity governs development pathway:       │
│        * Tier 1: Repurposing Pathway (8.5 yrs baseline)     │
│        * Tier 2: Investigational Advancement (10.6-5.3 yrs) │
│        * Tier 3: De Novo Preclinical Discovery (15.1 yrs)   │
│        * Tier 4: Disqualified from clinical development    │
│    - Historical transition benchmarks (DiMasi / Wong et al.)│
│    - Explicit Transferability Assessment (No pseudo-probs)  │
│    - Multi-Scenario Technological Progress (0%, 15%, 30%)   │
│    - Illustrative Sensitivity Spread (+/- 30-35% variance)  │
│    (src/analysis/emerging_disease_forecaster.py)            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Unified Explainable Dossier Generation                   │
│    - Sections A through G (Structured Terminal Output)      │
│    - Cross-disease candidate comparison synthesis table     │
│    - Structured JSON Audit Record (outputs/forecasts/*.json)│
│    - Comprehensive Markdown Dossier (outputs/forecasts/*.md)│
│    - Mandatory Scientific Disclaimer                        │
│    (src/inference/forecast_emerging_disease.py)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulations & Methodologies

### 2.1 Symptom Similarity Analysis & Prevalence Modeling
Unlike closed-world classifiers that force arbitrary inputs into an argmax choice over 41 predetermined diseases, TheraLens uses explicit open-world distance metrics:

- **Jaccard Similarity Coefficient**:
  $$\mathcal{J}(Q, R) = \frac{|Q \cap R|}{|Q \cup R|}$$
  where $Q$ is the set of recognized symptoms in the query profile, and $R$ is the consensus symptom set for reference disease $R$ (filtered to exclude zero-prevalence features).

- **Frequency-Weighted Jaccard Similarity**:
  $$\mathcal{J}_{\text{weighted}}(Q, R) = \frac{\sum_{s \in Q \cap R} f_R(s)}{|Q| + \sum_{s \in R \setminus Q} f_R(s)}$$
  where $f_R(s) \in (0, 1]$ is the empirical presentation frequency of symptom $s$ across observed cases of disease $R$.

- **Hallmark Symptom Identification**:
  $$\text{Hallmarks}(R) = \{ s \in R \mid f_R(s) \ge 0.50 \}$$

- **Cosine Similarity**:
  $$\cos(\mathbf{q}, \mathbf{r}) = \frac{\mathbf{q} \cdot \mathbf{r}}{\|\mathbf{q}\|_2 \|\mathbf{r}\|_2}$$
  where $\mathbf{r}$ represents the empirical symptom frequency vector across historical presentations of disease $R$.

### 2.2 Four-Tier Candidate Evidence Taxonomy
To resolve the critical scientific flaw where high computational scores or molecule-level approvals in unrelated diseases falsely qualified non-therapeutic compounds as treatments (e.g., Lindane in malaria), TheraLens enforces a strict evidence hierarchy:

1. **Tier 1: Indication-Specific Approved Drug**
   - Direct documented regulatory approvals for the reference disease indication (`chembl_drug_indications`, `max_phase_for_ind >= 4.0`).
   - Eligible for the **Label Expansion Repurposing Pathway** ($8.5\text{ years}$ baseline: Phase 2 $\to$ Phase 3 $\to$ FDA Review).
2. **Tier 2: Indication-Specific Investigational Candidate**
   - Active clinical trials documented specifically for the reference disease (`1.0 <= max_phase_for_ind < 4.0`).
   - Advances from its current verified clinical phase.
3. **Tier 3: Indirect Mechanistic Target Association**
   - In-vitro affinity against a biological target associated with the reference disease, but lacking disease-specific clinical trials or approvals.
   - **Crucial Rule**: Global approvals or Phase 2/3 testing in unrelated human indications do **NOT** confer indication maturity. Indirect target hits cannot bypass preclinical discovery or Phase 1 safety trials, requiring the full **De Novo Preclinical Discovery Pipeline** ($15.1\text{ years}$ baseline).
4. **Tier 4: Disqualified Non-Therapeutic Surrogate / Toxicant**
   - Molecules identified as pesticides, toxicants, or assay screening artifacts (e.g., Lindane, Rotenone, Paraquat).
   - Completely disqualified from clinical repurposing and excluded from prospective forecasting.

### 2.3 Dual-Milestone Phase Accounting (Addressing Phase 1 Accounting Paradox)
Static biomedical repositories (e.g. ChEMBL) document that clinical trials reached a certain maximum phase (e.g. `max_phase_for_ind = 1.0` for Ritonavir in malaria). However, static registry tables do **not** verify whether trials are actively ongoing, completed with proven efficacy, or discontinued. 

Furthermore, counting Phase 1 as both completed and remaining creates an accounting inconsistency. To address this rigorously, TheraLens implements **Dual-Milestone Horizon Modeling**:

1. **Horizon A: Direct Phase Advancement (Optimistic Continuation / Phase Completed)**:
   - Assumes historical Phase $k$ clinical safety and tolerability hurdles were successfully cleared and accepted by regulatory bodies.
   - Candidate initiates directly into subsequent development ($k+1$).
   - *Phase 1 Candidate*: Remaining milestones are `["phase_2", "phase_3", "fda_review"]` ($8.5\text{ years}$ baseline; benchmark pipeline success: $18.5\%$; estimated year: $2034\text{--}2035$).
   - *Phase 2 Candidate*: Remaining milestones are `["phase_3", "fda_review"]` ($5.3\text{ years}$ baseline; benchmark pipeline success: $52.8\%$).

2. **Horizon B: Clinical Re-Validation / In-Progress (Conservative / Phase Incomplete or Target Cohort Re-Run)**:
   - Conservatively accounts for Phase $k$ trial completion or preliminary safety/pharmacokinetic re-validation in the emerging disease patient population before Phase $k+1$ initiation.
   - *Phase 1 Candidate*: Remaining milestones are `["phase_1", "phase_2", "phase_3", "fda_review"]` ($10.6\text{ years}$ baseline; benchmark pipeline success: $11.0\%$; estimated year: $2037$).
   - *Phase 2 Candidate*: Remaining milestones are `["phase_2", "phase_3", "fda_review"]` ($8.5\text{ years}$ baseline).

TheraLens presents both horizons transparently, acknowledging that a single deterministic date cannot be assumed from static registry records.

### 2.4 Biological Role & Target Relevance (Host Targets vs Pathogen Targets)
TheraLens explicitly evaluates candidate suitability beyond numerical affinity scores:
- **Host Targets vs Pathogen Targets**:
  Open Targets associations identify human host susceptibility genes, drug-metabolizing enzymes (e.g., `CYP3A5`, `SLCO1B1`), and physiological receptors (e.g., `ADRA1D`, `CACNA1S`). In infectious syndromes, these represent host response mechanisms, not pathogen-encoded proteins.
- **Pharmacokinetic Enhancers / Adjuvants vs Direct Curative Agents**:
  Candidates such as **Ritonavir** (`CHEMBL163`) are potent CYP3A inhibitors historically studied in Phase 1 as pharmacokinetic adjuvants to elevate concentrations of partner antimalarials, **not as standalone curative antiparasitic agents**. TheraLens explicitly flags these pharmacological roles with biological suitability warnings to prevent misleading monotherapy assumptions.

### 2.5 Multi-Scenario Technological Progress Modeling
Development timelines are modeled from established biopharmaceutical industry benchmarks ([DiMasi et al., 2003, 2016](https://doi.org/10.1016/j.jhealeco.2016.01.012); [Hay et al., 2014](https://doi.org/10.1038/nbt.2786); [Wong et al., 2019](https://doi.org/10.1093/biostatistics/kxx069)):

| Clinical / Regulatory Phase | Median Duration (Years) | Historical Transition Success Rate |
| :--- | :---: | :---: |
| **Preclinical Lead Optimization** | 4.5 years | 40.0% |
| **Phase 1 First-in-Human Safety** | 2.1 years | 59.5% |
| **Phase 2 Proof-of-Concept Efficacy** | 3.2 years | 35.1% |
| **Phase 3 Pivotal Confirmatory Trials** | 3.8 years | 58.6% |
| **FDA NDA / BLA Dossier Review** | 1.5 years | 90.1% |

#### Therapeutic Transferability vs. False Probabilities
TheraLens explicitly **avoids** multiplying pipeline transition success by symptom similarity ($P \times \mathcal{J}$). Symptom overlap is a phenotypic clinical metric that does not measure biological target homology or therapeutic response. Instead, TheraLens reports:
1. **Historical Pipeline Success Rate**: The sequential transition rate benchmark derived from industry literature ($18.5\%$ for Horizon A, $11.0\%$ for Horizon B from Phase 1 to Approval; $4.4\%$ for de novo discovery).
2. **Qualitative Transferability Assessment**: Transparent explanation of biological and pharmacological caveats.

#### Technological Progress Scenarios (Sensitivity Simulation)
The application evaluates prospective development through three transparent exploratory scenarios under both milestone horizons:
1. **Historical Industry Baseline** ($1.00\times$ multiplier, $0\%$ speedup): Adheres strictly to historical DiMasi/Hay median phase durations.
2. **Moderate Technological Progress** ($0.85\times$ multiplier, $15\%$ speedup): Duration reductions in discovery and early clinical phases via automated screening and genomic biomarker stratification.
3. **Aggressive AI & Platform Acceleration** ($0.70\times$ multiplier, $30\%$ speedup): Breakthrough reductions via generative molecular design, in-silico ADMET toxicity filtering, and adaptive multi-arm platform trials.

#### Illustrative Duration Sensitivity Spread
Rather than claiming an uncalibrated "80% prediction interval", TheraLens reports an **Illustrative Sensitivity Spread** reflecting $\pm 30\text{–}35\%$ historical phase duration variance:
$$\text{Sensitivity Range} = \left[ Y_0 + D_{\text{adj}} \cdot 0.70, \; Y_0 + D_{\text{adj}} \cdot 1.35 \right]$$

---

## 3. How to Execute the Unified Workflow

### 3.1 Primary Entry Point: Unified Research Forecaster
Execute the standalone CLI:
```bash
python src/inference/forecast_emerging_disease.py \
    --symptoms "chills, high_fever, sweating, headache, vomiting" \
    --disease_name "Novel_Febrile_Syndrome_Z" \
    --top_k 3 \
    --min_similarity 0.10
```

### 3.2 Backward-Compatible Entry Point: Extended Disease Predictor
The existing inference interface supports seamless execution via the `--forecast` or `--full` flags:
```bash
python src/inference/predict_disease.py \
    --symptoms "chills, high_fever, sweating, headache, vomiting" \
    --forecast
```

### 3.3 Programmatic Python API
```python
from src.inference.forecast_emerging_disease import TheraLensUnifiedOrchestrator

orchestrator = TheraLensUnifiedOrchestrator()
report = orchestrator.analyze_emerging_disease(
    symptom_input="chills, vomiting, high_fever, sweating, headache",
    disease_name="Emerging_Tropical_Syndrome_X",
    top_k_diseases=3,
    min_similarity=0.10,
)

# Print formatted console summary
orchestrator.print_console_summary(report)

# Save structured JSON and Markdown reports
json_path, md_path = orchestrator.save_report(report)
print(f"Artifacts saved to {json_path} and {md_path}")
```

---

## 4. Required Output Sections (Sections A through G)

Every execution produces a structured report containing:
- **Section A: Query Summary**: Input symptoms, recognized 132-feature mappings, unrecognized terms, duplicate terms, and vocabulary limitations.
- **Section B: Symptom-Similarity Analysis**: Ranked known diseases, Jaccard similarity, Cosine similarity, shared symptoms, query-only symptoms, and reference-only symptoms.
- **Section C: Therapeutic Evidence**: Retrieved biological drug targets from Open Targets with association scores and clinical pipeline maturity.
- **Section D: Candidate Prioritization**: Ranked candidate compounds with ChEMBL IDs, target symbols, clinical phases, $p\text{ChEMBL}$ values, drug-likeness scores, Composite Priority Scores ($CPS$), and negative-control surrogate annotations.
- **Section E: Development-History Analysis**: Observational units, historical phase durations, and empirical phase-transition success rates.
- **Section F: Future-Year Forecast**: Forecast target, pathway type, remaining phases, baseline duration, multi-scenario horizons (Baseline, Moderate, Aggressive), 80% decision windows, and methodological limitations.
- **Section G: Research Conclusion & Mandatory Medical Disclaimer**: Synthesis of findings and academic disclaimer.

---

## 5. Automated Verification & Testing

The complete test suite contains **53 automated tests** covering unit, fixture, and integration behaviors with a 100% pass rate and zero warnings:
```bash
python -m pytest -v
```

Test breakdown:
- **`tests/test_emerging_disease_end_to_end.py` (27 tests)**:
  - Exact, partial, disjoint, empty, duplicate, alien, case-normalized, alias-mapped symptom similarity.
  - Frequency-weighted Jaccard and hallmark symptom identification.
  - Threshold filtering and descending ranking order.
  - Target and compound evidence retrieval with negative-control identification.
  - **Lindane-Malaria Disqualification**: Verification that Lindane is disqualified as a Tier 4 toxicant.
  - **Dual-Milestone Accounting**: Verification that Phase 1 candidates produce both Horizon A (Direct Phase Advancement: 8.5y) and Horizon B (Clinical Re-Validation: 10.6y).
  - **Pharmacological Role Annotation**: Verification that Ritonavir is annotated as a pharmacokinetic booster / host target interactor, with biological suitability warnings.
  - Repurposing vs preclinical development pathways.
  - Multi-scenario acceleration progression.
  - Insufficient similarity / no candidate / all-disqualified scientific restraint.
  - Cross-disease candidate comparison and end-to-end orchestration.
  - JSON schema validation and Markdown report persistence without circular references.
- **`tests/test_unit_and_pipeline_fixtures.py` (11 tests)**:
  - Vectorization, feature consistency, Lipinski boundary conditions, salt-stripping, epidemiological metrics.
- **`tests/test_symptom_classifier.py` (5 tests)**:
  - Deduplication, leakage-free split, reproducibility, and classification API.
- **`tests/test_therapeutic_prioritization.py` (4 tests)**:
  - Schema integrity, drug-likeness scoring, clinical bonus logic, ranking persistence.
- **`tests/test_timeline_validation.py` (3 tests)**:
  - Timeline schema, ChEMBL mapping rates, indication evidence classification.
- **`tests/test_epidemiological_analysis.py` (3 tests)**:
  - Trend analysis, missing value retention, and outbreak surge detection.

---

## 6. Scientific Limitations & Ethical Principles

1. **Exploratory Computational Hypothesis**: TheraLens is a translational research-support tool for academic exploration. It is **NOT** a diagnostic engine, a prescribing tool, or a substitute for medical advice.
2. **Phenotypic Similarity $\neq$ Etiological Identity**: Two diseases sharing symptoms (e.g. fever, headache, chills) may have completely different pathogens (viral vs bacterial vs protozoal). Symptom similarity must never be interpreted as biological proof that a treatment will transfer.
3. **Negative Controls & Off-Target Bindings**: As demonstrated by Medroxyprogesterone in AIDS (binding PGR), molecules can score highly on generic chemical tractability without being viable treatments. Laboratory target engagement assays and in-vivo infection models are mandatory.
4. **Epistemic Uncertainty in Long-Range Forecasts**: Precision dates decades in the future carry inherent scientific uncertainty. The application provides bounded multi-scenario ranges and explicitly withholds forecasts when evidence is insufficient, adhering to rigorous scientific integrity.
