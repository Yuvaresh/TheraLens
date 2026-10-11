# TheraLens: Future Therapeutic Development Forecasting & Limitations Report

**Project Title**: TheraLens: Machine Learning-Based Forecasting of Emerging Diseases and Future Therapeutic Development  
**Academic Document**: Scientific Methodology & Limitation Appraisal (Review-2)  
**Date**: October 2026  

---

## 1. Executive Summary & Objective

The TheraLens project investigates whether biomedical machine learning—combining clinical presentations, disease-target biological networks, in-vitro compound potency, and historical regulatory timelines—can model and forecast the trajectory of future therapeutic solutions.

In the initial project concept, **the year "2114"** was introduced as an illustrative headline highlighting the long-term quest for therapeutic interventions against challenging diseases. This report provides an academically rigorous, evidence-backed evaluation of what **can** and **cannot** be scientifically inferred from available biomedical datasets, distinguishes illustrative project concepts from empirical machine learning, and documents the experimental forecasting prototype implemented for Review-2.

---

## 2. Analysis of the Target Variable: Can an Exact Cure Year Be Learned?

### 2.1 The Nature of Available Historical Labels
In the acquired FDA Drugs@FDA dataset (`fda_drug_approval_timeline.csv`), the primary historical milestone column is:
$$\text{first\_approval\_year} \in [1939, 2026]$$
This represents the historical year an active pharmaceutical ingredient first received United States regulatory marketing authorization for an initial commercial product.

### 2.2 Critical Data Gaps for Supervised Target Prediction
To train a valid supervised machine learning regression model that predicts a disease's *future therapeutic solution year*, the dataset would need to contain:
1. **Historical Disease Discovery / Characterization Dates**: The baseline year from which R&D latency began.
2. **Preclinical & Investigational New Drug (IND) Filing Timestamps**: The chronological transition point from discovery to clinical testing.
3. **True Indication-Specific Label Approval Dates**: The exact year an FDA approval was granted specifically for that indicated pathology, rather than an unrelated prior indication.
4. **Unmet Disease Labels**: For diseases without any current cure or therapy, a future approval date is mathematically unobservable in historical data.

### 2.3 Scientific Rejection of Premature Machine Learning Extrapolations
As established in the TheraLens Review-2 guidelines:
- **No Invented Labels**: Training a regression algorithm (e.g., XGBoost, Random Forest Regressor, Linear Regression) by assigning arbitrary future target numbers (e.g., year 2114) would constitute fraudulent machine learning.
- **No Fixed Additions**: Simply taking $\text{Current Year} + k$ (e.g., $2026 + 88 = 2114$) is an unverified heuristic, not a learned data-driven model.
- **No Label Conflation**: An FDA drug's first approval year for condition $A$ cannot be treated as a confirmed historical solution year for indication $B$.

---

## 3. Dissecting the Illustrative "Year 2114" Headline

| Attribute | Project Proposal Headline ("2114") | Scientific Reality in Review-2 |
| :--- | :--- | :--- |
| **Status** | Illustrative / Metaphorical Horizon | Unobservable Speculative Extrapolation |
| **Data Basis** | Hypothetical conceptual target | No empirical training label exists |
| **Time Horizon** | 88 years into the future | Biopharmaceutical predictability is limited to 10–15 years |
| **Epistemic Uncertainty** | Infinite (biological paradigms change over century scales) | Bounded by clinical trial transition probabilities |
| **Recommendation** | Retained strictly as an educational / inspirational motivation | **Explicitly segregated from measured empirical results** |

---

## 4. Experimental Forecasting Prototype: Methodology & Assumptions

To address the project's objective scientifically, TheraLens implements an **empirical biopharmaceutical pipeline progression prototype** in `src/analysis/forecast_therapeutic_timeline.py`.

### 4.1 Benchmark Assumptions (DiMasi et al., Nature Reviews Drug Discovery)
Rather than ungrounded extrapolation, the prototype models clinical pipeline transition latencies based on multi-decade industry benchmarks across FDA and global clinical trial databases:

$$\text{Pipeline Duration} = \sum_{p \in \text{Remaining Phases}} \tau_p$$

| Clinical Phase | Median Phase Duration ($\tau_p$) | Historical Transition Probability ($P_p$) | Primary Attrition Mechanism |
| :--- | :---: | :---: | :--- |
| **Phase 1** | 2.1 years | 59.5% | Dose-limiting toxicities, pharmacokinetic failure |
| **Phase 2** | 3.2 years | 35.1% | Lack of clinical efficacy in target patient cohorts |
| **Phase 3** | 3.8 years | 58.6% | Failure in pivotal double-blind randomized trials |
| **FDA NDA Review** | 1.5 years | 90.1% | Regulatory rejection / manufacturing quality issues |
| **Cumulative (Phase 1 $\to$ Approval)** | **10.6 years** | **~11.3%** | **~89% overall candidate attrition rate** |

### 4.2 Mathematical Formulation of Projected Decision Window
For a disease with an advanced candidate currently at clinical phase $P_{\text{current}}$:
1. **Median Projected Year**:
   $$\hat{Y}_{\text{milestone}} = Y_{\text{base}} + \sum_{p = P_{\text{current}}}^{4.0} \tau_p$$
2. **Confidence Window ($80\%$ Uncertainty Interval)**:
   $$[Y_{\text{base}} + 0.65 \times \tau_{\text{cum}}, \; Y_{\text{base}} + 1.45 \times \tau_{\text{cum}}]$$
3. **Cumulative Probability of Success ($P_{\text{success}}$)**:
   $$P_{\text{success}} = \prod_{p = P_{\text{current}}}^{4.0} P_p$$

---

## 5. Empirical Results Generated by the Prototype

Evaluating the 37 disease classes present in the TheraLens candidate prioritization pipeline:
- **Approved Therapies Exist (Repurposing / Label Expansion)**: 33 diseases have compounds in Phase 4. For these, the therapeutic development milestone was already historically achieved; current research focuses on multi-target optimization and drug repurposing.
- **Active Clinical Trials (Phase 2 / Phase 3)**: Diseases with candidates in active clinical trials have empirical decision windows between **2028 and 2036** (median latency: 2–10 years).
- **Preclinical Discovery Stage**: Diseases with only in-vitro screened hits (Phase 0) have projected milestone windows between **2036 and 2045**, with an estimated cumulative realization probability of $\approx 4.5\%$ due to severe preclinical attrition.

The complete disease-by-disease projections are saved in:
`outputs/timeline_analysis/experimental_forecast_projections.csv`

---

## 6. What Can and Cannot Be Inferred: Master Summary

### What TheraLens CAN Infer:
1. Which biological targets exhibit the strongest evidence of genetic and pathway involvement for a disease (via Open Targets association scores).
2. Which chemical compounds exhibit confirmed sub-micromolar in-vitro binding affinity against those targets (via ChEMBL pChEMBL values).
3. Which compounds satisfy Lipinski and Veber drug-likeness rules for oral bioavailability and structural suitability.
4. The current clinical development phase and realistic decision window (with attrition probabilities) for active candidates.

### What TheraLens CANNOT Infer:
1. TheraLens **cannot** guarantee that a high-scoring compound is clinically safe, non-toxic, or effective in human clinical trials.
2. TheraLens **cannot** predict an exact calendar year (e.g., 2114) for the eradication or cure of any disease.
3. TheraLens **cannot** substitute for laboratory experimental pharmacology, randomized controlled trials, or regulatory medical evaluation.

---

## 7. Required Evidence for Future Multi-Decade Forecasting

To advance beyond pipeline latency modeling toward full predictive forecasting in subsequent project phases:
1. Ingestion of clinical trial registry datasets (ClinicalTrials.gov) with exact study start, completion, and termination dates.
2. Time-stamped patent application and expiration filings to track commercial IP investment timelines.
3. Natural history progression rates of emerging infectious variants to model evolving pathogen resistance against existing drug classes.
