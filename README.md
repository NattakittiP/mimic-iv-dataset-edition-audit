# Dataset Edition as a Source of Evaluation Divergence  
### A Controlled Audit of MIMIC-IV Demo vs Full Releases

---

## Overview

This repository accompanies the study:

**“Dataset Edition as a Source of Evaluation Divergence in Clinical Machine Learning:  
A Controlled Study on MIMIC-IV Demo and Full Releases.”**

The central question addressed in this work is:

> **Can the MIMIC-IV demo dataset be used as a faithful proxy for the full MIMIC-IV dataset under controlled and identical evaluation pipelines?**

Although the MIMIC-IV demo release is officially described as a subset intended for feasibility checks and educational use, it is increasingly employed in full experimental pipelines and published clinical ML studies. This work systematically evaluates whether such usage is methodologically justified.

---
## TL;DR
- Identical ML pipelines evaluated on MIMIC-IV Demo vs Full
- Only dataset edition differs
- Evaluation metrics, calibration, and model rankings diverge systematically
- Demo results do not reliably transfer to Full
- Dataset edition must be reported and treated as experimental factor

---

## Key Claim (Precisely Scoped)

This study **does not argue that demo-based studies are invalid**.

Instead, it demonstrates that:

> **Model evaluation outcomes obtained from the MIMIC-IV demo dataset are dataset-conditional and, under controlled conditions, do not reliably transfer to the full MIMIC-IV cohort.**

Therefore, **dataset edition itself must be treated as a first-order experimental factor**, not a benign implementation detail.

---

## Experimental Design Principles

To isolate the effect of *dataset edition alone*, we enforce **strict pipeline control**:

| Controlled Factor | Status |
|------------------|--------|
| Cohort construction | Identical |
| Data cleaning | Identical code |
| Feature definitions | Identical |
| Preprocessing | Identical |
| Models & hyperparameters | Identical |
| Random seeds | Identical |
| Cross-validation splits | Identical logic |
| Evaluation metrics | Identical |

**The only difference between runs is the dataset edition**:
- MIMIC-IV Demo (v2.2): 252 admissions (`Clear dataset/demo_analytic_dataset_mortality_all_admissions.csv`)
- MIMIC-IV Full (v3.1): 14,081 admissions (`Clear dataset/full_analytic_dataset_mortality_all_admissions.csv`)

Any observed divergence therefore reflects **dataset-induced effects**, not implementation artifacts.

---

## Repository Structure

---

## Core Pipeline

### `Code.py`
**Main experimental pipeline**

- Executes the **full clinical ML workflow** end-to-end
- Runs **demo and full analytic datasets independently**
- Ensures **identical model settings, folds, and evaluation protocols**
- Produces standardized outputs for downstream comparison

This file serves as the **single source of truth** for experimental execution.

---

## Statistical Comparison Engine

### `compare_demo_full_pro.py`
**Primary demo vs full comparison module**

Implements the core statistical analyses, including:

- **RSCE (Relative Shift in Calibration Error)** comparison
- **Paired metric deltas** across folds
- **Model ranking agreement and instability analysis**
- **Sign tests** and **effect size estimation**
- **Regression-based attribution** of divergence (demo vs full effects)

This module is the analytical backbone of the framework.

---

### `compare_demo_full_pro_addons.py`
**Extended and diagnostic analyses**

Adds deeper inspection tools on top of the core engine:

- Ablation-style summary statistics
- Per-fold paired hypothesis tests
- **Reliability curve distance analysis**
- Fold-wise stability diagnostics

Useful for **debugging divergence sources** and secondary validation.

---

## Visualization

### `compare_demo_full_heatmap_plot.py`
**High-level divergence visualization**

Generates publication-ready plots illustrating:

- Metric shift heatmaps
- Model ranking instability patterns
- Cross-metric divergence structure

Designed to provide **rapid global insight** into demo vs full discrepancies.

---

## Robustness & Stress Testing

### `Compare_PlusPlus.py`
**Robustness confirmation experiments**

Tests whether observed divergence persists under:

- Controlled subsampling regimes
- Alternative perturbation strategies
- Feature importance stability checks

This module helps distinguish **structural divergence** from
sampling or noise-driven artifacts.

---

## Prevalence-Standardized Evaluation

### `Prevalence_standarized.py`
**Prevalence-controlled metric computation**

- Computes decision metrics (e.g., PPV) at **fixed sensitivity**
- Applies a **common reference prevalence**
- Eliminates base-rate–induced distortions in demo vs full comparisons

Essential for fair evaluation in imbalanced clinical settings.

---

### `Compare_PPV.py`
**Paired PPV comparison**

- Performs **fold-level paired analysis** of prevalence-standardized PPV
- Compares demo and full datasets under matched operating points
- Reports statistical significance and effect sizes

---

## Intended Use Cases

- Validating whether **demo datasets are faithful surrogates**
- Detecting **hidden generalization failures**
- Studying **metric instability and calibration drift**
- Supporting **clinical ML reproducibility and auditability**
---

## Evaluation Dimensions

The comparison goes beyond single-metric performance and evaluates:

1. **Discrimination**
   - AUROC, AUPRC

2. **Calibration**
   - Log loss, Brier score
   - ECE / adaptive ECE
   - Reliability curves

3. **Decision-Level Metrics**
   - Prevalence-standardized PPV at fixed sensitivity

4. **Robustness**
   - Controlled perturbation worlds
   - Distribution shift, missingness, noise, label corruption

5. **Comparative Stability**
   - Model ranking agreement
   - Spearman / Kendall rank correlations

6. **Explainability Stability**
   - Feature importance consistency
   - SHAP-based stability (where applicable)

7. **Composite Evaluation**
   - RSCE (Reliability–Stability of Clinical Evaluation)

---

## Why Sample Size Alone Does Not Explain the Results

To rule out the trivial explanation that
“the demo dataset is just smaller,” we conduct:

- Size-matched subsampling of the full dataset
- Repeated subsample experiments
- Paired perturbation analysis with deterministic seeding

**Result:**  
The demo analytic cohort behaves as a **distinct statistical regime**, not as a typical small-sample realization of the full cohort.

---

## Interpretation Guidelines

This work should be interpreted as follows:

- ❌ **Not**: “Demo-based studies are wrong”
- ❌ **Not**: “Previous results should be discarded”
- ✅ **Yes**: Demo-based conclusions are **dataset-conditional**
- ✅ **Yes**: Dataset edition must be explicitly reported
- ✅ **Yes**: Cross-dataset comparability should not be assumed

---

## Reproducibility

- All experiments are deterministic given fixed seeds
- All perturbation worlds are paired across dataset editions
- No dataset-specific tuning or manual intervention is used
- The same codebase is executed independently on both datasets

---

## Ethical and Governance Notes

- All analyses use de-identified data released under
  MIMIC-IV data use agreements
- No patient-level re-identification is attempted
- The study complies fully with PhysioNet governance policies

---

## Final Note

This repository is intended as a **methodological audit**, not a benchmark leaderboard.

Its goal is to improve rigor, transparency, and interpretability in clinical
machine learning research by highlighting **dataset edition** as an
underappreciated source of evaluation variability.
