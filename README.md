# Can the MIMIC-IV Demo Support Model Selection? An Audit Against Full Data

This repository contains the complete, runnable pipeline and the aggregate results of an audit
that asks one question: **if a clinical machine-learning model is chosen on the open-access
MIMIC-IV Demo (100 patients), is it the same model that would be chosen on the full,
credentialed MIMIC-IV release?**

The same code, with the same settings, is run on the Demo and on the Full release in two
structurally different prediction tasks:

| Track | Release pair | Unit | Outcome | Prediction time (landmark) |
|---|---|---|---|---|
| **hosp** | MIMIC-IV v2.2 Demo / Full | adult hospital admission | in-hospital death | admission + 24 h |
| **ED** | MIMIC-IV-ED v2.2 Demo / Full | emergency-department stay | hospital encounter (admitted or observation) | ED arrival + 1 h |

Seven fixed classifiers are scored with a multi-world robustness benchmark (RSCE), with a
prevalence-standardized positive predictive value (PPV), and against null distributions built
from 1,000 Demo-sized samples of the Full release. Every number quoted below is read from a
file in [`Results/`](Results/), and the file is named next to the number.

---

## Contents

1. [Main findings](#1-main-findings)
2. [What these results do not establish](#2-what-these-results-do-not-establish)
3. [Study design](#3-study-design)
4. [Repository layout](#4-repository-layout)
5. [Data access](#5-data-access)
6. [Installation](#6-installation)
7. [How to reproduce](#7-how-to-reproduce)
8. [Result files reference](#8-result-files-reference)
9. [Reproducibility checks performed on this repository](#9-reproducibility-checks-performed-on-this-repository)
10. [Data citations](#10-data-citations)

---

## 1. Main findings

Values are shown to 4 decimal places; the files hold full precision.

### 1.1 Cohorts

Source: [`Results/table1/table1_combined.csv`](Results/table1/table1_combined.csv),
`Results/*/*.cohort_flow.json`.

| Dataset | Rows (units) | Patients | Positives | Negatives | Prevalence |
|---|---:|---:|---:|---:|---:|
| hosp Demo | 245 admissions | 100 | 14 | 231 | 0.0571 |
| hosp Full | 331,204 admissions | 148,256 | 7,302 | 323,902 | 0.0220 |
| ED Demo | 219 stays | 64 | 170 | 49 | 0.7763 |
| ED Full | 413,860 stays | 200,794 | 197,988 | 215,872 | 0.4784 |

* Every Demo patient is also a Full patient: 100 of 100 (hosp) and 64 of 64 (ED)
  ([`Results/checks/demo_subset_check_hosp.json`](Results/checks/demo_subset_check_hosp.json),
  [`..._ed.json`](Results/checks/demo_subset_check_ed.json)). The Demo is a slice of the Full
  population, not an independent cohort.
* Every Demo patient has an ICU stay; 50,920 Full patients have at least one ICU stay
  ([`Results/checks/full_icu_subject_ids.json`](Results/checks/full_icu_subject_ids.json)), out of
  299,712 patients in the Full `hosp/patients` table (about 17%). The Demo is therefore not a
  simple random sample of Full patients, which is why the null analysis (Section 1.5) is repeated
  with an ICU-patient pool.
* The outcome prevalence differs between releases in both tracks (hosp 0.0571 vs 0.0220; ED 0.7763
  vs 0.4784).

### 1.2 RSCE scores: every model scores higher on Full

Source: [`Results/compare/<track>/rsce_comparison_demo_vs_full.csv`](Results/compare/hosp/rsce_comparison_demo_vs_full.csv).
ΔRSCE = Full minus Demo.

**hosp**

| Model | RSCE Full | RSCE Demo | ΔRSCE | Rank Full | Rank Demo |
|---|---:|---:|---:|---:|---:|
| MLP | 0.9314 | 0.7575 | 0.1739 | 1 | 7 |
| ExtraTrees | 0.9312 | 0.8336 | 0.0976 | 2 | 2 |
| GradientBoosting | 0.9305 | 0.8133 | 0.1172 | 3 | 4 |
| RandomForest | 0.9277 | 0.8464 | 0.0813 | 4 | 1 |
| Logistic_L2 | 0.9237 | 0.7967 | 0.1269 | 5 | 6 |
| SVC_RBF | 0.8913 | 0.8296 | 0.0617 | 6 | 3 |
| GaussianNB | 0.8860 | 0.7995 | 0.0866 | 7 | 5 |

**ED**

| Model | RSCE Full | RSCE Demo | ΔRSCE | Rank Full | Rank Demo |
|---|---:|---:|---:|---:|---:|
| GradientBoosting | 0.8938 | 0.8017 | 0.0921 | 1 | 5 |
| Logistic_L2 | 0.8928 | 0.8444 | 0.0484 | 2 | 1 |
| RandomForest | 0.8927 | 0.8330 | 0.0596 | 3 | 3 |
| ExtraTrees | 0.8918 | 0.8331 | 0.0588 | 4 | 2 |
| MLP | 0.8784 | 0.7939 | 0.0845 | 5 | 6 |
| SVC_RBF | 0.8722 | 0.7929 | 0.0793 | 6 | 7 |
| GaussianNB | 0.8391 | 0.8144 | 0.0247 | 7 | 4 |

* ΔRSCE is positive for 7 of 7 models in both tracks; two-sided exact sign test p = 0.0156 in each
  ([`sign_test_summary.csv`](Results/compare/hosp/sign_test_summary.csv)). Mean ΔRSCE is 0.1064
  (hosp) and 0.0639 (ED) ([`Results/compare/cross_domain/cross_domain_summary.csv`](Results/compare/cross_domain/cross_domain_summary.csv)).
* Fold-level unpaired tests (15 Full folds vs 15 Demo folds; Welch, Holm-adjusted) reject equality
  for 7 of 7 hosp models and 6 of 7 ED models; the exception is ED GaussianNB (Δ 0.0247, 95% CI
  0.0026 to 0.0502, Welch Holm p = 0.0669)
  ([`rsce_full_vs_demo_unpaired_tests.csv`](Results/compare/ed/rsce_full_vs_demo_unpaired_tests.csv)).
* **The gap is mostly clean discrimination (R).** ΔRSCE decomposes exactly into weighted component
  changes (maximum reconstruction error 2.1e-16). The R term carries 95.3% of the mean ΔRSCE in hosp
  and 88.3% in ED ([`delta_rsce_decomposition.csv`](Results/compare/hosp/delta_rsce_decomposition.csv),
  `cross_domain_summary.csv`, `mean_contrib_R / mean_delta_rsce_full_minus_demo`).

### 1.3 Model ranking does not carry over from Demo to Full

Source: [`Results/compare/<track>/rank_agreement.csv`](Results/compare/hosp/rank_agreement.csv),
[`Results/compare/rsce_rsc_rank_agreement.csv`](Results/compare/rsce_rsc_rank_agreement.csv).

| Track | Score | Full leader | Demo leader | Full leader's rank on Demo | Spearman ρ | Kendall τ |
|---|---|---|---|---:|---:|---:|
| hosp | RSCE | MLP | RandomForest | 7 of 7 | -0.0714 | -0.0476 |
| hosp | RSCE_RSC | MLP | RandomForest | 7 of 7 | -0.3571 | -0.2381 |
| ED | RSCE | GradientBoosting | Logistic_L2 | 5 of 7 | 0.4286 | 0.3333 |
| ED | RSCE_RSC | ExtraTrees | Logistic_L2 | 2 of 7 | 0.2143 | 0.1429 |

RSCE_RSC drops the explanation term E for every model, so it compares all seven models on exactly
the same three components (Section 3.6). The disagreement is present under both scores.

Two qualifications that belong next to these numbers:

* **On Full, the top of the ranking is a statistical tie.** In hosp the Full leader (MLP) differs
  from ExtraTrees by 0.0002, GradientBoosting by 0.0009, RandomForest by 0.0037 and Logistic_L2 by
  0.0077 in mean fold RSCE, and none of these four differences is significant after Nadeau-Bengio
  correction and Holm adjustment (adjusted p ≥ 0.5614). In ED, GradientBoosting is not separated
  from ExtraTrees, Logistic_L2 or RandomForest (adjusted p = 1.0000)
  ([`Results/hosp_full/rsce/paired_tests.csv`](Results/hosp_full/rsce/paired_tests.csv),
  [`Results/ed_full/rsce/paired_tests.csv`](Results/ed_full/rsce/paired_tests.csv)). Full separates
  11 of 21 hosp model pairs and 15 of 21 ED model pairs at adjusted p < 0.05.
* **On Demo, no model pair is separated.** 0 of 21 pairs in either track reach adjusted p < 0.05
  ([`Results/hosp_demo/rsce/paired_tests.csv`](Results/hosp_demo/rsce/paired_tests.csv),
  [`Results/ed_demo/rsce/paired_tests.csv`](Results/ed_demo/rsce/paired_tests.csv)). The Demo
  ranking is a ranking of statistically indistinguishable models.

### 1.4 Prevalence-standardized PPV is lower on Demo for every model

Source: [`Results/compare/<track>/ppv/per_model_full_vs_demo_unpaired.csv`](Results/compare/hosp/ppv/per_model_full_vs_demo_unpaired.csv).
Operating point: the threshold reaching sensitivity 0.80 on inner out-of-fold training predictions;
PPV standardized to the Full prevalence π_ref (hosp 0.0220, ED 0.4784). Difference = Full minus Demo,
with a 95% bootstrap CI.

| Track | Model | PPV_std Full | PPV_std Demo | Difference | 95% CI |
|---|---|---:|---:|---:|---|
| hosp | GradientBoosting | 0.0946 | 0.0228 | 0.0718 | 0.0708 to 0.0727 |
| hosp | RandomForest | 0.0922 | 0.0284 | 0.0638 | 0.0599 to 0.0683 |
| hosp | MLP | 0.0855 | 0.0255 | 0.0600 | 0.0556 to 0.0637 |
| hosp | ExtraTrees | 0.0792 | 0.0301 | 0.0490 | 0.0449 to 0.0532 |
| hosp | GaussianNB | 0.0692 | 0.0220 | 0.0472 | 0.0447 to 0.0503 |
| hosp | Logistic_L2 | 0.0644 | 0.0255 | 0.0389 | 0.0346 to 0.0437 |
| hosp | SVC_RBF | 0.0455 | 0.0330 | 0.0125 | 0.0068 to 0.0187 |
| ED | GradientBoosting | 0.6694 | 0.5074 | 0.1621 | 0.1420 to 0.1852 |
| ED | MLP | 0.6610 | 0.5248 | 0.1362 | 0.0876 to 0.1836 |
| ED | RandomForest | 0.6633 | 0.5373 | 0.1259 | 0.0962 to 0.1598 |
| ED | SVC_RBF | 0.6556 | 0.5412 | 0.1144 | 0.0857 to 0.1437 |
| ED | ExtraTrees | 0.6599 | 0.5467 | 0.1132 | 0.0833 to 0.1459 |
| ED | Logistic_L2 | 0.6545 | 0.5568 | 0.0977 | 0.0656 to 0.1286 |
| ED | GaussianNB | 0.5745 | 0.4973 | 0.0772 | 0.0472 to 0.0992 |

All 14 CIs exclude zero. Each is a separate, unadjusted per-model interval. On Demo hosp the
threshold carried over from the training folds can collapse: GaussianNB has mean test specificity
0.0000 (so its PPV_std equals π_ref exactly) and GradientBoosting 0.0346 (columns `spec_mean_demo`,
`sens_mean_demo` of the same file). With 14 deaths in the whole Demo hosp cohort, each test fold holds
only 2 or 3 deaths.

### 1.5 Is the real Demo a typical Demo-sized draw? (trustworthiness nulls)

Source: [`Results/compare/<track>/<pool>/exp3_decision_stability*.csv`](Results/compare/hosp/trustworthiness/exp3_decision_stability.csv),
[`Results/compare/trustworthiness_selection_regret.csv`](Results/compare/trustworthiness_selection_regret.csv).

For each track, 1,000 Demo-sized samples of whole Full patients are drawn and evaluated with exactly
the procedure used on the real Demo (same 7 models, 5-fold patient-grouped CV, sigmoid calibration,
AUROC as the selection metric). Two sampling schemes (random; prevalence-matched to the Demo's number
of positive rows) and two patient pools (all Full patients; Full patients with an ICU stay) give four
nulls per track. Note that this analysis uses its own evaluation protocol (Section 3.7), so its
Full-best model (by AUROC) is not the RSCE leader of Section 1.3.

| Track | Pool | Null | P(draw picks the Full-best model) | Wilson 95% CI | P(Full-best in draw's top 3) |
|---|---|---|---:|---|---:|
| hosp | all patients | random | 0.0280 | 0.0194 to 0.0402 | 0.1780 |
| hosp | all patients | prevalence-matched | 0.0650 | 0.0513 to 0.0820 | 0.3010 |
| hosp | ICU patients | random | 0.0490 | 0.0373 to 0.0642 | 0.2750 |
| hosp | ICU patients | prevalence-matched | 0.0890 | 0.0729 to 0.1083 | 0.3720 |
| ED | all patients | random | 0.0760 | 0.0611 to 0.0941 | 0.3120 |
| ED | all patients | prevalence-matched | 0.1010 | 0.0838 to 0.1212 | 0.3970 |
| ED | ICU patients | random | 0.0860 | 0.0702 to 0.1050 | 0.3420 |
| ED | ICU patients | prevalence-matched | 0.1070 | 0.0893 to 0.1277 | 0.3560 |

* The Full-best model by AUROC is GradientBoosting (hosp, Full AUROC 0.8933) and MLP (ED, 0.8002).
  The real Demo picks RandomForest (hosp) and Logistic_L2 (ED), i.e. it misses the Full-best model,
  as do most null draws.
* A uniform random choice among 7 models would pick the Full-best model with probability 1/7 =
  0.1429. Every upper Wilson bound in the table is below 1/7.
* **Selection regret** (Full AUROC of the Full-best model minus Full AUROC of the model a draw
  selects): the real Demo's regret is 0.0121 (hosp) and 0.0140 (ED). Its percentile within the null
  regret distributions is 31.70 to 34.75 (hosp) and 58.00 to 64.90 (ED) across the four nulls, i.e.
  the real Demo's choice is neither unusually good nor unusually bad for a sample of its size. The
  probability that a null draw's regret is at most 0.01 is 0.0850 to 0.1540 (hosp) and 0.3760 to
  0.4420 (ED). Mean regret is 0.0274 to 0.0360 (hosp) and 0.0100 to 0.0173 (ED).
* Feature-importance agreement (exp4): the Spearman correlation between Demo and Full permutation
  importances is -0.2645 in hosp, at the 0.5 to 2.5 percentile of the four null distributions
  (two-sided empirical p 0.0120 to 0.0519), and 0.2673 in ED, at the 36.4 to 68.3 percentile
  (p 0.6354 to 0.8032) ([`exp4_importance_summary*.csv`](Results/compare/hosp/trustworthiness/exp4_importance_summary.csv)).

### 1.6 Cross-track summary

Both tracks agree on the direction (Full > Demo for every model, driven by R) and disagree with each
other only in size: the hosp gap and the hosp rank disagreement are larger than in ED. Neither track's
Demo identifies its Full leader ([`cross_domain_summary.md`](Results/compare/cross_domain/cross_domain_summary.md)).

---

## 2. What these results do not establish

* **The Demo-to-Full gap is not attributed to a single cause.** Sample size, outcome prevalence and
  the Demo's ICU-only patient selection all differ between releases. The prevalence-matched and
  ICU-pool nulls address the last two for the selection question (Section 1.5); the RSCE gap itself
  (Section 1.2) is reported as observed and is not decomposed by cause.
* **Two tasks, one release version.** The findings are established for in-hospital mortality and ED
  disposition on MIMIC-IV v2.2 and MIMIC-IV-ED v2.2, with the seven fixed model configurations listed
  in Section 3.4 (no hyperparameter tuning). Other outcomes, model families or tuned models were not
  tested.
* **Perturbation worlds are synthetic.** The ten RSCE worlds are controlled perturbations of the test
  fold (Section 3.5). They measure sensitivity to those perturbations, not real-world deployment
  shift. Within this repository, the correlation between RSCE and real degradation features is not
  significant after Holm adjustment in any of the 32 tests
  ([`degradation_analysis/RSCE_vs_realworld_degradation_correlation.csv`](Results/hosp_full/degradation_analysis/RSCE_vs_realworld_degradation_correlation.csv)).
* **"Full-best" is a point estimate.** On Full, several models are statistically tied at the top
  (Section 1.3). "The Demo misses the Full leader" therefore means it misses the model with the
  highest point estimate; regret (Section 1.5) quantifies how much that costs in Full AUROC.
* **Multiplicity.** Per-model CIs (Sections 1.2, 1.4) are separate unadjusted intervals; tests that
  are Holm-adjusted are labelled as such.

---

## 3. Study design

### 3.1 Data

The official PhysioNet releases, unchanged: MIMIC-IV v2.2 (`hosp`, `icu` modules) and MIMIC-IV-ED v2.2,
each as Demo and Full. Raw files are read directly (`.csv.gz`), and the SHA-256 of the ICU stays file is
checked against the official v2.2 checksum.

### 3.2 Cohorts and labels

**hosp** ([`prepare_hosp.py`](Pipeline/01_prepare_data/prepare_hosp.py)): all admissions of adult patients
(anchor_age ≥ 18); label = `hospital_expire_flag`. Prediction time T = admittime + 24 h. Only labs charted
up to T are used, and admissions that had already ended (discharge or death) by T are excluded, so every
included admission has the same 24-hour observation window and no lab drawn after T can leak the outcome.
Exclusion counts by label are in `Results/hosp_*/*.cohort_flow.json`.

**ED** ([`prepare_ed.py`](Pipeline/01_prepare_data/prepare_ed.py)): ED stays; label `label_ed_admit` = 1
when the stay resulted in a hospital encounter (`edstays.hadm_id` populated: inpatient admission or
hospital observation). Prediction time T = ED arrival + 1 h; stays that had left the ED or been admitted
by T are excluded; vital signs and medication events are restricted to charttime ≤ T. The ED `diagnosis`
table is deliberately not used (diagnoses are coded after discharge).

### 3.3 Features

| Track | Features (count) | Removed before modelling |
|---|---|---|
| hosp | 30 lab medians (the 30 most frequent lab itemids in Full; Demo uses exactly the same list), `anchor_age`, and 6 categorical fields: gender, race, marital_status, insurance, admission_type, admission_location (37) | hadm_id, subject_id, discharge_location (post-outcome), anchor_year, anchor_year_group |
| ED | 8 triage fields, 19 vital-sign summaries (mean/min/max of 6 signs, number of readings), number of home and ED medications, `anchor_age`, gender (31) | stay_id, subject_id |

The feature lists actually used are recorded in `Results/<dataset>/rsce/schema.json`.

### 3.4 Models and cross-validation

Seven fixed scikit-learn models, identical for Demo and Full: Logistic_L2, RandomForest (1,200 trees),
ExtraTrees (1,200 trees), GradientBoosting (700 estimators), SVC_RBF, MLP (256-128-64 with early
stopping on validation log-loss of a stratified 10% split of the training fold, best epoch restored)
and GaussianNB. Exact parameters: `model_params` in `schema.json`. Preprocessing (numeric: median imputation and
standard scaling; categorical: most-frequent imputation and one-hot encoding) is fitted inside each
training fold.

Evaluation: 5-fold × 3-repeat **StratifiedGroupKFold** grouped by `subject_id` (no patient in both
train and test), seed 42, 15 test folds per model. SVC training folds are capped at 20,000 rows
(`--svc_max_train_n 20000`; no effect at Demo size).

### 3.5 RSCE benchmark ([`run_rsce.py`](Pipeline/02_rsce_benchmark/run_rsce.py))

Each trained model is evaluated on ten versions ("worlds") of its test fold; models are always trained
on clean data.

| World | Perturbation (test fold only) |
|---|---|
| WA_clean | none |
| WB_noise_outliers | Gaussian noise (0.2 SD) plus 2% outliers on numeric features |
| WC_missingness | MCAR/MAR missingness (base 0.10, extra 0.15) |
| WD_shift | additive mean shift (0.3 SD) |
| WE_surrogate_corrupt | surrogate corruption (γ = 0.5) |
| WF_nonlinear | signed-log1p nonlinear distortion (α = 0.6) |
| WH_subgroup_shift | shift applied to a subgroup (0.35 SD) |
| WI_prevalence_shift | resampling to prevalence 0.35 |
| WJ_concept_drift | concept drift (k = 3, 0.35 SD) |
| WG_label_noise | 10% label noise (reported as Q_label; not part of RSCE) |

Components per (fold, model):

* **R** = AUROC on the clean world.
* **S** (S_ratio) = mean over the 8 covariate worlds (all except clean and label noise) of
  min(1, AUROC_world / AUROC_clean).
* **C** (C_linear) = 1 − mean over the same 8 worlds of min(1, |ECE_world − ECE_clean|).
* **E** (E_mix) = mean of cosine, rank and top-k Jaccard similarity between clean-world and
  perturbed-world mean |SHAP| importance vectors, over the worlds except clean, label noise and
  prevalence shift, computed on min(50, test-fold size) stratified, model-independent rows. E exists
  only for the models SHAP explains here (RandomForest, ExtraTrees, GradientBoosting, Logistic_L2).

**RSCE = 0.4 R + 0.3 S + 0.2 C + 0.1 E.** For a model without E, the available weights are renormalized:
(0.4 R + 0.3 S + 0.2 C) / 0.9. Reported alongside for every model:

* **RSCE_RSC** = (0.4 R + 0.3 S + 0.2 C) / 0.9 for all seven models (like-for-like comparison);
* **RSCE_legacy** = 0.4 R + 0.3 S_ratio + 0.2 C_exp + 0.1 E with missing E set to 0
  (C_exp = mean exp(−|ΔECE| / ECE_clean)); kept as a sensitivity variant.

Pairwise model comparisons use the Nadeau-Bengio corrected resampled t-test with Holm adjustment (also
Wilcoxon and naive t-test). Component ablations (S_drop vs S_ratio, C_exp vs C_linear, E variants) and a
weight sensitivity sweep are in `ablation_*.csv`.

### 3.6 Demo-vs-Full comparison ([`03_compare_demo_vs_full/`](Pipeline/03_compare_demo_vs_full/))

* per-model ΔRSCE, sign test, Spearman/Kendall rank agreement;
* exact decomposition ΔRSCE = Σ w_k Δk (k = R, S, C, E, with the renormalized weights for models
  without E), checked to reproduce ΔRSCE to machine precision;
* unpaired fold-level tests (Welch, Mann-Whitney, Hedges g, bootstrap CI): Demo fold k and Full fold k
  are unrelated, so paired tests are not used across releases;
* per-metric, per-world deltas with bootstrap CIs; reliability-curve distances; world-by-world rank
  agreement matrices;
* RSCE vs real-degradation correlations (Spearman/Kendall with bootstrap CIs and permutation p-values,
  Holm-adjusted);
* cross-track synthesis.

### 3.7 Prevalence-standardized PPV ([`run_ppv.py`](Pipeline/04_ppv/run_ppv.py))

Same 7 models and the same 5 × 3 grouped CV. In each training fold a threshold reaching sensitivity
0.80 is chosen on inner (3-fold) out-of-fold predictions and applied unchanged to the test fold. PPV is
standardized to a common reference prevalence π_ref (the Full prevalence, for both Full and Demo):

PPV_std = sens · π_ref / (sens · π_ref + (1 − spec) · (1 − π_ref)).

The pooled estimator sums TP/FP/TN/FN over the 5 folds of a repeat before computing sensitivity and
specificity; values are averaged over the 3 repeats. [`compare_ppv.py`](Pipeline/04_ppv/compare_ppv.py)
compares Full and Demo per model with a bootstrap CI and checks both sides used identical model settings.

### 3.8 Trustworthiness nulls ([`compare_trustworthiness.py`](Pipeline/03_compare_demo_vs_full/compare_trustworthiness.py))

Question: is the real Demo a typical Demo-sized draw from Full? For each null, 1,000 samples of whole
Full patients are drawn until the sample has at least as many rows as the Demo (hosp 245, ED 219):

* **random**: patients drawn at random (draws that cannot support 5-fold CV are redrawn; the acceptance
  rate is recorded in `run_info.json`);
* **prevalence-matched**: patients with a positive row are drawn until the Demo's number of positive
  rows is reached, then negative-only patients;
* **pool all**: all Full patients; **pool icu**: only Full patients with ≥ 1 ICU stay (list built by
  [`make_icu_subject_list.py`](Pipeline/01_prepare_data/make_icu_subject_list.py) from the official
  `icu/icustays.csv.gz`; it selects patients only and is never a feature).

Each draw and the real Demo are evaluated identically (7 models, 5-fold patient-grouped CV, sigmoid
calibration with group-aware inner splits). Experiments: exp1 Demo metric percentiles in the null; exp2
rank correlation with the Full ordering; exp3 probability that a draw selects the Full-best model (Wilson
CI) and top-k membership; exp4 agreement of cross-validated permutation importances with Full. The ICU
pool additionally computes a Full-ICU reference (`*_vs_fullicu*.csv`). Each draw uses a generator seeded
by (seed, null mode, draw index), so results do not depend on order, parallelism or resumption.

### 3.9 Derived analyses (no model fitting)

* [`trustworthiness_selection_regret.py`](Pipeline/03_compare_demo_vs_full/trustworthiness_selection_regret.py):
  selection regret of every null draw and of the real Demo; re-derives the exp3 hit probability and
  asserts equality with `exp3_decision_stability*.csv`.
* [`rsce_rsc_rank_agreement.py`](Pipeline/03_compare_demo_vs_full/rsce_rsc_rank_agreement.py): leaders,
  cross-ranks and rank agreement under RSCE and RSCE_RSC.

---

## 4. Repository layout

```
.
├── README.md
├── requirements.txt            pinned versions (pip)
├── environment.yml             pinned versions (conda)
├── MIMIC_Dataset/              empty: place the PhysioNet releases here (see its README)
├── Pipeline/
│   ├── run_full_pipeline.py    one-command orchestrator (all stages, both tracks)
│   ├── capture_environment.py  writes environment_lock.json (exact package versions)
│   ├── 01_prepare_data/
│   │   ├── prepare_hosp.py             hosp analytic dataset (landmark 24 h)
│   │   ├── prepare_ed.py               ED analytic dataset (landmark 1 h)
│   │   ├── full_lab_itemids.csv        header-only list of the 30 Full lab features
│   │   ├── check_demo_subset_of_full.py
│   │   ├── make_icu_subject_list.py
│   │   └── make_table1.py
│   ├── 02_rsce_benchmark/
│   │   └── run_rsce.py                 RSCE multi-world benchmark
│   ├── 03_compare_demo_vs_full/
│   │   ├── make_compare_base.py        gathers Demo and Full RSCE outputs into one folder
│   │   ├── compare_pro.py              ΔRSCE, sign test, rank agreement, decomposition, CIs
│   │   ├── compare_addons.py           ablations, paired-test table, per-fold tests, reliability
│   │   ├── compare_world_heatmap.py    world-by-world rank agreement
│   │   ├── compare_rsce_vs_degradation.py
│   │   ├── compare_trustworthiness.py  Demo-sized null distributions
│   │   ├── trustworthiness_selection_regret.py
│   │   ├── rsce_rsc_rank_agreement.py
│   │   └── synthesize_cross_domain.py
│   ├── 04_ppv/
│   │   ├── run_ppv.py                  prevalence-standardized PPV
│   │   └── compare_ppv.py
│   └── 05_figures/
│       └── make_figures.py             Figures 1 to 5 from Results/ only
├── Results/                    aggregate results of the reported runs (Section 8)
└── figures/                    Figures 1 to 5 (PNG, 600 dpi), source data, manifest
```

`full_lab_itemids.csv` contains only a header row (the 30 `lab_<itemid>` column names of the Full hosp
dataset, in order). It lets a user without Full access build the Demo hosp dataset with exactly the Full
feature set. It contains no data.

---

## 5. Data access

* **Demo releases** (open access, no credentials):
  [MIMIC-IV Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/),
  [MIMIC-IV-ED Demo v2.2](https://physionet.org/content/mimic-iv-ed-demo/2.2/).
* **Full releases** (credentialed access, CITI training and the PhysioNet Credentialed Health Data Use Agreement):
  [MIMIC-IV v2.2](https://physionet.org/content/mimiciv/2.2/),
  [MIMIC-IV-ED v2.2](https://physionet.org/content/mimic-iv-ed/2.2/).

Place them under `MIMIC_Dataset/` with the folder names given in
[`MIMIC_Dataset/README.md`](MIMIC_Dataset/README.md).

**What this repository does not contain, by design:** no MIMIC source file, no row-level analytic
dataset (Demo or Full), no patient identifier list, and no per-patient prediction. `Results/` holds
aggregates only (per-fold and per-model metrics, test statistics, counts). `.gitignore` keeps the
row-level files that the pipeline writes out of version control.

---

## 6. Installation

Python 3.11 was used for the reported runs (`Results/environment_lock.json`: Python 3.11.16,
numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.1, statsmodels 0.15.0,
shap 0.51.0).

```bash
# conda
conda env create -f environment.yml
conda activate mimic-demo-full-audit

# or pip
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Exact package versions matter: patch-level changes in numpy, pandas or scikit-learn can move RSCE values
in the last decimals. `capture_environment.py` records the versions of any new run.

---

## 7. How to reproduce

All commands are run from the repository root unless stated otherwise. Three levels are possible,
depending on data access.

### 7.1 Level A: from the shipped `Results/` (no MIMIC data needed)

Everything downstream of model fitting can be recomputed from the aggregate files in this repository.

```bash
# 1) rebuild the per-track comparison input folders (copies of the Demo and Full RSCE outputs)
python Pipeline/03_compare_demo_vs_full/make_compare_base.py --demo_dir Results/hosp_demo/rsce --full_dir Results/hosp_full/rsce --outdir Results/compare/hosp/_base
python Pipeline/03_compare_demo_vs_full/make_compare_base.py --demo_dir Results/ed_demo/rsce   --full_dir Results/ed_full/rsce   --outdir Results/compare/ed/_base

# 2) Demo-vs-Full comparisons (shown for hosp; repeat with ed)
python Pipeline/03_compare_demo_vs_full/compare_pro.py    --base Results/compare/hosp/_base --outdir Results/compare/hosp --dataset_tag hosp
python Pipeline/03_compare_demo_vs_full/compare_addons.py --base Results/compare/hosp/_base --outdir Results/compare/hosp
python Pipeline/03_compare_demo_vs_full/compare_world_heatmap.py --input Results/compare/hosp/metrics_aggregated_deltas_demo_vs_full.csv --outdir Results/compare/hosp --dataset_tag hosp
python Pipeline/04_ppv/compare_ppv.py --full_per_fold Results/hosp_full/ppv/ppv_std_per_fold.csv --demo_per_fold Results/hosp_demo/ppv/ppv_std_per_fold.csv --outdir Results/compare/hosp/ppv
python Pipeline/03_compare_demo_vs_full/compare_rsce_vs_degradation.py --rsce_scores Results/hosp_full/rsce/rsce_scores.csv --metrics_aggregated Results/hosp_full/rsce/metrics_aggregated.csv --outdir Results/hosp_full/degradation_analysis --dataset_tag hosp_full

# 3) cross-track synthesis and derived analyses
python Pipeline/03_compare_demo_vs_full/synthesize_cross_domain.py --hosp_compare_dir Results/compare/hosp --ed_compare_dir Results/compare/ed --outdir Results/compare/cross_domain
python Pipeline/03_compare_demo_vs_full/trustworthiness_selection_regret.py --results Results
python Pipeline/03_compare_demo_vs_full/rsce_rsc_rank_agreement.py --results Results

# 4) figures
python Pipeline/05_figures/make_figures.py --root . --outdir figures
```

These commands overwrite the shipped files with recomputed ones. `git diff --stat Results` afterwards
shows whether anything changed (Section 9 reports the result of this check).

### 7.2 Level B: Demo end to end (open-access data only)

Place the two Demo releases under `MIMIC_Dataset/`, then:

```bash
cd Pipeline
python run_full_pipeline.py --demo_only --stage prepare   # analytic datasets (hosp uses full_lab_itemids.csv)
python run_full_pipeline.py --demo_only --stage rsce      # RSCE on Demo, both tracks
```

For the PPV, the reported Demo runs are standardized to the **Full** prevalence and use the full model
zoo. Without Full data, pass the reported π_ref (the `pi_ref` field of
`Results/hosp_full/ppv/ppv_run_info.json` and `Results/ed_full/ppv/ppv_run_info.json`) explicitly and do
not use `--fast`:

```bash
python 04_ppv/run_ppv.py --data ../Results/hosp_demo/demo_analytic_dataset_mortality_all_admissions.csv \
    --target label_mortality --outdir ../Results/hosp_demo/ppv --pi_ref 0.02204683518314996 \
    --drop_cols hadm_id stay_id subject_id discharge_location anchor_year anchor_year_group --svc_max_train_n 20000
python 04_ppv/run_ppv.py --data ../Results/ed_demo/demo_ed_analytic_dataset_admission.csv \
    --target label_ed_admit --outdir ../Results/ed_demo/ppv --pi_ref 0.4783936596916832 \
    --drop_cols hadm_id stay_id subject_id discharge_location anchor_year anchor_year_group --svc_max_train_n 20000
```

(`run_full_pipeline.py --demo_only --stage ppv` is a quicker plumbing test: it uses the Demo's own
prevalence and `--fast`, so its PPV numbers are not the reported ones.)

### 7.3 Level C: full pipeline (credentialed access)

```bash
cd Pipeline
python run_full_pipeline.py --list         # show the plan
python run_full_pipeline.py                # prepare Demo and Full, subset checks, feasibility estimates; stops before the long runs
python run_full_pipeline.py --run_full     # everything: RSCE, PPV, comparisons, both trustworthiness pools,
                                           # Table 1, cross-track synthesis, derived analyses, figures
```

Stages, in order: `prepare`, `checks`, `estimate`, `rsce`, `ppv`, `compare`, `extras`. A single stage can
be (re)run with `--stage <name>`. RSCE, PPV and the trustworthiness nulls checkpoint every finished unit
under `<outdir>/_checkpoint/`; re-running the same command resumes, and a changed setting, data file or
package version is refused rather than mixed in. Useful options: `--trust_n_jobs` (worker processes for
the null draws), `--svc_max_train_n` (default 20000), `--shap_samples` (default 50), `--continue_on_error`.
Every command, exit code and wall time is appended to `Results/run_full_pipeline.log`. The `extras`
stage ends by rewriting `Results/environment_lock.json` with the versions of the current environment.

The Full-scale runs are long (hours to days on a workstation, dominated by RandomForest/ExtraTrees SHAP
and the 4 × 1,000 null draws per track). Run `python run_full_pipeline.py` first and read the
`estimate_timing.csv` files it writes under `Results/*/rsce_estimate` and `Results/*/ppv_estimate`.

The ICU-patient null can also be run on its own:

```bash
python 01_prepare_data/make_icu_subject_list.py --icustays ../MIMIC_Dataset/MIMIC-IV-Full-2.2/icu/icustays.csv.gz \
    --out ../Results/checks/full_icu_subject_ids.csv
python 03_compare_demo_vs_full/compare_trustworthiness.py \
    --full_path ../Results/hosp_full/full_analytic_dataset_mortality_all_admissions.csv \
    --demo_path ../Results/hosp_demo/demo_analytic_dataset_mortality_all_admissions.csv \
    --target_col label_mortality --svc_max_train_n 20000 --n_jobs 8 \
    --null_pool icu --icu_subjects_path ../Results/checks/full_icu_subject_ids.csv \
    --outdir ../Results/compare/hosp/trustworthiness_icu
```

(ED: `--full_path ../Results/ed_full/full_ed_analytic_dataset_admission.csv --demo_path
../Results/ed_demo/demo_ed_analytic_dataset_admission.csv --target_col label_ed_admit --outdir
../Results/compare/ed/trustworthiness_icu`.)

---

## 8. Result files reference

`<ds>` is one of `hosp_demo`, `hosp_full`, `ed_demo`, `ed_full`; `<track>` is `hosp` or `ed`.

### 8.1 Per dataset: `Results/<ds>/`

| File | Content |
|---|---|
| `*.cohort_flow.json` | cohort construction counts: raw units, exclusions at the landmark by label, final rows, label prevalence, missingness |
| `rsce/rsce_scores.csv` | final scores per model: RSCE_full, has_E, RSCE_RSC, RSCE_legacy |
| `rsce/rsce_per_fold.csv` | R, Q_label, S and C variants, E variants, fold RSCE and RSCE_RSC per (fold, model) |
| `rsce/metrics_per_fold.csv` | AUROC, Brier (+ REL/RES/UNC decomposition), LogLoss, ECE, aECE per (fold, model, world) |
| `rsce/metrics_aggregated.csv` | means and bootstrap CIs of the above per (model, world) |
| `rsce/paired_tests.csv` | all 21 model pairs: Nadeau-Bengio corrected t, Wilcoxon, naive t, each with Holm-adjusted p |
| `rsce/ablation_components_per_fold.csv`, `ablation_summary.csv`, `ablation_rank_agreement.csv`, `E_ablation_per_fold.csv` | component variants and how the ranking changes under each |
| `rsce/reliability_curve_points.csv` | reliability-diagram bins (mean prediction, observed rate, count) |
| `rsce/compute_cost.csv`, `compute_cost_per_fold.csv` | fit, prediction and SHAP time |
| `rsce/schema.json` | features, dropped columns, CV, model parameters, worlds, scoring settings, package versions |
| `ppv/ppv_std_per_fold.csv` | threshold, confusion counts, observed and standardized PPV per (fold, repeat, model) |
| `ppv/ppv_std_per_repeat.csv` | pooled confusion counts and PPV_std per (model, repeat) |
| `ppv/ppv_std_aggregated.csv` | per-model summary over repeats |
| `ppv/ppv_run_info.json` | π_ref, settings, model parameters |
| `degradation_analysis/RSCE_vs_realworld_degradation_correlation.csv` | correlation of RSCE with degradation features (bootstrap CI, permutation p, Holm) |
| `degradation_analysis/worst_worlds_by_metric.csv` | worst world per model and metric |

### 8.2 Per track: `Results/compare/<track>/`

| File | Content |
|---|---|
| `rsce_comparison_demo_vs_full.csv` | per-model RSCE, RSCE_RSC, RSCE_legacy on both releases, deltas, ranks |
| `sign_test_summary.csv`, `rank_agreement.csv` | sign test on ΔRSCE; Spearman/Kendall of the two rankings |
| `delta_rsce_decomposition.csv` | exact R/S/C/E contributions to ΔRSCE with reconstruction check |
| `rsce_full_vs_demo_unpaired_tests.csv` | fold-level unpaired tests (Welch, Mann-Whitney, Hedges g, CI, Holm) |
| `metrics_aggregated_deltas_demo_vs_full.csv` | per (model, world) metric means on both releases and their deltas |
| `metrics_per_fold_tests_demo_vs_full.csv`, `metrics_per_fold_summary_demo_vs_full.csv` | per-cell unpaired tests and their summary per metric |
| `bootstrap_CI_per_model.csv`, `bootstrap_CI_per_world.csv` | bootstrap CIs of mean metric deltas |
| `ablation_summary_demo_vs_full.csv`, `ablation_rank_agreement_demo_vs_full.csv`, `paired_tests_file_demo_vs_full.csv`, `compute_cost_compare.csv` | side-by-side tables |
| `reliability_curve_distance*.csv` | distance between Demo and Full reliability curves |
| `world_rank_spearman_matrix_{demo,full}.csv` | model-rank agreement between every pair of worlds |
| `ppv/per_model_full_vs_demo_unpaired.csv` | PPV_std Full vs Demo per model with bootstrap CIs and descriptive tests |
| `trustworthiness/`, `trustworthiness_icu/` | null analysis for the all-patient and ICU-patient pools (below) |

Trustworthiness folders (files ending `_prevmatched` are the prevalence-matched null; files containing
`_vs_fullicu` use the Full-ICU reference and exist only in `trustworthiness_icu/`):

| File | Content |
|---|---|
| `demo_metrics.csv`, `full_reference_metrics.csv`, `full_icu_reference_metrics.csv` | AUROC, AUPRC, LogLoss, Brier, ECE of the 7 models on the Demo and on the whole Full (or Full-ICU) data |
| `subsample_metrics_long*.csv` | the same metrics for every null draw (1,000 × 7 rows) with draw size, positives, patients, attempts |
| `exp1_demo_percentiles*.csv` | percentile of each Demo metric in the null |
| `exp2_rank_stability*.csv` | Spearman of the model ordering with Full; top-k overlap |
| `exp3_decision_stability*.csv` | Full-best vs Demo-best; null probability of picking the Full-best (Wilson CI); top-k |
| `exp4_importance_stability*.csv`, `exp4_importance_summary*.csv`, `importance_*.csv` | permutation importance agreement |
| `run_info.json` | settings, acceptance rates, achieved size and prevalence of the draws |

### 8.3 Other

| File | Content |
|---|---|
| `Results/compare/cross_domain/cross_domain_summary.{csv,md}` | side-by-side summary of both tracks |
| `Results/compare/trustworthiness_selection_regret.csv` | regret statistics per track, pool and null |
| `Results/compare/rsce_rsc_rank_agreement.csv` | leaders and agreement under RSCE and RSCE_RSC |
| `Results/checks/demo_subset_check_{hosp,ed}.json` | Demo-in-Full patient overlap |
| `Results/checks/full_icu_subject_ids.json` | provenance of the ICU patient list (source checksum, counts); the list itself is not shipped |
| `Results/table1/table1_combined.csv`, `table1.md` | baseline characteristics of the four datasets |
| `Results/environment_lock.json` | exact platform and package versions |
| `figures/Fig*.png`, `figures/source_data/Fig*_source_data.csv`, `figures/figure_manifest.json` | figures, every plotted number with its source file, SHA-256 of inputs and outputs |

Absolute paths of the original machine inside `run_info.json`, `ppv_run_info.json` and the subset-check
files are replaced by `<PROJECT_ROOT>`. No other value in `Results/` is edited.

The figures: Fig. 1 study design and cohorts; Fig. 2 RSCE on Full vs Demo and the ΔRSCE decomposition;
Fig. 3 rank reproducibility; Fig. 4 prevalence-standardized PPV; Fig. 5 recovery of the Full-best model
in the trustworthiness nulls. `make_figures.py` also writes PDF and TIFF versions; only PNG is included
here.

---

## 9. Reproducibility checks performed on this repository

Run on a clean copy of this repository (Linux, Python 3.13, the package versions of Section 6):

* **Level A.** All 50 comparison, synthesis and derived CSV/Markdown outputs recomputed from the shipped
  `Results/` (Section 7.1) agree with the shipped files (identical strings; numeric columns equal within a
  relative tolerance of 1e-9). The `_base` folders rebuilt by `make_compare_base.py` are byte-identical
  copies of the RSCE outputs. `make_figures.py` runs to completion with all of its 105 internal cross-file
  consistency checks passing, and reproduces the figure source data.
* **Level B, data preparation.** Rebuilding the Demo datasets from the open-access Demo releases with
  `run_full_pipeline.py --demo_only --stage prepare` reproduces the ED Demo analytic dataset byte for
  byte and the hosp Demo dataset value for value (same 245 × 43 table, same column order; the shipped run
  was written with Windows line endings). Both cohort-flow summaries are identical in content.
* **Level B, RSCE.** Re-running the Demo RSCE benchmark (`--demo_only --stage rsce`, about 4 min for hosp
  and 6 min for ED on 2 CPU cores) reproduces 5 of 7 models exactly in ED and 6 of 7 in hosp (every
  per-fold, per-world AUROC identical). The remaining models differ slightly: GradientBoosting in both
  tracks (RSCE difference 0.0001 in hosp, 0.0005 in ED) and SVC_RBF in ED (0.0002); single-fold AUROC
  values of these models differ by at most 0.0145 (hosp GradientBoosting), 0.0229 (ED GradientBoosting)
  and 0.0333 (ED SVC_RBF). Both models are configured deterministically (fixed `random_state`, no row or
  feature subsampling), so the differences are attributed to the platform (the reported runs used
  Windows and Python 3.11.16; the check used Linux and Python 3.13); this attribution was not tested
  further. The model ranking under RSCE and under RSCE_RSC is identical to the shipped one in both
  tracks, and 0 of 21 model pairs are separated in either run.
* **Full-scale runs** (Full RSCE, Full PPV, the trustworthiness nulls) were not re-executed for these
  checks; their outputs are the shipped results of the reported runs.

---

## 10. Data citations

If you use MIMIC data, cite the releases as required by PhysioNet:

* Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., and Mark, R. MIMIC-IV (version 2.2).
  PhysioNet (2023). https://doi.org/10.13026/6mm1-ek67
* Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., and Horng, S. MIMIC-IV-ED (version 2.2).
  PhysioNet (2023). https://doi.org/10.13026/5ntk-km72
* Johnson, A. E. W., Bulgarelli, L., Shen, L., et al. MIMIC-IV, a freely accessible electronic health
  record dataset. Scientific Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x
* Goldberger, A. L., et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 101(23), e215 to e220
  (2000).
* Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., and Mark, R. MIMIC-IV Clinical
  Database Demo (version 2.2). PhysioNet (2023). https://doi.org/10.13026/dp1f-ex47
* Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Horng, S., and Mark, R. MIMIC-IV-ED Demo
  (version 2.2). PhysioNet (2023). https://doi.org/10.13026/jzz5-vs76
