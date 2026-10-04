# Cross-domain synthesis: Hospital module vs ED module

Auto-generated from `compare_pro.py` outputs for each domain. This is descriptive (no new hypothesis test across domains -- domains use different cohorts/targets and aren't directly poolable), meant to make it easy to see, at a glance, whether the Demo-vs-Full pattern generalizes.

## Direction of ΔRSCE (Full - Demo)

- **Hospital**: mean ΔRSCE = 0.1064, median = 0.0976, range [0.0617, 0.1739]; 100% of models have Full > Demo -> Full systematically HIGHER than Demo.
- **ED**: mean ΔRSCE = 0.0639, median = 0.0596, range [0.0247, 0.0921]; 100% of models have Full > Demo -> Full systematically HIGHER than Demo.

**Cross-domain consistency (based on the sign tests, not the sign of the mean):** CONSISTENT: both domains show the same systematic direction.


## Which component drives ΔRSCE (mean weighted contribution, exact decomposition)

- **Hospital**: R=+0.1014, S=+0.0069, C=-0.0020, E=+0.0002
- **ED**: R=+0.0564, S=+0.0022, C=+0.0056, E=-0.0003

## Rank agreement (Spearman, Demo vs Full)

- **Hospital**: Spearman rho = -0.071
- **ED**: Spearman rho = 0.429

## Best model: does Demo pick the same winner as Full?

- **Hospital**: Full-best=MLP, Demo-best=RandomForest -> MISMATCH.
- **ED**: Full-best=GradientBoosting, Demo-best=Logistic_L2 -> MISMATCH.

## Sign test on ΔRSCE (two-sided binomial)

- **Hospital**: 7 positive / 0 negative deltas, p=0.0156.
- **ED**: 7 positive / 0 negative deltas, p=0.0156.

---
*Interpretation note:* Hospital-module and ED-module datasets have different cohorts, targets, prevalence and feature sets, so this is a consistency check on the DIRECTION and QUALITATIVE pattern of the Demo-vs-Full relationship, not a pooled statistical test. If the two domains agree, that's evidence the Demo-vs-Full finding generalizes beyond a single dataset; if they disagree, that is itself a finding worth reporting rather than a bug to fix.