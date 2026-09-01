> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Bayesian Fit Score

Hierarchical Beta-Binomial scoring model that turns sparse count data into a calibrated 1-5 fit score.

## Why Bayesian?

Naive rates are misleading with small samples. A program with 7/9 positives (78%) is **not** more receptive than one with 151/280 (54%) — the first estimate has a 95% CI spanning 40%–97%. This model applies Bayesian shrinkage (partial pooling) to automatically correct for small-sample noise.

## Architecture

```
Excel inputs → signal extraction → PyMC hierarchical model → calibrated 1-5 score
```

**Three signals:**
- Geographic receptivity rate (θ_geo)
- Composition pipeline match (θ_comp)
- Trajectory-intent lift (δ_traj)

**Model:**
Each program's true rate θ_i ~ Beta(α, β), where α and β are learned from all programs jointly via a hyperprior. Small-n programs are pulled toward the population mean; large-n programs are barely moved.

## Setup

```bash
pip install -r requirements.txt
python main.py --data sample_data.xlsx
```

## Output

```
Program  n    Naive   Posterior  CI_lo  CI_hi  Score
Prog-001 312  0.82    0.81       0.76   0.86   5
Prog-003   9  0.78    0.61       0.37   0.84   3 (!)
```

Programs with n < 30 are flagged with a wide-CI warning.

## Deliverables (production engagement)

1. `score()` — deployable Python function with serialized posterior
2. Plain-language write-up: every weight, cutoff, and assumption explained
3. Validation report: calibration plot, AUC, Brier score

## Author

Dr. Sandeep Grover — PhD Data Science | Charité Berlin, Uni Lübeck, Tübingen  
60+ peer-reviewed publications in clinical and health informatics
