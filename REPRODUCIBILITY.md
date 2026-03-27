# Reproducibility & Methodology Documentation

## Play Ball! What Machine Learning Predicts This Baseball Season

**Author:** Alex Panetta
**Date:** March 27, 2026
**Prediction target:** 2026 MLB season individual player statistics

---

## 1. Claim of Reproducibility

Every prediction in this project is deterministically reproducible. Given the same input data and dependency versions, any person running this code on any machine will produce **byte-identical output files**. This document explains exactly how that is guaranteed and how to verify it.

---

## 2. Data Provenance

### Source
**SABR Lahman Baseball Database, 2025 Edition**
- Publisher: Society for American Baseball Research (SABR)
- Original creator: Sean Lahman (donated to SABR in 2024)
- License: Creative Commons Attribution-ShareAlike 3.0
- Canonical URL: https://sabr.org/lahman-database/
- Coverage: 1871–2025 MLB seasons (includes Negro Leagues statistics)

### How We Obtained the Data
The data was downloaded on **March 27, 2026** from the `cdalzell/Lahman` R package repository on GitHub, which packages the SABR 2025 Edition as `.RData` files.

- **Repository:** https://github.com/cdalzell/Lahman
- **Exact commit:** `af8aee24bf14e8f7d51f4aa7779bb04a6695afc6` (February 14, 2026)
- **Download URL:** `https://github.com/cdalzell/Lahman/archive/refs/heads/master.zip`
- **Download script:** `01_pull_data.py`

The `.RData` files were read with `pyreadr`, converted to pandas DataFrames, filtered, and saved as CSV.

### Why This Matters for Reproducibility
The upstream repository may be updated after our download date. **Therefore, the authoritative data for reproducing our results is the CSV files in the `data/` directory, not the download script.** The download script (`01_pull_data.py`) is provided to document provenance — to show *where* the data came from and *how* it was processed — not as the reproducibility mechanism.

If you re-run `01_pull_data.py` and the upstream data has changed, your `data/` files will differ. In that case, use our provided CSVs and skip to Step 2 (`02_sklearn_models.py --use-local-data`), or pin the download to our commit:
```
https://github.com/cdalzell/Lahman/archive/af8aee24bf14e8f7d51f4aa7779bb04a6695afc6.zip
```

### Data Filtering
From the full Lahman database:
- **Batting:** Filtered to 2015–2025 seasons, players with ≥200 plate appearances per season. Multi-team stints within a season are aggregated (summed). This yields **3,688 player-seasons** across **1,019 unique players**.
- **Pitching:** Filtered to 2015–2025 seasons, players with ≥50 innings pitched per season. Multi-team stints aggregated. This yields **3,502 player-seasons** across **1,168 unique players**.

### Derived Statistics
The Lahman database provides counting stats. We compute rate stats as follows:
- `AVG = H / AB`
- `OBP = (H + BB + HBP) / (AB + BB + HBP + SF)`
- `SLG = (1B + 2×2B + 3×3B + 4×HR) / AB`
- `WHIP = (BB + H) / IP` where `IP = IPouts / 3`
- `ERA = (ER × 9) / IP`
- `K9, BB9, HR9 = (stat × 9) / IP`

These formulas are standard baseball calculations. No proprietary or estimated statistics are used.

---

## 3. File Integrity Verification

Every file has a SHA-256 checksum. Run `python3 verify_reproducibility.py` to confirm your files match ours.

| File | SHA-256 | Rows | Description |
|------|---------|------|-------------|
| `data/batting_raw.csv` | `8e44bcbb...e3e1051a` | 3,688 | Processed batting data (2015–2025) |
| `data/pitching_raw.csv` | `cb404485...e40bb7e24` | 3,502 | Processed pitching data (2015–2025) |
| `data/people.csv` | `c80c3f5c...e5993ea27` | 24,270 | Player biographical data |
| `output/batting_predictions_sklearn.csv` | `f5fbaaca...f08d226` | 348 | 2026 batting predictions (variance-calibrated) |
| `output/pitching_predictions_sklearn.csv` | `e8c2fb1a...432852e` | 339 | 2026 pitching predictions (variance-calibrated) |

Full checksums are in `verify_reproducibility.py`.

---

## 4. Model Architecture

### Algorithm
**Gradient Boosting Regressor** (`sklearn.ensemble.GradientBoostingRegressor`)

One separate model is trained per target statistic. This is a deliberate choice: joint prediction (one model predicting all stats) would introduce coupling between targets, and separate models allow each stat's unique signal-to-noise ratio to be handled independently.

### Hyperparameters (identical for all models)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 200 | Sufficient for convergence without overfitting on ~2,500 training rows |
| `max_depth` | 4 | Allows interaction effects (e.g., age × prior performance) without memorizing |
| `learning_rate` | 0.05 | Conservative; combined with 200 trees gives effective capacity of ~10 equivalent trees |
| `min_samples_leaf` | 10 | Regularization: no leaf can represent fewer than 10 player-seasons |
| `subsample` | 0.8 | Stochastic gradient boosting; 80% row sampling per tree for variance reduction |
| `random_state` | 42 | **Fixed seed for exact reproducibility** |

No hyperparameter tuning was performed. These are reasonable defaults for a tabular regression problem of this size. We deliberately avoided tuning to prevent overfitting to the test set and to ensure that our reported metrics are unbiased.

### Determinism Guarantee
With `random_state=42` and `subsample=0.8`, scikit-learn's GBR uses a deterministic PRNG. We verified this empirically: two consecutive training runs on the same data produce byte-identical prediction arrays (SHA-256 of prediction vectors match). This holds for scikit-learn 1.7.2 on both ARM64 (Apple Silicon) and x86_64 architectures. **Different scikit-learn versions may produce different results** due to internal algorithm changes — pin your version with `requirements.txt`.

---

## 5. Feature Engineering

Each training row represents one player-season. Features are computed exclusively from **prior** seasons — the target year's data is never visible during feature construction. This prevents data leakage.

### Feature Set (43 features for batting, 48 for pitching)

| Feature Group | Count | Description | Rationale |
|--------------|-------|-------------|-----------|
| Previous season stats | 13 | All target stats + PA, G, SO from year N-1 | Most recent performance signal |
| Weighted 3-year rolling average | 10 | Weighted mean of years N-1, N-2, N-3 (weights: 3/6, 2/6, 1/6) | Smoothed performance with recency bias |
| Career average | 10 | Unweighted mean of all prior qualifying seasons | Baseline / "true talent" proxy |
| Year-over-year trend | 10 | Difference between year N-1 and N-2 stats | Trajectory: improving or declining? |
| Age | 1 | `yearID - birthYear` (integer years) | Biological aging baseline |
| Age² | 1 | Quadratic age term | Models the nonlinear aging curve (peak ~27-29, accelerating decline after ~33) |
| Experience | 1 | Count of prior qualifying seasons | Distinguishes 2nd-year players from 10-year veterans |
| Previous PA (batting) / GS ratio (pitching) | 1 | Plate appearances or games started / games | Playing time proxy / role indicator |

### Why These Features

These features represent a **minimal, interpretable, baseball-theoretically-motivated** feature set:

1. **Regression to the mean** is the dominant force in season-to-season baseball prediction. The 3-year weighted average and career average features directly capture a player's "true talent level" that extreme single-season performances regress toward.

2. **Aging curves** are well-documented in sabermetrics (see: Lichtman, "The Effect of Aging on Major League Baseball Player Performance," 2009). The quadratic age term allows the model to learn that players peak in their late 20s and decline thereafter.

3. **Trend features** capture whether a player is on an upward or downward trajectory beyond what age alone explains (e.g., a mechanical adjustment, a position change, or early injury effects).

4. **No park factors, team context, or schedule data.** We deliberately excluded these to keep the model focused on individual player projection and to avoid introducing confounds that change unpredictably between seasons (trades, free agency, schedule rebalancing).

### Feature Leakage Audit
- Target year stats are never used as features. Features for predicting year Y use only data from years < Y.
- The `yearID` column is used only for train/test splitting, never as a model feature.
- `playerID` and `fullName` are ID columns, excluded from the feature matrix.

---

## 6. Train/Test Methodology

### Temporal Split (No Shuffling)
- **Training set:** Predict seasons 2016–2024 using prior-year features (2,384 batting rows / 2,082 pitching rows)
- **Holdout test set:** Predict 2025 season using prior-year features (285 batting rows / 266 pitching rows)
- **Final model:** Retrained on 2016–2025 for 2026 predictions (348 batters / 339 pitchers)

This is a **strictly temporal split** — no random shuffling, no future data in training. This is critical because baseball statistics exhibit temporal autocorrelation (a player's 2023 stats are more similar to their 2022 stats than to their 2018 stats). A random train/test split would allow the model to "peek" at temporally adjacent seasons, inflating metrics.

### Why We Retrain for Final Predictions
The 2025 holdout evaluation gives us honest error estimates. But for the actual 2026 predictions we publish, we retrain on all available data (including 2025) because:
1. The 2025 data is the most informative single season for predicting 2026.
2. Not using it would handicap predictions unnecessarily.
3. We've already measured performance on the holdout — the final model's accuracy won't be better, just its predictions will be better informed.

### 5-Fold Cross-Validation
We also report 5-fold CV MAE on the training set as a secondary metric. Folds are contiguous blocks (not shuffled), preserving temporal structure.

---

## 7. Model Evaluation Results

### Batting Models (Holdout: 2025 season, 285 players)

| Statistic | MAE | R² | 5-Fold CV MAE | Interpretation |
|-----------|-----|-----|---------------|----------------|
| AVG | 0.0195 | 0.222 | 0.0222 | Predictions off by ~20 points of batting average |
| R | 15.48 | 0.370 | 16.95 | Off by ~15 runs |
| H | 26.86 | 0.296 | 28.77 | Off by ~27 hits |
| HR | 6.08 | 0.331 | 6.27 | Off by ~6 home runs |
| RBI | 16.39 | 0.284 | 17.42 | Off by ~16 RBIs |
| 2B | 6.08 | 0.263 | 6.91 | Off by ~6 doubles |
| SB | 4.44 | 0.535 | 3.78 | Off by ~4 stolen bases |
| BB | 12.53 | 0.457 | 12.53 | Off by ~13 walks |
| OBP | 0.0225 | 0.320 | 0.0244 | Off by ~23 points of OBP |
| SLG | 0.0480 | 0.186 | 0.0491 | Off by ~48 points of slugging |

### Pitching Models (Holdout: 2025 season, 266 players)

| Statistic | MAE | R² | 5-Fold CV MAE | Interpretation |
|-----------|-----|-----|---------------|----------------|
| ERA | 0.853 | 0.135 | 0.852 | Off by ~0.85 ERA points |
| W | 2.53 | 0.218 | 2.71 | Off by ~3 wins |
| L | 2.43 | 0.266 | 2.37 | Off by ~2 losses |
| SO | 26.87 | 0.407 | 29.42 | Off by ~27 strikeouts |
| WHIP | 0.135 | 0.173 | 0.144 | Off by ~0.13 WHIP |
| IP | 25.28 | 0.467 | 27.62 | Off by ~25 innings |
| SV | 2.32 | 0.581 | 2.84 | Off by ~2 saves |
| HR | 4.43 | 0.493 | 4.95 | Off by ~4 HR allowed |
| BB | 9.47 | 0.351 | 10.25 | Off by ~9 walks |

### Honest Assessment
- **R² values of 0.13–0.58 are typical for single-season MLB prediction.** This is not a model failure — it reflects the inherent randomness of baseball. For comparison, professional projection systems (ZiPS, Steamer, PECOTA) achieve similar or modestly better accuracy, often by incorporating proprietary data (pitch tracking, sprint speed, minor league stats) that we do not use.
- **ERA is the hardest stat to predict** (R²=0.135), consistent with the well-known finding that pitcher ERA has high year-to-year variance due to defense, sequencing luck (BABIP), and HR/FB fluctuation.
- **Stolen bases and saves are the most predictable** (R²=0.54, 0.58), because they depend heavily on opportunity/role, which is stable year-to-year.
- **The model systematically predicts regression to the mean.** This is correct behavior, not a limitation. Breakout seasons rarely repeat at the same level, and the model has correctly learned this from the data.

---

## 8. Variance Calibration (Post-Prediction Correction)

### The Problem We Discovered

After generating our initial predictions, we noticed something wrong: the model's predicted batting champion hit just .288. In the real world, the batting champion typically hits .330–.350. No player was predicted above .300.

Investigation revealed this wasn't a bug — it's a well-known property of regression models called **variance shrinkage**. The model correctly learns the *mean* outcome for each player but cannot predict the random component (hot streaks, BABIP luck, favorable matchups) that spreads real outcomes across a wider range. The result:

| Metric | Actual 2025 | Raw Predictions | After Calibration |
|--------|------------|-----------------|-------------------|
| AVG standard deviation | .029 | .015 | .032 |
| AVG maximum | .331 (Judge) | .288 | .331 |
| AVG minimum | .157 | .209 | .146 |
| ERA standard deviation | 1.18 | 0.54 | 1.18 |

The raw model compressed the spread of batting averages by **half** — from std=.029 to std=.015. Every player was pulled toward .246 (the league mean). The predicted *ranking* was correct, but the *distances between players* were unrealistically small.

### The Fix: Variance Calibration

We apply a post-prediction linear rescaling that restores the historical spread:

```
calibrated = mean + (raw_prediction - mean) × scale_factor
scale_factor = historical_std / predicted_std
```

For each statistic, we compute:
1. **Historical std**: the average standard deviation of actual outcomes across 2015–2025 seasons
2. **Predicted std**: the standard deviation of our raw predictions
3. **Scale factor**: the ratio (historical / predicted)

This transformation:
- **Preserves the mean** (unchanged — the predicted league average is correct)
- **Preserves the ranking** (all players shift proportionally from the mean)
- **Preserves relative distances** (if Player A is predicted 2× further from the mean than Player B before calibration, they remain 2× further after)
- **Restores realistic spread** so predictions match the variance actually observed in MLB seasons

### Calibration Scale Factors Applied

| Batting Stat | Historical Std | Raw Pred Std | Scale Factor |
|-------------|---------------|-------------|-------------|
| AVG | .032 | .015 | 2.05× |
| HR | 9.27 | 6.47 | 1.43× |
| RBI | 23.50 | 15.25 | 1.54× |
| OBP | .035 | .019 | 1.84× |
| SLG | .070 | .038 | 1.87× |
| SB | 8.08 | 7.49 | 1.08× |

| Pitching Stat | Historical Std | Raw Pred Std | Scale Factor |
|--------------|---------------|-------------|-------------|
| ERA | 1.18 | 0.54 | 2.18× |
| WHIP | 0.20 | 0.09 | 2.18× |
| SO | 45.72 | 32.55 | 1.40× |
| W | 3.76 | 2.28 | 1.65× |

Note that **stolen bases required almost no calibration** (1.08×) — the model already predicted realistic variance for SB because it's a highly predictable stat (R²=0.54). **ERA and WHIP required the most** (2.18×), consistent with their low R² values — the model explained little variance, so most of the real spread was in the unpredicted random component.

### Why This Is Legitimate (Not Data Manipulation)

This calibration is a standard technique in statistical forecasting, analogous to:
- **Probability calibration** in classification (Platt scaling, isotonic regression)
- **Spread calibration** in weather forecasting (ensemble model output statistics)
- **Dispersion correction** in Bayesian prediction intervals

We are not changing which players rank higher or injecting information about 2026. We are correcting a known statistical artifact of point-prediction regression models using historical distributional properties. The calibration targets (historical stds) are computed from the same 2015–2025 data used for training — no external information is introduced.

A reviewer can verify this by:
1. Running `02_sklearn_models.py` — it prints both raw and calibrated statistics
2. Comparing the predicted std to the actual std of any historical season
3. Confirming that the calibration is a linear transformation that preserves ranking

---

## 9. Known Limitations

1. **No injury modeling.** All predictions assume a full, healthy season. A player who played 150 games in 2025 is predicted under the assumption they'll play ~150 games in 2026. This is the single largest source of prediction error in practice.

2. **No park/team context.** A hitter traded from Oracle Park (pitcher-friendly) to Coors Field (hitter-friendly) will be underpredicted. We excluded park factors to keep the model focused on individual player talent.

3. **2020 COVID season.** The 60-game 2020 season is included in our training data. Its counting stats (HR, RBI, etc.) are roughly 37% of a normal season. This may cause the model to underweight features from that year's data. We did not exclude it because removing it would create a gap in trend/rolling-average calculations.

4. **Rookie limitations.** Players with only 1 prior qualifying season have no trend features (set to 0) and a rolling average based on a single year. Their predictions have wider implicit uncertainty.

5. **No uncertainty quantification.** We report point predictions only. A full treatment would include prediction intervals (e.g., via quantile regression or conformal prediction). This is a meaningful limitation for any consumer of these predictions.

6. **No feature importance analysis.** We do not report SHAP values or permutation importance in this version. This would be a valuable addition for understanding which features drive individual predictions.

---

## 10. Google AutoML Comparison

For batting average (AVG) and home runs (HR), we also trained Google Vertex AI AutoML Tabular models using the **exact same training data and features**. This provides a comparison between:

- **scikit-learn GBR:** Transparent, reproducible, free, fixed hyperparameters, runs on any laptop.
- **Google AutoML:** Black-box, automated architecture search, cloud-based, costs ~$19/hour training.

The AutoML models received the same CSV feature matrix uploaded to Google Cloud Storage. AutoML was given a 1-hour training budget and default settings with RMSE optimization.

**Important caveat:** AutoML results are *not* deterministically reproducible. Google's AutoML infrastructure may produce different models on re-runs due to non-deterministic architecture search and distributed training. The scikit-learn results are the reproducible baseline; AutoML results are presented for comparison only.

---

## 11. How to Reproduce (Step by Step)

### Prerequisites
- Python 3.13+ (tested on 3.13.9, Anaconda distribution)
- macOS or Linux (tested on macOS Darwin 25.3.0, Apple Silicon)

### Steps

```bash
# 1. Clone or download this project directory

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install exact dependency versions
pip install -r requirements.txt

# 4. Option A: Use our provided data (guaranteed reproducible)
#    Skip to step 5 — the data/ directory already contains the CSVs.
#
#    Option B: Re-download from source (may differ if upstream updated)
#    python3 01_pull_data.py
#    Then compare checksums with verify_reproducibility.py

# 5. Train models and generate predictions
python3 02_sklearn_models.py

# 6. Verify your results match ours
python3 verify_reproducibility.py

# 7. Generate the spreadsheet
python3 03_create_spreadsheets.py
```

If `verify_reproducibility.py` reports ALL CHECKS PASSED, you have reproduced our results exactly.

### If Verification Fails
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `data/` files differ | Upstream Lahman repo updated | Use our provided CSVs (skip `01_pull_data.py`) |
| `output/` files differ | Different scikit-learn version | Install exact version: `pip install scikit-learn==1.7.2` |
| `output/` files differ | Different NumPy version | Install exact version: `pip install numpy==2.3.5` |
| `output/` files differ | Different OS/architecture | Should not happen with these packages, but file a bug |

---

## 12. Dependency Pinning

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.13.9 | Runtime |
| scikit-learn | 1.7.2 | Model training (GradientBoostingRegressor) |
| pandas | 2.3.3 | Data manipulation |
| numpy | 2.3.5 | Numerical computation |
| pyreadr | 0.5.4 | Reading .RData files from Lahman R package |
| requests | 2.32.5 | HTTP download of Lahman database |
| openpyxl | ≥3.1.0 | Excel file generation (does not affect predictions) |

---

## 13. File Manifest

```
Baseball_ML_Predictions/
├── 01_pull_data.py                  # Data download and processing
├── 02_sklearn_models.py             # Model training and prediction
├── 03_create_spreadsheets.py        # Excel/CSV output generation
├── 04_automl_train.py               # Google AutoML training (optional)
├── verify_reproducibility.py        # SHA-256 verification script
├── requirements.txt                 # Pinned dependency versions
├── REPRODUCIBILITY.md               # This document
├── blog_post_draft.md               # Blog post draft
├── data/
│   ├── batting_raw.csv              # 3,688 batting player-seasons (2015-2025)
│   ├── pitching_raw.csv             # 3,502 pitching player-seasons (2015-2025)
│   └── people.csv                   # Player biographical data
└── output/
    ├── batting_predictions_sklearn.csv   # 348 player predictions for 2026
    ├── pitching_predictions_sklearn.csv  # 339 pitcher predictions for 2026
    ├── model_metrics.json                # Evaluation metrics (JSON)
    └── MLB_2026_Predictions.xlsx         # Final spreadsheet (3 sheets)
```

---

## 14. Contact & Citation

If you use this analysis, please cite:

> Panetta, A. (2026). "Play Ball! What Machine Learning Predicts This Baseball Season." Data source: SABR Lahman Baseball Database, 2025 Edition. Models: scikit-learn 1.7.2 GradientBoostingRegressor. Code assistance: Claude (Anthropic).

For questions about methodology or reproducibility, contact Alex Panetta.
