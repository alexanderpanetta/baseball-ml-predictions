"""
Step 2: Build scikit-learn models to predict 2026 MLB season stats.

Approach:
- For each player who played in 2025, predict their 2026 stats
- Features: rolling averages (3-year weighted), age, age-squared (aging curve),
  year-over-year trends, career totals
- Model: Gradient Boosting Regressor (one per target stat)
- Train on 2015-2024 data (predicting next-year stats), test on 2025 actuals
- Then retrain on full 2015-2025 data and predict 2026

Reproducibility: All random seeds are fixed. Run after 01_pull_data.py.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import os
import json
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/data"
OUTPUT_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# ============================================================
# BATTING PREDICTIONS
# ============================================================
print("=" * 60)
print("BATTING MODEL")
print("=" * 60)

batting = pd.read_csv(os.path.join(DATA_DIR, "batting_raw.csv"))
batting = batting.sort_values(["playerID", "yearID"])

BATTING_TARGETS = ["AVG", "R", "H", "HR", "RBI", "2B", "SB", "BB", "OBP", "SLG"]

def build_batting_features(df):
    """Build features for batting prediction. Each row = one player-season.
    Features are based on PRIOR seasons only (no data leakage)."""
    players = df.groupby("playerID")
    rows = []

    for pid, group in players:
        group = group.sort_values("yearID")
        for i in range(1, len(group)):
            current = group.iloc[i]
            history = group.iloc[:i]  # all prior seasons

            row = {"playerID": pid, "yearID": current["yearID"],
                   "fullName": current["fullName"]}

            # Target stats (what we're predicting)
            for stat in BATTING_TARGETS:
                row[f"target_{stat}"] = current[stat]

            # Feature: age and age^2 (aging curve)
            row["age"] = current["age"]
            row["age_sq"] = current["age"] ** 2

            # Feature: number of prior qualified seasons
            row["n_seasons"] = len(history)

            # Feature: most recent season stats
            prev = group.iloc[i - 1]
            for stat in BATTING_TARGETS + ["PA", "G", "SO"]:
                row[f"prev_{stat}"] = prev[stat]

            # Feature: weighted 3-year average (more recent = higher weight)
            recent = history.tail(3)
            weights = np.array([1, 2, 3])[-len(recent):]
            weights = weights / weights.sum()
            for stat in BATTING_TARGETS:
                vals = recent[stat].values
                row[f"wavg3_{stat}"] = np.average(vals, weights=weights)

            # Feature: career average
            for stat in BATTING_TARGETS:
                row[f"career_{stat}"] = history[stat].mean()

            # Feature: year-over-year change (trend)
            if len(history) >= 2:
                prev2 = group.iloc[i - 2]
                for stat in BATTING_TARGETS:
                    row[f"trend_{stat}"] = prev[stat] - prev2[stat]
            else:
                for stat in BATTING_TARGETS:
                    row[f"trend_{stat}"] = 0.0

            # Feature: plate appearances (volume proxy)
            row["prev_PA"] = prev["PA"]

            rows.append(row)

    return pd.DataFrame(rows)

print("Building batting features...")
batting_features = build_batting_features(batting)
print(f"  {len(batting_features)} player-season rows with features")

# Split: train on predicting 2016-2024, test on predicting 2025
feature_cols = [c for c in batting_features.columns
                if c not in ["playerID", "yearID", "fullName"]
                and not c.startswith("target_")]

train_bat = batting_features[batting_features["yearID"].between(2016, 2024)]
test_bat = batting_features[batting_features["yearID"] == 2025]
print(f"  Train: {len(train_bat)} rows (predicting 2016-2024)")
print(f"  Test: {len(test_bat)} rows (predicting 2025)")

# Train one model per target stat
batting_models = {}
batting_results = {}
print("\nTraining batting models...")
for stat in BATTING_TARGETS:
    target = f"target_{stat}"
    X_train = train_bat[feature_cols].fillna(0)
    y_train = train_bat[target]
    X_test = test_bat[feature_cols].fillna(0)
    y_test = test_bat[target]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_SEED, subsample=0.8
    )
    model.fit(X_train, y_train)

    # Evaluate on 2025 holdout
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

    batting_models[stat] = model
    batting_results[stat] = {"MAE": round(mae, 4), "R2": round(r2, 4),
                              "CV_MAE": round(-cv_scores.mean(), 4)}
    print(f"  {stat:>4s}: MAE={mae:.4f}, R²={r2:.4f}, CV_MAE={-cv_scores.mean():.4f}")

# ---- Generate 2026 predictions ----
# Retrain on ALL data (2016-2025) for final predictions
print("\nRetraining on full data for 2026 predictions...")
full_train_bat = batting_features[batting_features["yearID"].between(2016, 2025)]

batting_models_final = {}
for stat in BATTING_TARGETS:
    target = f"target_{stat}"
    X = full_train_bat[feature_cols].fillna(0)
    y = full_train_bat[target]
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_SEED, subsample=0.8
    )
    model.fit(X, y)
    batting_models_final[stat] = model

# Build features for 2026 prediction (using 2025 as most recent season)
batting_2025 = batting[batting["yearID"] == 2025].copy()
predict_rows = []
for _, player in batting_2025.iterrows():
    pid = player["playerID"]
    history = batting[batting["playerID"] == pid].sort_values("yearID")
    if len(history) < 2:
        # Need at least 2 seasons for trend features; use career avg for trend=0
        pass  # still include but with zero trends

    row = {"playerID": pid, "fullName": player["fullName"]}

    # Age in 2026
    row["age"] = player["age"] + 1
    row["age_sq"] = (player["age"] + 1) ** 2
    row["n_seasons"] = len(history)

    # Most recent (2025) stats
    for stat in BATTING_TARGETS + ["PA", "G", "SO"]:
        row[f"prev_{stat}"] = player[stat]

    # Weighted 3-year average
    recent = history.tail(3)
    weights = np.array([1, 2, 3])[-len(recent):]
    weights = weights / weights.sum()
    for stat in BATTING_TARGETS:
        row[f"wavg3_{stat}"] = np.average(recent[stat].values, weights=weights)

    # Career average
    for stat in BATTING_TARGETS:
        row[f"career_{stat}"] = history[stat].mean()

    # Trend
    if len(history) >= 2:
        prev = history.iloc[-1]
        prev2 = history.iloc[-2]
        for stat in BATTING_TARGETS:
            row[f"trend_{stat}"] = prev[stat] - prev2[stat]
    else:
        for stat in BATTING_TARGETS:
            row[f"trend_{stat}"] = 0.0

    row["prev_PA"] = player["PA"]
    predict_rows.append(row)

predict_bat_df = pd.DataFrame(predict_rows)

# Generate raw predictions
for stat in BATTING_TARGETS:
    X_pred = predict_bat_df[feature_cols].fillna(0)
    predict_bat_df[f"pred_{stat}"] = batting_models_final[stat].predict(X_pred)

# ============================================================
# VARIANCE CALIBRATION
# ============================================================
# Regression models systematically compress the variance of predictions.
# The predicted mean is correct, but the spread is too narrow — the model
# cannot predict the random component (hot streaks, BABIP luck, etc.) that
# creates the full range of real-world outcomes.
#
# Fix: rescale each stat's predictions so the standard deviation matches
# the historical standard deviation of actual outcomes. This preserves:
#   - The mean (unchanged)
#   - The ranking (all players shift proportionally from the mean)
#   - The relative distances between players
# It restores the realistic spread so the predicted batting champion
# hits ~.320 instead of ~.288.
#
# We use the average std across 2015-2025 seasons as the calibration target.
print("\nApplying variance calibration...")
batting_calibration = {}
for stat in BATTING_TARGETS:
    # Compute historical std: average across all years
    yearly_stds = []
    for year in batting["yearID"].unique():
        year_data = batting[batting["yearID"] == year][stat].dropna()
        if len(year_data) > 10:
            yearly_stds.append(year_data.std())
    historical_std = np.mean(yearly_stds)

    # Compute predicted std
    raw_preds = predict_bat_df[f"pred_{stat}"]
    pred_mean = raw_preds.mean()
    pred_std = raw_preds.std()

    # Rescale: shift each prediction away from the mean by the ratio
    if pred_std > 0:
        scale_factor = historical_std / pred_std
        predict_bat_df[f"pred_{stat}"] = pred_mean + (raw_preds - pred_mean) * scale_factor
    else:
        scale_factor = 1.0

    batting_calibration[stat] = {
        "historical_std": round(historical_std, 4),
        "raw_pred_std": round(pred_std, 4),
        "scale_factor": round(scale_factor, 4)
    }
    print(f"  {stat:>4s}: historical_std={historical_std:.4f}, pred_std={pred_std:.4f}, "
          f"scale={scale_factor:.2f}x")

# Clean up predictions
pred_cols = ["fullName", "playerID"] + [f"pred_{s}" for s in BATTING_TARGETS]
batting_preds = predict_bat_df[pred_cols].copy()

# Round appropriately
for stat in ["AVG", "OBP", "SLG"]:
    batting_preds[f"pred_{stat}"] = batting_preds[f"pred_{stat}"].round(3)
    batting_preds[f"pred_{stat}"] = batting_preds[f"pred_{stat}"].clip(0.100, 0.500 if stat == "AVG" else 0.700)
for stat in ["R", "H", "HR", "RBI", "2B", "SB", "BB"]:
    batting_preds[f"pred_{stat}"] = batting_preds[f"pred_{stat}"].round(0).astype(int)
    batting_preds[f"pred_{stat}"] = batting_preds[f"pred_{stat}"].clip(0, None)

# Rename columns for display
batting_preds.columns = ["Player", "playerID"] + [f"sklearn_{s}" for s in BATTING_TARGETS]
batting_preds = batting_preds.sort_values("sklearn_HR", ascending=False)

print(f"\n2026 Batting Predictions (top 20 by HR):")
print(batting_preds.head(20).to_string(index=False))

batting_preds.to_csv(os.path.join(OUTPUT_DIR, "batting_predictions_sklearn.csv"), index=False)

# ============================================================
# PITCHING PREDICTIONS
# ============================================================
print("\n" + "=" * 60)
print("PITCHING MODEL")
print("=" * 60)

pitching = pd.read_csv(os.path.join(DATA_DIR, "pitching_raw.csv"))
pitching = pitching.sort_values(["playerID", "yearID"])

PITCHING_TARGETS = ["ERA", "W", "L", "SO", "WHIP", "IP", "SV", "HR", "BB"]

def build_pitching_features(df):
    """Build features for pitching prediction."""
    players = df.groupby("playerID")
    rows = []

    for pid, group in players:
        group = group.sort_values("yearID")
        for i in range(1, len(group)):
            current = group.iloc[i]
            history = group.iloc[:i]

            row = {"playerID": pid, "yearID": current["yearID"],
                   "fullName": current["fullName"]}

            for stat in PITCHING_TARGETS:
                row[f"target_{stat}"] = current[stat]

            row["age"] = current["age"]
            row["age_sq"] = current["age"] ** 2
            row["n_seasons"] = len(history)

            prev = group.iloc[i - 1]
            for stat in PITCHING_TARGETS + ["G", "GS", "BFP", "K9", "BB9", "HR9"]:
                row[f"prev_{stat}"] = prev[stat]

            recent = history.tail(3)
            weights = np.array([1, 2, 3])[-len(recent):]
            weights = weights / weights.sum()
            for stat in PITCHING_TARGETS:
                vals = recent[stat].values
                row[f"wavg3_{stat}"] = np.average(vals, weights=weights)

            for stat in PITCHING_TARGETS:
                row[f"career_{stat}"] = history[stat].mean()

            if len(history) >= 2:
                prev2 = group.iloc[i - 2]
                for stat in PITCHING_TARGETS:
                    row[f"trend_{stat}"] = prev[stat] - prev2[stat]
            else:
                for stat in PITCHING_TARGETS:
                    row[f"trend_{stat}"] = 0.0

            row["prev_GS_ratio"] = prev["GS"] / max(prev["G"], 1)  # starter vs reliever
            rows.append(row)

    return pd.DataFrame(rows)

print("Building pitching features...")
pitching_features = build_pitching_features(pitching)
print(f"  {len(pitching_features)} player-season rows with features")

p_feature_cols = [c for c in pitching_features.columns
                  if c not in ["playerID", "yearID", "fullName"]
                  and not c.startswith("target_")]

train_pitch = pitching_features[pitching_features["yearID"].between(2016, 2024)]
test_pitch = pitching_features[pitching_features["yearID"] == 2025]
print(f"  Train: {len(train_pitch)} rows, Test: {len(test_pitch)} rows")

pitching_models = {}
pitching_results = {}
print("\nTraining pitching models...")
for stat in PITCHING_TARGETS:
    target = f"target_{stat}"
    X_train = train_pitch[p_feature_cols].fillna(0)
    y_train = train_pitch[target]
    X_test = test_pitch[p_feature_cols].fillna(0)
    y_test = test_pitch[target]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_SEED, subsample=0.8
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")

    pitching_models[stat] = model
    pitching_results[stat] = {"MAE": round(mae, 4), "R2": round(r2, 4),
                               "CV_MAE": round(-cv_scores.mean(), 4)}
    print(f"  {stat:>4s}: MAE={mae:.4f}, R²={r2:.4f}, CV_MAE={-cv_scores.mean():.4f}")

# Retrain on full data for 2026 predictions
print("\nRetraining on full data for 2026 predictions...")
full_train_pitch = pitching_features[pitching_features["yearID"].between(2016, 2025)]

pitching_models_final = {}
for stat in PITCHING_TARGETS:
    target = f"target_{stat}"
    X = full_train_pitch[p_feature_cols].fillna(0)
    y = full_train_pitch[target]
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=RANDOM_SEED, subsample=0.8
    )
    model.fit(X, y)
    pitching_models_final[stat] = model

# Build 2026 prediction features
pitching_2025 = pitching[pitching["yearID"] == 2025].copy()
predict_rows_p = []
for _, player in pitching_2025.iterrows():
    pid = player["playerID"]
    history = pitching[pitching["playerID"] == pid].sort_values("yearID")

    row = {"playerID": pid, "fullName": player["fullName"]}
    row["age"] = player["age"] + 1
    row["age_sq"] = (player["age"] + 1) ** 2
    row["n_seasons"] = len(history)

    for stat in PITCHING_TARGETS + ["G", "GS", "BFP", "K9", "BB9", "HR9"]:
        row[f"prev_{stat}"] = player[stat]

    recent = history.tail(3)
    weights = np.array([1, 2, 3])[-len(recent):]
    weights = weights / weights.sum()
    for stat in PITCHING_TARGETS:
        row[f"wavg3_{stat}"] = np.average(recent[stat].values, weights=weights)

    for stat in PITCHING_TARGETS:
        row[f"career_{stat}"] = history[stat].mean()

    if len(history) >= 2:
        prev = history.iloc[-1]
        prev2 = history.iloc[-2]
        for stat in PITCHING_TARGETS:
            row[f"trend_{stat}"] = prev[stat] - prev2[stat]
    else:
        for stat in PITCHING_TARGETS:
            row[f"trend_{stat}"] = 0.0

    row["prev_GS_ratio"] = player["GS"] / max(player["G"], 1)
    predict_rows_p.append(row)

predict_pitch_df = pd.DataFrame(predict_rows_p)

for stat in PITCHING_TARGETS:
    X_pred = predict_pitch_df[p_feature_cols].fillna(0)
    predict_pitch_df[f"pred_{stat}"] = pitching_models_final[stat].predict(X_pred)

# ---- Variance calibration for pitching ----
print("\nApplying variance calibration (pitching)...")
pitching_calibration = {}
for stat in PITCHING_TARGETS:
    yearly_stds = []
    for year in pitching["yearID"].unique():
        year_data = pitching[pitching["yearID"] == year][stat].dropna()
        if len(year_data) > 10:
            yearly_stds.append(year_data.std())
    historical_std = np.mean(yearly_stds)

    raw_preds = predict_pitch_df[f"pred_{stat}"]
    pred_mean = raw_preds.mean()
    pred_std = raw_preds.std()

    if pred_std > 0:
        scale_factor = historical_std / pred_std
        predict_pitch_df[f"pred_{stat}"] = pred_mean + (raw_preds - pred_mean) * scale_factor
    else:
        scale_factor = 1.0

    pitching_calibration[stat] = {
        "historical_std": round(historical_std, 4),
        "raw_pred_std": round(pred_std, 4),
        "scale_factor": round(scale_factor, 4)
    }
    print(f"  {stat:>4s}: historical_std={historical_std:.4f}, pred_std={pred_std:.4f}, "
          f"scale={scale_factor:.2f}x")

pred_cols_p = ["fullName", "playerID"] + [f"pred_{s}" for s in PITCHING_TARGETS]
pitching_preds = predict_pitch_df[pred_cols_p].copy()

# Round appropriately
for stat in ["ERA", "WHIP"]:
    pitching_preds[f"pred_{stat}"] = pitching_preds[f"pred_{stat}"].round(2)
    pitching_preds[f"pred_{stat}"] = pitching_preds[f"pred_{stat}"].clip(0.50, None)
for stat in ["W", "L", "SO", "IP", "SV", "HR", "BB"]:
    pitching_preds[f"pred_{stat}"] = pitching_preds[f"pred_{stat}"].round(0).astype(int)
    pitching_preds[f"pred_{stat}"] = pitching_preds[f"pred_{stat}"].clip(0, None)
pitching_preds[f"pred_IP"] = pitching_preds[f"pred_IP"].round(1)

pitching_preds.columns = ["Player", "playerID"] + [f"sklearn_{s}" for s in PITCHING_TARGETS]
pitching_preds = pitching_preds.sort_values("sklearn_ERA")

print(f"\n2026 Pitching Predictions (top 20 by ERA):")
print(pitching_preds.head(20).to_string(index=False))

pitching_preds.to_csv(os.path.join(OUTPUT_DIR, "pitching_predictions_sklearn.csv"), index=False)

# ============================================================
# SAVE MODEL EVALUATION METRICS
# ============================================================
metrics = {
    "batting": batting_results,
    "pitching": pitching_results,
    "model_params": {
        "algorithm": "GradientBoostingRegressor",
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
        "min_samples_leaf": 10, "subsample": 0.8, "random_state": RANDOM_SEED
    },
    "training_data": {
        "source": "SABR Lahman Database 2025 edition",
        "batting_seasons": "2015-2025 (200+ PA per season)",
        "pitching_seasons": "2015-2025 (50+ IP per season)",
        "feature_engineering": [
            "Previous season stats (all target stats + PA, G, SO/BFP)",
            "Weighted 3-year rolling average (weights: 1/6, 2/6, 3/6)",
            "Career average of each stat",
            "Year-over-year trend (difference between last two seasons)",
            "Age and age-squared (quadratic aging curve)",
            "Number of qualified prior seasons",
            "GS/G ratio for pitchers (starter vs reliever indicator)"
        ],
        "train_test_split": "Train on 2016-2024 predictions, test on 2025 predictions",
        "final_model": "Retrained on 2016-2025 for 2026 predictions"
    },
    "variance_calibration": {
        "description": "Raw regression predictions have compressed variance (too narrow spread). "
                       "We rescale each stat so the predicted std matches the historical avg std "
                       "across 2015-2025 seasons. Mean and ranking are preserved.",
        "batting": batting_calibration,
        "pitching": pitching_calibration
    }
}

with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("Done!")
