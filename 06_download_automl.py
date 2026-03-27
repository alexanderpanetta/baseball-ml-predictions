"""
Step 6: Download all AutoML batch prediction results, merge with sklearn predictions.
AutoML splits output across multiple CSV shards — this script reads all of them.
"""
import pandas as pd
import numpy as np
import os
from google.cloud import storage

PROJECT_ID = "shiller-cape-analysis"
BUCKET = "shiller-cape-data-panetta"
OUTPUT_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/output"
DATA_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/data"

client = storage.Client(project=PROJECT_ID)
bucket_obj = client.bucket(BUCKET)

BATTING_TARGETS = ["AVG", "R", "H", "HR", "RBI", "2B", "SB", "BB", "OBP", "SLG"]

def download_batch_predictions(prefix, stat_name):
    """Download and concatenate all prediction result shards."""
    blobs = list(bucket_obj.list_blobs(prefix=prefix))
    # Get only the results files, not errors
    result_blobs = [b for b in blobs if "prediction.results" in b.name and b.name.endswith(".csv")]
    print(f"  Found {len(result_blobs)} result shards for {stat_name}")

    dfs = []
    for blob in result_blobs:
        local_tmp = f"/tmp/automl_{stat_name}_{os.path.basename(blob.name)}"
        blob.download_to_filename(local_tmp)
        df = pd.read_csv(local_tmp)
        if len(df) > 0:
            dfs.append(df)

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Combined: {len(combined)} rows, columns: {combined.columns.tolist()[-5:]}")
        return combined
    else:
        print(f"  WARNING: No prediction data found for {stat_name}")
        return None

# Download AVG predictions
print("Downloading AVG predictions...")
avg_df = download_batch_predictions("baseball_ml_2026/predictions_AVG/", "AVG")
if avg_df is not None:
    # The prediction column is typically 'predicted_target_AVG' or similar
    pred_cols = [c for c in avg_df.columns if 'predicted' in c.lower() or c == 'target_AVG']
    print(f"  Prediction columns: {pred_cols}")
    print(f"  All columns: {avg_df.columns.tolist()}")
    print(avg_df.head(3))

# Download HR predictions
print("\nDownloading HR predictions...")
hr_df = download_batch_predictions("baseball_ml_2026/predictions_HR/", "HR")
if hr_df is not None:
    pred_cols = [c for c in hr_df.columns if 'predicted' in c.lower() or c == 'target_HR']
    print(f"  Prediction columns: {pred_cols}")
    print(f"  All columns: {hr_df.columns.tolist()}")
    print(hr_df.head(3))

# ============================================================
# Match predictions back to players
# ============================================================
# The batch prediction input didn't have player names (just features).
# We need to match by row order — the prediction output preserves input order.

# Rebuild the prediction input to get player names in order
batting = pd.read_csv(os.path.join(DATA_DIR, "batting_raw.csv"))
batting = batting.sort_values(["playerID", "yearID"])
batting_2025 = batting[batting["yearID"] == 2025].copy()

predict_rows = []
for _, player in batting_2025.iterrows():
    pid = player["playerID"]
    history = batting[batting["playerID"] == pid].sort_values("yearID")
    row = {"playerID": pid, "fullName": player["fullName"]}
    row["age"] = player["age"] + 1
    row["age_sq"] = (player["age"] + 1) ** 2
    row["n_seasons"] = len(history)
    for stat in BATTING_TARGETS + ["PA", "G", "SO"]:
        row[f"prev_{stat}"] = player[stat]
    recent = history.tail(3)
    weights = np.array([1, 2, 3])[-len(recent):]
    weights = weights / weights.sum()
    for stat in BATTING_TARGETS:
        row[f"wavg3_{stat}"] = np.average(recent[stat].values, weights=weights)
    for stat in BATTING_TARGETS:
        row[f"career_{stat}"] = history[stat].mean()
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

player_df = pd.DataFrame(predict_rows)

# Match by joining on feature values (age, prev_AVG, etc.)
# The safest way is to join on a set of features that uniquely identify each player
print(f"\nPlayer lookup table: {len(player_df)} players")

# Try matching by prev_AVG + prev_HR + age (should be unique per player)
if avg_df is not None and len(avg_df) > 0:
    # Find the prediction column
    automl_pred_col_avg = [c for c in avg_df.columns if 'predicted' in c.lower()]
    if not automl_pred_col_avg:
        # Sometimes it's just appended as the last column
        automl_pred_col_avg = [avg_df.columns[-1]]
    print(f"\n  Using prediction column for AVG: {automl_pred_col_avg[0]}")

    # Match on feature values
    merge_keys = ["age", "prev_AVG", "prev_HR", "prev_PA"]
    merged_avg = player_df.merge(
        avg_df[merge_keys + automl_pred_col_avg].drop_duplicates(subset=merge_keys),
        on=merge_keys, how="left"
    )
    merged_avg = merged_avg.rename(columns={automl_pred_col_avg[0]: "automl_AVG"})
    matched = merged_avg["automl_AVG"].notna().sum()
    print(f"  Matched {matched}/{len(player_df)} players for AVG")

if hr_df is not None and len(hr_df) > 0:
    automl_pred_col_hr = [c for c in hr_df.columns if 'predicted' in c.lower()]
    if not automl_pred_col_hr:
        automl_pred_col_hr = [hr_df.columns[-1]]
    print(f"  Using prediction column for HR: {automl_pred_col_hr[0]}")

    merged_hr = player_df.merge(
        hr_df[merge_keys + automl_pred_col_hr].drop_duplicates(subset=merge_keys),
        on=merge_keys, how="left"
    )
    merged_hr = merged_hr.rename(columns={automl_pred_col_hr[0]: "automl_HR"})
    matched = merged_hr["automl_HR"].notna().sum()
    print(f"  Matched {matched}/{len(player_df)} players for HR")

    # Combine
    merged_avg["automl_HR"] = merged_hr["automl_HR"]

# Apply variance calibration to AutoML predictions too
print("\nApplying variance calibration to AutoML predictions...")
for stat in ["AVG", "HR"]:
    col = f"automl_{stat}"
    if col in merged_avg.columns:
        raw = merged_avg[col].dropna()
        pred_mean = raw.mean()
        pred_std = raw.std()

        # Get historical std
        yearly_stds = []
        for year in batting["yearID"].unique():
            year_data = batting[batting["yearID"] == year][stat].dropna()
            if len(year_data) > 10:
                yearly_stds.append(year_data.std())
        historical_std = np.mean(yearly_stds)

        if pred_std > 0:
            scale = historical_std / pred_std
            merged_avg[col] = pred_mean + (merged_avg[col] - pred_mean) * scale
            print(f"  {stat}: pred_std={pred_std:.4f}, hist_std={historical_std:.4f}, scale={scale:.2f}x")

        if stat == "AVG":
            merged_avg[col] = merged_avg[col].round(3).clip(0.100, 0.500)
        else:
            merged_avg[col] = merged_avg[col].round(0).clip(0, None).astype("Int64")

# Save
automl_output = merged_avg[["fullName", "playerID", "automl_AVG", "automl_HR"]].copy()
automl_output.to_csv(os.path.join(OUTPUT_DIR, "automl_predictions.csv"), index=False)

# Show comparison
print("\n" + "=" * 70)
print("scikit-learn vs Google AutoML — Top 15 by HR")
print("=" * 70)
sklearn = pd.read_csv(os.path.join(OUTPUT_DIR, "batting_predictions_sklearn.csv"))
compare = sklearn[["Player", "sklearn_AVG", "sklearn_HR"]].merge(
    automl_output.rename(columns={"fullName": "Player"})[["Player", "automl_AVG", "automl_HR"]],
    on="Player", how="left"
)
print(compare.sort_values("sklearn_HR", ascending=False).head(15).to_string(index=False))

print("\n" + "=" * 70)
print("scikit-learn vs Google AutoML — Top 15 by AVG")
print("=" * 70)
print(compare.sort_values("sklearn_AVG", ascending=False).head(15).to_string(index=False))

print(f"\nAutoML predictions saved to {OUTPUT_DIR}/automl_predictions.csv")
