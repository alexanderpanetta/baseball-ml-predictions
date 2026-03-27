"""
Step 4: Train Google AutoML Tabular models for AVG and HR predictions.
Uses Vertex AI AutoML Tables via the Python SDK.

Prerequisites:
- gcloud auth login (already done)
- Vertex AI API enabled (already done)
- pip install google-cloud-aiplatform
"""
import pandas as pd
import os
import time
from google.cloud import aiplatform, storage

PROJECT_ID = "shiller-cape-analysis"
REGION = "us-central1"
BUCKET = "shiller-cape-data-panetta"
DATA_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/data"
OUTPUT_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/output"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")

# ============================================================
# Prepare training data (same features as scikit-learn)
# ============================================================
print("Preparing training data for AutoML...")

batting = pd.read_csv(os.path.join(DATA_DIR, "batting_raw.csv"))
batting = batting.sort_values(["playerID", "yearID"])

import numpy as np

BATTING_TARGETS = ["AVG", "R", "H", "HR", "RBI", "2B", "SB", "BB", "OBP", "SLG"]

def build_batting_features(df):
    players = df.groupby("playerID")
    rows = []
    for pid, group in players:
        group = group.sort_values("yearID")
        for i in range(1, len(group)):
            current = group.iloc[i]
            history = group.iloc[:i]
            row = {"playerID": pid, "yearID": int(current["yearID"]),
                   "fullName": current["fullName"]}
            for stat in BATTING_TARGETS:
                row[f"target_{stat}"] = current[stat]
            row["age"] = current["age"]
            row["age_sq"] = current["age"] ** 2
            row["n_seasons"] = len(history)
            prev = group.iloc[i - 1]
            for stat in BATTING_TARGETS + ["PA", "G", "SO"]:
                row[f"prev_{stat}"] = prev[stat]
            recent = history.tail(3)
            weights = np.array([1, 2, 3])[-len(recent):]
            weights = weights / weights.sum()
            for stat in BATTING_TARGETS:
                row[f"wavg3_{stat}"] = np.average(recent[stat].values, weights=weights)
            for stat in BATTING_TARGETS:
                row[f"career_{stat}"] = history[stat].mean()
            if len(history) >= 2:
                prev2 = group.iloc[i - 2]
                for stat in BATTING_TARGETS:
                    row[f"trend_{stat}"] = prev[stat] - prev2[stat]
            else:
                for stat in BATTING_TARGETS:
                    row[f"trend_{stat}"] = 0.0
            row["prev_PA"] = prev["PA"]
            rows.append(row)
    return pd.DataFrame(rows)

features_df = build_batting_features(batting)

# AutoML needs: features + target column, no ID columns
# We'll train on 2016-2025 data (same as sklearn final model)
train_df = features_df[features_df["yearID"].between(2016, 2025)].copy()

feature_cols = [c for c in train_df.columns
                if c not in ["playerID", "yearID", "fullName"]
                and not c.startswith("target_")]

# ============================================================
# Upload training CSVs to GCS
# ============================================================
gcs_prefix = "baseball_ml_2026"

for target_stat in ["AVG", "HR"]:
    # Prepare CSV with features + target
    cols = feature_cols + [f"target_{target_stat}"]
    upload_df = train_df[cols].dropna().copy()

    local_path = os.path.join(OUTPUT_DIR, f"automl_train_{target_stat}.csv")
    upload_df.to_csv(local_path, index=False)

    gcs_path = f"{gcs_prefix}/automl_train_{target_stat}.csv"

    # Upload to GCS
    client = storage.Client(project=PROJECT_ID)
    bucket_obj = client.bucket(BUCKET)
    blob = bucket_obj.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} -> gs://{BUCKET}/{gcs_path} ({len(upload_df)} rows)")

# ============================================================
# Train AutoML models
# ============================================================
jobs = {}
for target_stat in ["AVG", "HR"]:
    gcs_uri = f"gs://{BUCKET}/{gcs_prefix}/automl_train_{target_stat}.csv"
    target_col = f"target_{target_stat}"

    print(f"\nCreating AutoML dataset for {target_stat}...")
    dataset = aiplatform.TabularDataset.create(
        display_name=f"baseball_{target_stat}_2026",
        gcs_source=gcs_uri,
    )
    print(f"  Dataset created: {dataset.resource_name}")

    print(f"Training AutoML model for {target_stat}...")
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"baseball_{target_stat}_predict_2026",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse",
    )

    model = job.run(
        dataset=dataset,
        target_column=target_col,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=1000,  # 1 hour training budget
        model_display_name=f"baseball_{target_stat}_2026_model",
    )

    jobs[target_stat] = {"model": model, "job": job}
    print(f"  Model trained: {model.resource_name}")

# ============================================================
# Generate predictions with AutoML models
# ============================================================
print("\n\nGenerating 2026 predictions with AutoML models...")

# Build 2026 prediction features (same as sklearn)
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

predict_df = pd.DataFrame(predict_rows)

for target_stat in ["AVG", "HR"]:
    model = jobs[target_stat]["model"]

    # Prepare prediction input
    pred_input = predict_df[feature_cols].fillna(0).copy()

    # Batch predict
    pred_input_path = os.path.join(OUTPUT_DIR, f"automl_predict_input_{target_stat}.csv")
    pred_input.to_csv(pred_input_path, index=False)

    # Upload prediction input to GCS
    gcs_pred_path = f"{gcs_prefix}/predict_input_{target_stat}.csv"
    blob = bucket_obj.blob(gcs_pred_path)
    blob.upload_from_filename(pred_input_path)

    gcs_pred_uri = f"gs://{BUCKET}/{gcs_pred_path}"
    gcs_output_uri = f"gs://{BUCKET}/{gcs_prefix}/predictions_{target_stat}/"

    print(f"\nRunning batch prediction for {target_stat}...")
    batch_job = model.batch_predict(
        job_display_name=f"baseball_{target_stat}_batch_predict",
        gcs_source=gcs_pred_uri,
        gcs_destination_prefix=gcs_output_uri,
        instances_format="csv",
        predictions_format="csv",
    )
    print(f"  Batch prediction complete for {target_stat}")

    # Download predictions
    output_blobs = list(bucket_obj.list_blobs(prefix=f"{gcs_prefix}/predictions_{target_stat}/"))
    csv_blobs = [b for b in output_blobs if b.name.endswith(".csv")]

    if csv_blobs:
        pred_results = pd.read_csv(f"gs://{BUCKET}/{csv_blobs[0].name}", storage_options={"project": PROJECT_ID})
        predict_df[f"automl_{target_stat}"] = pred_results.iloc[:, -1].values
        print(f"  Got {len(pred_results)} predictions for {target_stat}")
    else:
        # Try reading from the output prefix
        print(f"  Checking output location...")
        for blob in output_blobs:
            print(f"    {blob.name}")

# Save AutoML predictions
automl_preds = predict_df[["fullName", "playerID"]].copy()
for stat in ["AVG", "HR"]:
    col = f"automl_{stat}"
    if col in predict_df.columns:
        automl_preds[col] = predict_df[col]
        if stat == "AVG":
            automl_preds[col] = automl_preds[col].round(3).clip(0.100, 0.500)
        else:
            automl_preds[col] = automl_preds[col].round(0).astype(int).clip(0, None)

automl_preds.to_csv(os.path.join(OUTPUT_DIR, "automl_predictions.csv"), index=False)
print(f"\nAutoML predictions saved to {OUTPUT_DIR}/automl_predictions.csv")

# Show comparison
print("\nTop 15 HR predictions — scikit-learn vs AutoML:")
sklearn_preds = pd.read_csv(os.path.join(OUTPUT_DIR, "batting_predictions_sklearn.csv"))
merged = sklearn_preds[["Player", "sklearn_AVG", "sklearn_HR"]].merge(
    automl_preds.rename(columns={"fullName": "Player"})[["Player", "automl_AVG", "automl_HR"]],
    on="Player", how="left"
)
print(merged.sort_values("sklearn_HR", ascending=False).head(15).to_string(index=False))
