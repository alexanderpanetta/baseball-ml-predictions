"""
Step 5: Recover AutoML predictions and run HR batch prediction.
The AVG batch prediction completed but failed to download.
The HR model trained but batch prediction never ran.
"""
import pandas as pd
import numpy as np
import os
from google.cloud import aiplatform, storage

PROJECT_ID = "shiller-cape-analysis"
REGION = "us-central1"
BUCKET = "shiller-cape-data-panetta"
OUTPUT_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/output"

aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}")
client = storage.Client(project=PROJECT_ID)
bucket_obj = client.bucket(BUCKET)

# ============================================================
# 1. Download AVG batch predictions (already completed)
# ============================================================
print("Downloading AVG batch predictions...")
prefix = "baseball_ml_2026/predictions_AVG/"
blobs = list(bucket_obj.list_blobs(prefix=prefix))
print(f"  Found {len(blobs)} files in {prefix}")
for b in blobs:
    print(f"    {b.name}")

csv_blobs = [b for b in blobs if b.name.endswith(".csv")]
if csv_blobs:
    local_path = os.path.join(OUTPUT_DIR, "automl_avg_raw.csv")
    csv_blobs[0].download_to_filename(local_path)
    avg_preds = pd.read_csv(local_path)
    print(f"  AVG predictions: {len(avg_preds)} rows")
    print(f"  Columns: {avg_preds.columns.tolist()}")
    print(avg_preds.head())
else:
    # Check for jsonl
    jsonl_blobs = [b for b in blobs if b.name.endswith(".jsonl")]
    if jsonl_blobs:
        local_path = os.path.join(OUTPUT_DIR, "automl_avg_raw.jsonl")
        jsonl_blobs[0].download_to_filename(local_path)
        avg_preds = pd.read_json(local_path, lines=True)
        print(f"  AVG predictions (jsonl): {len(avg_preds)} rows")
        print(f"  Columns: {avg_preds.columns.tolist()}")
        print(avg_preds.head())

# ============================================================
# 2. Run HR batch prediction
# ============================================================
print("\nRunning HR batch prediction...")

# Find the HR model
models = aiplatform.Model.list(
    filter='display_name="baseball_HR_2026_model"',
    order_by="create_time desc"
)
if models:
    hr_model = models[0]
    print(f"  Found HR model: {hr_model.resource_name}")
else:
    print("  ERROR: HR model not found. Listing all models...")
    all_models = aiplatform.Model.list(order_by="create_time desc")
    for m in all_models[:10]:
        print(f"    {m.display_name}: {m.resource_name}")
    raise RuntimeError("Cannot find HR model")

# Upload prediction input for HR
gcs_pred_path = "baseball_ml_2026/predict_input_HR.csv"
pred_input = pd.read_csv(os.path.join(OUTPUT_DIR, "automl_train_HR.csv"))

# We need the prediction input (2026 features), not training data
# Build it from the same features used in training, but for 2025 players
# Actually, let's just use the feature columns from the AVG prediction input
# since they're the same features
avg_input_blob = bucket_obj.blob("baseball_ml_2026/predict_input_AVG.csv")
if avg_input_blob.exists():
    print("  Using existing prediction input from AVG...")
    # Copy to HR path
    bucket_obj.copy_blob(avg_input_blob, bucket_obj, "baseball_ml_2026/predict_input_HR.csv")
    gcs_pred_uri = f"gs://{BUCKET}/baseball_ml_2026/predict_input_HR.csv"
else:
    print("  Need to create prediction input...")
    # Read batting data and build features
    DATA_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/data"
    batting = pd.read_csv(os.path.join(DATA_DIR, "batting_raw.csv"))
    batting = batting.sort_values(["playerID", "yearID"])

    BATTING_TARGETS = ["AVG", "R", "H", "HR", "RBI", "2B", "SB", "BB", "OBP", "SLG"]

    batting_2025 = batting[batting["yearID"] == 2025].copy()
    predict_rows = []
    for _, player in batting_2025.iterrows():
        pid = player["playerID"]
        history = batting[batting["playerID"] == pid].sort_values("yearID")
        row = {}
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
    local_pred = os.path.join(OUTPUT_DIR, "automl_predict_input_HR.csv")
    predict_df.to_csv(local_pred, index=False)
    blob = bucket_obj.blob(gcs_pred_path)
    blob.upload_from_filename(local_pred)
    gcs_pred_uri = f"gs://{BUCKET}/{gcs_pred_path}"

gcs_pred_uri = f"gs://{BUCKET}/baseball_ml_2026/predict_input_HR.csv"
gcs_output_uri = f"gs://{BUCKET}/baseball_ml_2026/predictions_HR/"

batch_job = hr_model.batch_predict(
    job_display_name="baseball_HR_batch_predict",
    gcs_source=gcs_pred_uri,
    gcs_destination_prefix=gcs_output_uri,
    instances_format="csv",
    predictions_format="csv",
)
print(f"  HR batch prediction complete")

# Download HR predictions
hr_blobs = list(bucket_obj.list_blobs(prefix="baseball_ml_2026/predictions_HR/"))
hr_csv = [b for b in hr_blobs if b.name.endswith(".csv")]
hr_jsonl = [b for b in hr_blobs if b.name.endswith(".jsonl")]

if hr_csv:
    local_path = os.path.join(OUTPUT_DIR, "automl_hr_raw.csv")
    hr_csv[0].download_to_filename(local_path)
    hr_preds = pd.read_csv(local_path)
    print(f"  HR predictions: {len(hr_preds)} rows")
    print(hr_preds.head())
elif hr_jsonl:
    local_path = os.path.join(OUTPUT_DIR, "automl_hr_raw.jsonl")
    hr_jsonl[0].download_to_filename(local_path)
    hr_preds = pd.read_json(local_path, lines=True)
    print(f"  HR predictions (jsonl): {len(hr_preds)} rows")
    print(hr_preds.head())

print("\nDone! Check output/ for automl_avg_raw and automl_hr_raw files.")
