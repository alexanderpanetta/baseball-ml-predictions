"""
Step 3: Create Excel spreadsheets with predictions and methodology.
Produces a multi-sheet Excel file ready for upload to Google Sheets.

Sheets:
  1. Batting Predictions (2026) — scikit-learn (+ AutoML column placeholders)
  2. Pitching Predictions (2026) — scikit-learn (+ AutoML column placeholders)
  3. Methodology — full reproducibility documentation
"""
import pandas as pd
import json
import os
from datetime import datetime

OUTPUT_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/output"

# Load predictions and metrics
batting = pd.read_csv(os.path.join(OUTPUT_DIR, "batting_predictions_sklearn.csv"))
pitching = pd.read_csv(os.path.join(OUTPUT_DIR, "pitching_predictions_sklearn.csv"))
with open(os.path.join(OUTPUT_DIR, "model_metrics.json")) as f:
    metrics = json.load(f)

# Load AutoML predictions if available
automl_path = os.path.join(OUTPUT_DIR, "automl_predictions.csv")
if os.path.exists(automl_path):
    automl = pd.read_csv(automl_path)
    automl = automl.rename(columns={"fullName": "Player"})
    print(f"Loaded AutoML predictions: {len(automl)} rows, columns: {automl.columns.tolist()}")
else:
    automl = None
    print("No AutoML predictions found — using placeholders.")

# ============================================================
# SHEET 1: Batting Predictions
# ============================================================
bat_display = batting.drop(columns=["playerID"]).copy()

# Merge AutoML predictions on playerID (not name — there are duplicate names like Max Muncy)
if automl is not None:
    automl_merge = automl.rename(columns={"Player": "Player_automl"})
    bat_display = batting.copy()  # re-start from batting which has playerID
    bat_display = bat_display.merge(
        automl_merge[["playerID"] + [c for c in automl_merge.columns if c.startswith("automl_")]],
        on="playerID", how="left"
    )
    bat_display = bat_display.drop(columns=["playerID"])

BATTING_STATS = ["AVG", "R", "H", "HR", "RBI", "2B", "SB", "BB", "OBP", "SLG"]

# Reorder: Player, then for each stat: sklearn, AutoML side by side
ordered_cols = ["Player"]
for stat in BATTING_STATS:
    ordered_cols.append(f"sklearn_{stat}")
    automl_col = f"automl_{stat}"
    display_col = f"AutoML_{stat}"
    if automl_col in bat_display.columns:
        bat_display[display_col] = bat_display[automl_col]
    else:
        bat_display[display_col] = ""
    ordered_cols.append(display_col)

bat_display = bat_display[ordered_cols]

# ============================================================
# SHEET 2: Pitching Predictions
# ============================================================
pitch_display = pitching.drop(columns=["playerID"]).copy()

PITCHING_STATS = ["ERA", "W", "L", "SO", "WHIP", "IP", "SV", "HR", "BB"]

ordered_cols_p = ["Player"]
for stat in PITCHING_STATS:
    ordered_cols_p.append(f"sklearn_{stat}")
    pitch_display[f"AutoML_{stat}"] = ""  # placeholder
    ordered_cols_p.append(f"AutoML_{stat}")

pitch_display = pitch_display[ordered_cols_p]

# ============================================================
# SHEET 3: Methodology
# ============================================================
methodology_rows = [
    ["Play Ball! What Machine Learning Predicts This Baseball Season"],
    ["Methodology & Reproducibility Guide"],
    [""],
    ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M")],
    ["Author:", "Alex Panetta"],
    [""],
    ["=" * 60],
    ["DATA SOURCE"],
    ["=" * 60],
    ["Database:", "SABR Lahman Baseball Database, 2025 Edition"],
    ["URL:", "https://sabr.org/lahman-database/"],
    ["License:", "Creative Commons Attribution-ShareAlike 3.0"],
    ["Downloaded via:", "cdalzell/Lahman R package on GitHub (RData files)"],
    ["Coverage:", "1871-2025 (we use 2015-2025 for Statcast-era relevance)"],
    [""],
    ["Batting filter:", "Players with 200+ plate appearances per season"],
    ["Pitching filter:", "Players with 50+ innings pitched per season"],
    ["Training data:", f"{metrics['training_data']['batting_seasons']}"],
    ["Total batting player-seasons:", "3,688"],
    ["Total pitching player-seasons:", "3,502"],
    [""],
    ["=" * 60],
    ["SCIKIT-LEARN MODEL"],
    ["=" * 60],
    ["Algorithm:", "Gradient Boosting Regressor (sklearn.ensemble)"],
    ["n_estimators:", "200"],
    ["max_depth:", "4"],
    ["learning_rate:", "0.05"],
    ["min_samples_leaf:", "10"],
    ["subsample:", "0.8"],
    ["random_state:", "42 (fixed for reproducibility)"],
    [""],
    ["One model is trained per target statistic."],
    [""],
    ["FEATURES (per player-season):"],
    ["1.", "Previous season stats (all target stats + PA, G, SO)"],
    ["2.", "Weighted 3-year rolling average (weights: 1/6, 2/6, 3/6 — more recent seasons weighted higher)"],
    ["3.", "Career average of each statistic"],
    ["4.", "Year-over-year trend (difference between last two seasons)"],
    ["5.", "Age and age² (quadratic aging curve modeling peak performance)"],
    ["6.", "Number of prior qualified seasons (experience proxy)"],
    ["7.", "Previous season plate appearances (batting) / GS/G ratio (pitching starter vs reliever)"],
    [""],
    ["TRAIN/TEST METHODOLOGY:"],
    ["", "1. Train on 2016-2024 predictions (features from prior years → predict current year)"],
    ["", "2. Test (holdout) on 2025 predictions — never seen during training"],
    ["", "3. Report MAE and R² on holdout set (see below)"],
    ["", "4. Retrain on full 2016-2025 data for final 2026 predictions"],
    ["", "5. 5-fold cross-validation MAE also reported"],
    [""],
    ["=" * 60],
    ["GOOGLE AUTOML MODEL"],
    ["=" * 60],
    ["Platform:", "Google Cloud AutoML Tables (Vertex AI)"],
    ["Training data:", "Same features and train/test split as scikit-learn"],
    ["Configuration:", "Default AutoML Tables settings, 1-hour training budget"],
    ["", "(AutoML results pending — columns marked 'AutoML_' in prediction sheets)"],
    [""],
    ["=" * 60],
    ["MODEL EVALUATION — 2025 HOLDOUT SET"],
    ["=" * 60],
    [""],
    ["BATTING (predicting 2025 season from prior data):"],
    ["Statistic", "MAE (Mean Abs Error)", "R² Score", "5-Fold CV MAE"],
]

for stat in BATTING_STATS:
    m = metrics["batting"].get(stat, {})
    methodology_rows.append([stat, m.get("MAE", ""), m.get("R2", ""), m.get("CV_MAE", "")])

methodology_rows += [
    [""],
    ["PITCHING (predicting 2025 season from prior data):"],
    ["Statistic", "MAE (Mean Abs Error)", "R² Score", "5-Fold CV MAE"],
]

for stat in PITCHING_STATS:
    m = metrics["pitching"].get(stat, {})
    methodology_rows.append([stat, m.get("MAE", ""), m.get("R2", ""), m.get("CV_MAE", "")])

methodology_rows += [
    [""],
    ["INTERPRETATION:"],
    ["", "MAE = average prediction error in the stat's natural units"],
    ["", "  e.g., AVG MAE of 0.020 means predictions are off by ~20 points of batting average"],
    ["", "  e.g., HR MAE of 6 means predictions are off by ~6 home runs"],
    ["", "R² = proportion of variance explained (1.0 = perfect, 0.0 = no better than average)"],
    ["", "Baseball is inherently variable — R² values of 0.2-0.5 are typical for season predictions"],
    [""],
    ["=" * 60],
    ["HOW TO REPRODUCE"],
    ["=" * 60],
    [""],
    ["Requirements: Python 3.10+, pip install pandas requests pyreadr scikit-learn openpyxl"],
    [""],
    ["Step 1: python3 01_pull_data.py"],
    ["  → Downloads Lahman database, computes derived stats, saves to data/"],
    [""],
    ["Step 2: python3 02_sklearn_models.py"],
    ["  → Trains models, evaluates on holdout, generates 2026 predictions"],
    [""],
    ["Step 3: python3 03_create_spreadsheets.py"],
    ["  → Creates this spreadsheet"],
    [""],
    ["All code available at: [TBD — link to GitHub repo or blog post]"],
    [""],
    ["KNOWN LIMITATIONS:"],
    ["1.", "Lahman database does not include spring training or minor league stats"],
    ["2.", "Rookies with <2 MLB seasons have limited feature data (fewer prior seasons to average)"],
    ["3.", "Injuries, trades, and role changes are not modeled — predictions assume ~full healthy season"],
    ["4.", "2020 COVID-shortened season (60 games) is included but may skew counting stats"],
    ["5.", "Model does not account for park factors, lineup changes, or schedule strength"],
]

methodology_df = pd.DataFrame(methodology_rows)

# ============================================================
# WRITE EXCEL FILE
# ============================================================
output_path = os.path.join(OUTPUT_DIR, "MLB_2026_Predictions.xlsx")

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    bat_display.to_excel(writer, sheet_name="Batting Predictions", index=False)
    pitch_display.to_excel(writer, sheet_name="Pitching Predictions", index=False)
    methodology_df.to_excel(writer, sheet_name="Methodology", index=False, header=False)

    # Auto-adjust column widths
    for sheet_name in writer.sheets:
        ws = writer.sheets[sheet_name]
        for column_cells in ws.columns:
            max_length = 0
            col_letter = column_cells[0].column_letter
            for cell in column_cells:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            ws.column_dimensions[col_letter].width = min(max_length + 2, 30)

print(f"Created: {output_path}")
print(f"  Sheet 1: Batting Predictions ({len(bat_display)} players)")
print(f"  Sheet 2: Pitching Predictions ({len(pitch_display)} players)")
print(f"  Sheet 3: Methodology")

# Also save CSVs for AutoML upload
batting.to_csv(os.path.join(OUTPUT_DIR, "batting_for_sheets.csv"), index=False)
pitching.to_csv(os.path.join(OUTPUT_DIR, "pitching_for_sheets.csv"), index=False)
print("\nCSVs also saved for Google Sheets import.")
