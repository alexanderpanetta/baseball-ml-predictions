"""
Step 1: Pull historical MLB batting and pitching data from the Lahman Database (2025 edition).
Source: SABR Lahman Database via cdalzell/Lahman R package on GitHub.
Uses .RData files which contain data through the 2025 season.

Reproducibility: pip install pandas requests pyreadr scikit-learn openpyxl
Then run: python3 01_pull_data.py
"""
import pandas as pd
import pyreadr
import os
import requests
import tempfile
from zipfile import ZipFile
from io import BytesIO

OUTPUT_DIR = "/Users/alexpanetta/Desktop/Baseball_ML_Predictions/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download from cdalzell/Lahman GitHub (SABR 2025 edition)
REPO_URL = "https://github.com/cdalzell/Lahman/archive/refs/heads/master.zip"
print("Downloading Lahman database (2025 SABR edition) from GitHub...")
resp = requests.get(REPO_URL, timeout=120, allow_redirects=True)
resp.raise_for_status()
z = ZipFile(BytesIO(resp.content))

tmpdir = tempfile.mkdtemp()

def read_rdata(name):
    """Extract and read an RData file from the Lahman repo zip."""
    path = f"Lahman-master/data/{name}.RData"
    z.extract(path, tmpdir)
    result = pyreadr.read_r(os.path.join(tmpdir, path))
    return list(result.values())[0]

batting_lahman = read_rdata("Batting")
pitching_lahman = read_rdata("Pitching")
people = read_rdata("People")

# R prefixes numeric column names with 'X' — rename back
batting_lahman = batting_lahman.rename(columns={"X2B": "2B", "X3B": "3B"})

# Convert object columns that should be numeric
for col in ["RBI", "SB", "CS", "SO", "IBB", "HBP", "SH", "SF", "GIDP"]:
    if col in batting_lahman.columns:
        batting_lahman[col] = pd.to_numeric(batting_lahman[col], errors="coerce")
for col in pitching_lahman.columns:
    if pitching_lahman[col].dtype == "object" and col not in ["playerID", "teamID", "lgID"]:
        pitching_lahman[col] = pd.to_numeric(pitching_lahman[col], errors="coerce")

print(f"Full Lahman batting: {len(batting_lahman)} rows, years {batting_lahman['yearID'].min()}-{batting_lahman['yearID'].max()}")
print(f"Full Lahman pitching: {len(pitching_lahman)} rows, years {pitching_lahman['yearID'].min()}-{pitching_lahman['yearID'].max()}")

# Filter to 2015+ (Statcast era)
batting_lahman = batting_lahman[batting_lahman["yearID"] >= 2015].copy()
pitching_lahman = pitching_lahman[pitching_lahman["yearID"] >= 2015].copy()

# People table for names and birth year
people_slim = people[["playerID", "nameFirst", "nameLast", "birthYear", "birthMonth"]].copy()
people_slim["fullName"] = people_slim["nameFirst"].fillna("") + " " + people_slim["nameLast"].fillna("")

batting_lahman = batting_lahman.merge(people_slim, on="playerID", how="left")
pitching_lahman = pitching_lahman.merge(people_slim, on="playerID", how="left")

# ---- BATTING: aggregate stints & compute stats ----
b = batting_lahman.copy()
agg_cols = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP", "G"]
id_cols = ["playerID", "yearID", "fullName", "birthYear", "birthMonth", "nameFirst", "nameLast"]
b[agg_cols] = b[agg_cols].fillna(0)
b = b.groupby(id_cols, as_index=False)[agg_cols].sum()

b["PA"] = b["AB"] + b["BB"] + b["HBP"] + b["SF"] + b["SH"]
b["AVG"] = (b["H"] / b["AB"]).round(3)
b["OBP"] = ((b["H"] + b["BB"] + b["HBP"]) / (b["AB"] + b["BB"] + b["HBP"] + b["SF"])).round(3)
b["SLG"] = ((b["H"] - b["2B"] - b["3B"] - b["HR"] + 2*b["2B"] + 3*b["3B"] + 4*b["HR"]) / b["AB"]).round(3)
b["1B"] = b["H"] - b["2B"] - b["3B"] - b["HR"]
b["age"] = b["yearID"] - b["birthYear"]

b = b[b["PA"] >= 200].copy()
b = b.replace([float('inf'), float('-inf')], float('nan')).dropna(subset=["AVG", "OBP", "SLG"])

# ---- PITCHING: aggregate stints & compute stats ----
p = pitching_lahman.copy()
p_agg_cols = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "HR", "BB", "SO", "IBB", "HBP", "WP", "BFP", "R"]
p_id_cols = ["playerID", "yearID", "fullName", "birthYear", "birthMonth", "nameFirst", "nameLast"]
p[p_agg_cols] = p[p_agg_cols].fillna(0)
p = p.groupby(p_id_cols, as_index=False)[p_agg_cols].sum()

p["IP"] = p["IPouts"] / 3
p["WHIP"] = ((p["BB"] + p["H"]) / p["IP"]).round(3)
p["ERA"] = ((p["ER"] * 9) / p["IP"]).round(2)
p["K9"] = ((p["SO"] * 9) / p["IP"]).round(2)
p["BB9"] = ((p["BB"] * 9) / p["IP"]).round(2)
p["HR9"] = ((p["HR"] * 9) / p["IP"]).round(2)
p["age"] = p["yearID"] - p["birthYear"]

p = p[p["IP"] >= 50].copy()
p = p.replace([float('inf'), float('-inf')], float('nan')).dropna(subset=["ERA", "WHIP"])

# Save raw data
b.to_csv(os.path.join(OUTPUT_DIR, "batting_raw.csv"), index=False)
p.to_csv(os.path.join(OUTPUT_DIR, "pitching_raw.csv"), index=False)
people_slim.to_csv(os.path.join(OUTPUT_DIR, "people.csv"), index=False)

print(f"\n{'='*60}")
print(f"BATTING: {len(b)} player-seasons (2015+, 200+ PA)")
print(f"PITCHING: {len(p)} player-seasons (2015+, 50+ IP)")
print(f"Years covered: {sorted(b['yearID'].unique())}")
print(f"{'='*60}")

print("\n--- Most recent year batting (top 15 by HR) ---")
latest = b["yearID"].max()
bl = b[b["yearID"] == latest].sort_values("HR", ascending=False)
print(f"Year: {latest}, {len(bl)} qualified batters")
print(bl[["fullName","yearID","AVG","R","H","HR","RBI","2B","SB","BB","OBP","SLG"]].head(15).to_string())

print(f"\n--- Most recent year pitching (top 15 by ERA) ---")
latest_p = p["yearID"].max()
pl = p[p["yearID"] == latest_p].sort_values("ERA")
print(f"Year: {latest_p}, {len(pl)} qualified pitchers")
print(pl[["fullName","yearID","ERA","W","L","SO","WHIP","IP"]].head(15).to_string())
