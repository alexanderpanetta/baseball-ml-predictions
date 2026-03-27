"""
Reproducibility Verification Script
====================================
Run this AFTER running 01_pull_data.py and 02_sklearn_models.py.
It checks that your outputs match the original author's outputs exactly.

Usage: python3 verify_reproducibility.py
"""
import hashlib
import sys
import os

EXPECTED = {
    "data/batting_raw.csv": "8e44bcbbbff7e9cfb7f7ad956ce2aaabe46812b8e927d6773822ba33e3e1051a",
    "data/pitching_raw.csv": "cb4044856ad14701792520cd89783b8bfa82d63da29ae958eb64497e40bb7e24",
    "data/people.csv": "c80c3f5c009b5be16544d5cfcbc3bac013cbc30ca08f822c27c5b87e5993ea27",
    "output/batting_predictions_sklearn.csv": "f5fbaaca7f9a61a699d4b6f1cf3d9a15574be96c5daa92510932eba55f08d226",
    "output/pitching_predictions_sklearn.csv": "e8c2fb1a97ccaaf2d466bf6762f8281f681ba5324fe8dfd65f29fb3ca432852e",
}

BASE = os.path.dirname(os.path.abspath(__file__))
all_pass = True

print("=" * 60)
print("REPRODUCIBILITY VERIFICATION")
print("=" * 60)
print()

for filepath, expected_hash in EXPECTED.items():
    full_path = os.path.join(BASE, filepath)
    if not os.path.exists(full_path):
        print(f"  MISSING  {filepath}")
        all_pass = False
        continue

    with open(full_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    if actual_hash == expected_hash:
        print(f"  PASS     {filepath}")
    else:
        print(f"  FAIL     {filepath}")
        print(f"           Expected: {expected_hash}")
        print(f"           Got:      {actual_hash}")
        all_pass = False

print()
if all_pass:
    print("ALL CHECKS PASSED — your results are identical to the original.")
else:
    print("SOME CHECKS FAILED — see above. Common causes:")
    print("  1. Different Python or scikit-learn version (see requirements.txt)")
    print("  2. Lahman source repo was updated (use data/ CSVs instead of re-downloading)")
    print("  3. Files were modified after generation")

sys.exit(0 if all_pass else 1)
