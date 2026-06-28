"""
Verify Pipeline Outputs
"""

from pathlib import Path

required_files = [

    "models/best_model.pkl",

    "models/preprocessor.pkl",

    "reports/model_results.csv",

    "reports/metrics.json"

]

print("=" * 60)

print("Pipeline Verification")

print("=" * 60)

missing = False

for file in required_files:

    path = Path(file)

    if path.exists():

        print(f"✓ {file}")

    else:

        print(f"✗ {file}")

        missing = True

if missing:

    raise FileNotFoundError(
        "Pipeline verification failed."
    )

print()

print("Pipeline Verified Successfully")