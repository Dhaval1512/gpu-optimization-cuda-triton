# ---------------------------------------------------------------------
# 🔽 Save results also as CSV and Excel for easy analysis
# ---------------------------------------------------------------------
import os, json
import pandas as pd


def _debug(msg: str) -> None:
    """Emit lightweight debug output without altering functionality."""
    print(f"[DEBUG] {msg}", flush=True)
out_json = "profiling/phase2_individuals_mnist.json"

# ✅ create folder if needed
os.makedirs("profiling", exist_ok=True)
_debug("Ensured profiling directory exists")

# ✅ if file doesn't exist, create empty list
if not os.path.exists(out_json):
    with open(out_json, "w") as f:
        json.dump([], f)
    _debug(f"Created new JSON store at {out_json}")
else:
    _debug(f"Found existing JSON store at {out_json}")

out_csv  = "profiling/phase2_individuals_mnist.csv"
out_xlsx = "profiling/phase2_individuals_mnist.xlsx"
_debug(f"Output CSV path: {out_csv}")
_debug(f"Output XLSX path: {out_xlsx}")

# Load the list of dictionaries we just dumped
with open(out_json, "r") as f:
    data = json.load(f)
_debug(f"Loaded {len(data)} records from JSON")

df = pd.DataFrame(data)
_debug(f"Created DataFrame with shape {df.shape}")
df.to_csv(out_csv, index=False)
_debug("Wrote CSV output")
df.to_excel(out_xlsx, index=False)
_debug("Wrote Excel output")

print(f"✅ Results also saved as:\n  {out_csv}\n  {out_xlsx}")
