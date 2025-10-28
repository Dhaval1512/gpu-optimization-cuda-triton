# ---------------------------------------------------------------------
# 🔽 Save results also as CSV and Excel for easy analysis
# ---------------------------------------------------------------------
import os, json
import pandas as pd
out_json = "profiling/phase2_individuals_mnist.json"

# ✅ create folder if needed
os.makedirs("profiling", exist_ok=True)

# ✅ if file doesn't exist, create empty list
if not os.path.exists(out_json):
    with open(out_json, "w") as f:
        json.dump([], f)

out_csv  = "profiling/phase2_individuals_mnist.csv"
out_xlsx = "profiling/phase2_individuals_mnist.xlsx"

# Load the list of dictionaries we just dumped
with open(out_json, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.to_csv(out_csv, index=False)
df.to_excel(out_xlsx, index=False)

print(f"✅ Results also saved as:\n  {out_csv}\n  {out_xlsx}")
