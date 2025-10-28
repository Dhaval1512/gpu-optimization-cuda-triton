import os
import glob
import json
import pandas as pd

# Ensure summary directory exists
os.makedirs("report/summary", exist_ok=True)

# Collect all experiment JSONs
files = glob.glob("report/runs/*.json")

if not files:
    print("⚠️ No JSON files found in report/runs/. Run some experiments first!")
    exit()

records = []
for f in files:
    with open(f, "r") as infile:
        data = json.load(infile)
        # Extract experiment name from filename (e.g., "2025-10-28T17-20-30_baseline.json" → "baseline")
        exp_name = os.path.basename(f).replace(".json", "").split("_", 1)[-1]
        data["experiment_name"] = exp_name
        records.append(data)

# Convert to DataFrame
df = pd.DataFrame(records)

# Reorder important columns for clarity
cols = [
    "experiment_name", "device", "dataset", "num_epochs", "batch_size", "lr",
    "fp16", "best_accuracy_pct", "avg_epoch_time_sec",
    "avg_throughput_images_per_sec", "peak_gpu_memory_MB", "model_path"
]
df = df[[c for c in cols if c in df.columns]]

# Save as CSV and Excel
csv_path = "report/summary/performance_summary.csv"
xlsx_path = "report/summary/performance_summary.xlsx"
df.to_csv(csv_path, index=False)
df.to_excel(xlsx_path, index=False)

# Print preview in terminal
print("\n✅ Summary saved:")
print(f" - CSV:   {csv_path}")
print(f" - Excel: {xlsx_path}")
print("\n📊 Preview:\n")
print(df.head())
