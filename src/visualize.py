import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
RECORD_DIR = PROJECT_ROOT / "recordings"

csvs = sorted(glob.glob(str(RECORD_DIR / "cloud_*.csv")))
if not csvs:
    raise RuntimeError("No cloud_*.csv found in recordings/")

csv_path = csvs[-1]
print("Using:", csv_path)

df = pd.read_csv(csv_path)

# Force numeric (fixes 'blank plot' when values are strings)
df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce")
df["cloud_coverage"] = pd.to_numeric(df["cloud_coverage"], errors="coerce")

df = df.dropna(subset=["t_sec", "cloud_coverage"]).reset_index(drop=True)

print("rows after dropna:", len(df))
print("cloud min/max:", df["cloud_coverage"].min(), df["cloud_coverage"].max())

plt.figure(figsize=(10, 5))
plt.scatter(df["t_sec"], df["cloud_coverage"], s=6)
plt.plot(df["t_sec"], df["cloud_coverage"], linewidth=1)

plt.ylim(-0.05, 1.05)          # force visible even if flat
plt.xlim(df["t_sec"].min(), df["t_sec"].max())

plt.xlabel("Time (s)")
plt.ylabel("Cloud Coverage (0–1)")
plt.title("Cloud Coverage vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()