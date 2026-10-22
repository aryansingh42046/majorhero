from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

report_path = Path("datasets/manifests/preprocessing_report.csv")
output_path = Path("datasets/manifests/iqa_report.csv")
df = pd.read_csv(report_path)
df = df[df["status"].astype(str).str.lower() == "ok"].copy()

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="IQA"):
    path = Path(str(row["output_path"]))
    record = {
        "dataset": row["dataset"],
        "image_path": str(path),
        "blur_score": np.nan,
        "mean_brightness": np.nan,
        "dark_pixel_pct": np.nan,
        "bright_pixel_pct": np.nan,
        "status": "ok",
        "quality_flag": "review"
    }

    try:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Could not read image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        dark_pct = float((gray < 15).mean() * 100)
        bright_pct = float((gray > 245).mean() * 100)

        flags = []
        if blur < 50:
            flags.append("possible_blur")
        if brightness < 35:
            flags.append("possibly_dark")
        if brightness > 220:
            flags.append("possibly_overexposed")
        if not flags:
            flags.append("no_simple_threshold_flag")

        record.update({
            "blur_score": round(blur, 3),
            "mean_brightness": round(brightness, 3),
            "dark_pixel_pct": round(dark_pct, 3),
            "bright_pixel_pct": round(bright_pct, 3),
            "quality_flag": ";".join(flags)
        })

    except Exception as exc:
        record["status"] = "error"
        record["quality_flag"] = str(exc)

    results.append(record)

out = pd.DataFrame(results)
output_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(output_path, index=False)

print("\nIQA report:", output_path)
print("Images checked:", len(out))
print("\nQuality flags:")
print(out["quality_flag"].value_counts().to_string())
print("\nProcessing status:")
print(out["status"].value_counts().to_string())
print("\nFlag counts by dataset:")
print(out.groupby(["dataset", "quality_flag"]).size().to_string())
print("\nThese thresholds are preliminary screening flags, not clinical quality labels.")
