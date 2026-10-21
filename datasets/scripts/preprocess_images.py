
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets" / "manifests" / "dataset_manifest.csv"
OUT_ROOT = ROOT / "datasets" / "processed"
SIZE = 512

# Keep CLAHE disabled initially; compare it as a separate experiment.
USE_CLAHE = False

def preprocess(src: Path, dst: Path):
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            rgb = np.array(im)

        # Crop black borders conservatively, avoiding an empty crop.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray > 8
        if mask.any():
            ys, xs = np.where(mask)
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            if x2 > x1 and y2 > y1:
                rgb = rgb[y1:y2, x1:x2]

        if USE_CLAHE:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            rgb = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

        h, w = rgb.shape[:2]
        scale = min(SIZE / w, SIZE / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        x = (SIZE - nw) // 2
        y = (SIZE - nh) // 2
        canvas[y:y+nh, x:x+nw] = resized

        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(dst, quality=95)
        return "ok", ""
    except Exception as exc:
        return "error", str(exc)

def main():
    df = pd.read_csv(MANIFEST)
    results = []

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        src = Path(row.image_path)
        dst = OUT_ROOT / row.dataset / f"{src.stem}.jpg"

        status, error = preprocess(src, dst)
        results.append({
            "dataset": row.dataset,
            "image_path": str(src),
            "output_path": str(dst),
            "status": status,
            "error": error,
        })

    report = pd.DataFrame(results)
    report_path = ROOT / "datasets" / "manifests" / "preprocessing_report.csv"
    report.to_csv(report_path, index=False)

    print("\nPreprocessing complete")
    print(report["status"].value_counts().to_string())
    print("Report:", report_path)
    print("Output:", OUT_ROOT)

if __name__ == "__main__":
    main()
