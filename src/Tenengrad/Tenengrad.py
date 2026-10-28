"""
Tenengrad for Image Clarity Assessment
"""

from pathlib import Path
import numpy as np
import csv
import cv2

project_dir = Path(__file__).resolve().parent.parent.parent
training_dir = project_dir / "CERTH_ImageBlurDataset" / "TrainingSet"
folders = [path for path in training_dir.iterdir() if path.is_dir()]

output_dir = project_dir / "outputs" / "tenengrad"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "tenengrad_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label", "score"])
    for path in folders:
        for image_path in path.glob("*.jpg"):
            #image = cv2.imread(str(image_path))
            image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"读取失败，跳过：{image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            score = (gx ** 2 + gy ** 2).mean()
            
            label = "clear" if path.name == "Undistorted" else "blurry"

            writer.writerow([
                str(image_path.relative_to(training_dir)),
                label,
                score,
            ])
            
print(f"Tenengrad scores saved to {output_dir / 'tenengrad_scores.csv'}")

        