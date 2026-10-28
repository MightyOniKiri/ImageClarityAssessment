import argparse
import cv2
import numpy as np
from pathlib import Path
import pandas as pd


# 从项目根目录寻找标签文件
project_dir = Path(__file__).resolve().parent.parent.parent

# 测自然模糊时用 "NaturalBlurSet"
# 测人工模糊时用 "DigitalBlurSet"
parser = argparse.ArgumentParser(description="Evaluate Laplacian variance blur detection")
parser.add_argument("--dataset", choices=["NaturalBlurSet", "DigitalBlurSet"], default="NaturalBlurSet")
parser.add_argument("--show-errors", action="store_true", help="Open the first ten misclassified images")
args = parser.parse_args()
dataset_name = args.dataset

excel_path = project_dir/ "CERTH_ImageBlurDataset" / "EvaluationSet" / f"{dataset_name}.xlsx"
data = pd.read_excel(excel_path, usecols=[0, 1])
data.columns = ["Image Name", "Blur Label"]
data = data.dropna(subset=["Image Name", "Blur Label"])

image_dir = excel_path.parent / f"{dataset_name}"

#print(data.head())  # 显示前五行
#print(data.columns)  # 显示列名

results = []

for _, row in data.iterrows():
    # Excel 中没有扩展名，需要补上 .jpg
    name = str(row["Image Name"]).strip()
    filename = name if name.lower().endswith(".jpg") else f"{name}.jpg"
    image_path = image_dir / filename
    actual = int(row["Blur Label"])

    image = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(f"图片读取失败：{image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 1 为模糊，-1 为清晰
    predicted = 1 if score < 40 else -1

    results.append([name, actual, score, predicted])

results = pd.DataFrame(results, columns=["image_name", "actual", "score", "predicted"])

output_dir = project_dir / "outputs" / "laplacian_variance"
output_dir.mkdir(parents=True, exist_ok=True)
results.to_csv(
    output_dir / f"{dataset_name}_test.csv",
    index=False,
    encoding="utf-8-sig"
)

blurry_as_clear = ((results["actual"] == 1) & (results["predicted"] == -1)).sum()
clear_as_blurry = ((results["actual"] == -1) & (results["predicted"] == 1)).sum()
blurry_total = (results["actual"] == 1).sum()
clear_total = (results["actual"] == -1).sum()
accuracy = (results["actual"] == results["predicted"]).mean()

print(f"测试图片数：{len(results)}")
print(f"模糊被放行：{blurry_as_clear}/{blurry_total}")
print(f"清晰被误删：{clear_as_blurry}/{clear_total}")
print(f"准确率：{accuracy:.2%}")