"""
Analyze the Laplacian variance scores of images in the training set.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent.parent.parent
csv_path = project_dir / "outputs" / "tenengrad" / "tenengrad_scores.csv"

data = pd.read_csv(csv_path)

# 从路径中提取类别文件夹名
data["category"] = (
    data["image_path"]
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[0]
)

result = data.groupby("category")["score"].agg(
    ["count", "min", "median", "max"]
)

print(result)

for category, group in data.groupby("category"):
    plt.hist(
        np.log1p(group["score"]),
        bins=30,
        alpha=0.4,
        label=category
    )

plt.title("Tenengrad Score Distribution by Image Category")
plt.xlabel("log(1 + score)")    # 因为分数从接近 0 到 2,500 多，转换后更容易看清分布
plt.ylabel("Number of images")
plt.legend()
plt.tight_layout()
plt.savefig(project_dir / "outputs" / "tenengrad" / "tenengrad_scores_histogram.png")

for threshold in [500, 1000, 1500, 2000, 2500, 3000, 4000]:
    predicted = data["score"] >= threshold
    actual = data["label"] == "clear"

    # 模糊图片被判为清晰
    blurry_as_clear = (predicted & ~actual).sum()

    # 清晰图片被判为模糊
    clear_as_blurry = (~predicted & actual).sum()

    accuracy = (predicted == actual).mean()

    print(
        f"阈值：{threshold:3d} | "
        f"模糊被放行：{blurry_as_clear:3d}/370 | "
        f"清晰被误删：{clear_as_blurry:3d}/630 | "
        f"准确率：{accuracy:.2%}"
    )
