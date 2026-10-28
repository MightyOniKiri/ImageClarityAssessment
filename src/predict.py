import argparse
from pathlib import Path

import cv2
import numpy as np

from clarity import laplacian_score, tenengrad_score

# 定义终端可以输入的参数
parser = argparse.ArgumentParser(description="检测图片是否模糊")
parser.add_argument("--image", required=True, help="图片路径")
parser.add_argument(
    "--method",
    choices=["laplacian", "tenengrad"],
    default="tenengrad"
)
args = parser.parse_args()

# 读取图片
image_path = Path(args.image)

if not image_path.is_file():
    parser.error(f"图片不存在：{image_path}")

image = cv2.imdecode(
    np.fromfile(str(image_path), dtype=np.uint8),
    cv2.IMREAD_COLOR
)

if image is None:
    parser.error(f"图片读取失败：{image_path}")

# 根据参数选择方法及对应阈值
if args.method == "laplacian":
    score = laplacian_score(image)
    threshold = 40
else:
    score = tenengrad_score(image)
    threshold = 1000

result = "模糊" if score < threshold else "清晰"

print(f"方法：{args.method}")
print(f"清晰度分数：{score:.2f}")
print(f"判断阈值：{threshold}")
print(f"结果：{result}")