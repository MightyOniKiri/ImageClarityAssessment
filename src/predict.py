import argparse
from pathlib import Path
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

from clarity import laplacian_score, tenengrad_score

# 定义终端可以输入的参数
parser = argparse.ArgumentParser(description="检测图片是否模糊")
parser.add_argument(
    "--image", 
    required=True, 
    help="图片路径"
)
parser.add_argument(
    "--method",
    choices=["laplacian", "tenengrad", "ResNet18"],
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
    result = "模糊" if score < threshold else "清晰"
elif args.method == "tenengrad":
    score = tenengrad_score(image)
    threshold = 1000
    result = "模糊" if score < threshold else "清晰"
elif args.method == "ResNet18":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = Path(__file__).resolve().parent / "ResNet18Blur" / "best_model_params.pt"
    
    # 结构必须与训练时一致：ResNet18，最后输出 2 类
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(weight_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    # 必须与训练时的验证预处理一致
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted = probabilities.argmax().item()

    class_names = ["清晰", "模糊"]
    result = class_names[predicted]

print(f"方法：{args.method}")
if args.method == "laplacian" or args.method == "tenengrad":
    print(f"清晰度分数：{score:.2f}")
    print(f"判断阈值：{threshold}")
print(f"结果：{result}")