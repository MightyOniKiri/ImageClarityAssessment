import random
from pathlib import Path
import os
import shutil
from collections import defaultdict
import torch

from torchvision import datasets, transforms

def split_dataset(ratios_r = 0.8, seed = 42):
    source_data_dir = Path(__file__).resolve().parent.parent.parent / 'CERTH_ImageBlurDataset' / 'TrainingSet'

    # 新建文件夹，把新的数据集存放在新的文件夹里边
    data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
    ratios = (ratios_r, 1-ratios_r)
    image_types = {".jpg"}

    if data_dir.exists() and any(data_dir.iterdir()):
        raise ValueError(f"用来存放划分数据集的文件夹内不为空：{data_dir}")

    class_map = {
        "Undistorted": 0,
        "Artificially-Blurred": 1,
        "Naturally-Blurred": 1
    }

    random.seed(seed)

    class_images = defaultdict(list)
    for group_name, class_name in class_map.items():
        group_folder = source_data_dir / group_name

        if not group_folder.is_dir():
            raise ValueError(f"找不到组文件夹：{group_folder}")

        for image in group_folder.rglob("*"):
            if image.is_file() and image.suffix.lower() in image_types:
                class_images[class_name].append((group_name, image))
            
    counts = {'train': 0, 'val': 0}

    for raw_label, images in class_images.items():
        random.shuffle(images)
        n_train = int(len(images) * ratios[0])
        splits = {
            'train': images[:n_train],
            'val': images[n_train:]
        }
        
        for split_name, split_items in splits.items():
            target_folder = data_dir / split_name / f"class{raw_label}"
            target_folder.mkdir(parents=True, exist_ok=True)

            for index, (group_name, image) in enumerate(split_items):
                # 加组名和序号，避免 B、C 中同名文件互相覆盖
                target_name = f"{group_name}_{index:05d}_{image.name}"
                shutil.copy2(image, target_folder / target_name)
                counts[split_name] += 1

    print(counts)
    
if __name__ == '__main__':
    split_dataset(ratios_r = 0.8, seed = 42)
    
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = Path(__file__).resolve().parent.parent.parent / 'data'
image_loader = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
data_sets = {x: torch.utils.data.DataLoader(image_loader[x], batch_size = 4) for x in ['train', 'val']}
dataset_sizes = {x: len(image_loader[x]) for x in ['train', 'val']}
class_names = image_loader['train'].classes

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"