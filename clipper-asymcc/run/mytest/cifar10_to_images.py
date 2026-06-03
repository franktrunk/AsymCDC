import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
from PIL import Image

def cifar10_to_images(data_path="./data/cifar10", output_path="./data/cifar10/test"):
    """将CIFAR-10测试集转换为图片文件"""
    os.makedirs(output_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    for idx, (img, label) in enumerate(testset):
        # img shape: [3, 32, 32]
        img = img.permute(1, 2, 0)  # 转成 [32, 32, 3]
        img = (img.numpy() * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(output_path, f"{idx}.png"))
        
        if (idx + 1) % 1000 == 0:
            print(f"已转换 {idx + 1}/{len(testset)} 张图片")

    print(f"测试集图片保存在: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    cifar10_to_images()
