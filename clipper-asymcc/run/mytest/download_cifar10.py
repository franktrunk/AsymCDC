import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_cifar10(data_path=""):
    """下载并保存CIFAR-10数据集"""
    os.makedirs(data_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 下载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform
    )

    # 下载测试集
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )

    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"数据保存在: {os.path.abspath(data_path)}")

if __name__ == "__main__":
    download_cifar10()
