"""
Prepare CIFAR-10 test set as image files for run_exp.py.

Usage:
    python prepare_dataset.py                              # default: ./data/cifar10_test
    python prepare_dataset.py --data_dir /path/to/dir      # custom output directory

Output structure:
    <data_dir>/
    ├── 0.png
    ├── 1.png
    ├── 2.png
    └── ...
        9999.png

These files are then consumed by:
    python run_exp.py --path <data_dir> --conf config/simple.json
"""
import argparse
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def prepare_cifar10_test(output_dir: str, download: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print(f"Loading CIFAR-10 test set...")
    testset = torchvision.datasets.CIFAR10(
        root=output_dir,
        train=False,
        download=download,
        transform=transform,
    )

    print(f"Converting {len(testset)} images to PNG files...")
    for idx, (img_tensor, label) in enumerate(testset):
        # img_tensor: [3, 32, 32], values in [0, 1]
        # Permute to [32, 32, 3], scale to [0, 255], cast to uint8
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
        pil_img.save(os.path.join(output_dir, f'{idx}.png'))

        if (idx + 1) % 1000 == 0:
            print(f"  [{idx + 1}/{len(testset)}] done")

    print(f"\nAll images saved to: {os.path.abspath(output_dir)}")
    print(f"Total files: {len(testset)}")
    print(f"\nRun experiments with:")
    print(f"  python run_exp.py --path {os.path.abspath(output_dir)} --conf config/simple.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CIFAR-10 test set as image files')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/hdd/wyt/dataset/cifar10/mytest',
        help='Directory to save CIFAR-10 test images (default: /hdd/wyt/dataset/cifar10/mytest)',
    )
    parser.add_argument(
        '--no_download',
        action='store_true',
        help='Skip downloading; use cached data if available',
    )
    args = parser.parse_args()

    prepare_cifar10_test(args.data_dir, download=not args.no_download)
