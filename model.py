import torch
import torch.nn as nn
import pandas
import numpy as np
import os
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch.utils.data as data


class ImageClassificationCNN():
    def __init__(self, num_class=10):
        super(ImageClassificationCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1)
        )


def main():
    dataset_path = Path("dataset/ai")
    for wnid_folder in dataset_path.iterdir():
        if wnid_folder.is_dir():
            images = list(wnid_folder.glob('*.png'))
            print(f"Found {len(images)} images in folder {wnid_folder.name}")

            train_size = int(len(images) * 0.8)
            val_size = len(images) - train_size
            seed = torch.Generator().manual_seed(42)
            train_images, val_images = torch.utils.data.random_split(images, [train_size, val_size], generator=seed)
            print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")
            # Here you would typically load the images and labels, preprocess them, and create DataLoader instances.

            for img_file in wnid_folder.iterdir():
                if img_file.suffix.lower() in ['.png']:
                    print(f"Processing image: {img_file.name} in folder {wnid_folder.name}")
