Вот изучай пока import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import segmentation

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGES_DIR = "/kaggle/input/supermarket-shelves-dataset/Supermarket shelves/Supermarket shelves/images"
ANNOTATIONS_DIR = "/kaggle/input/supermarket-shelves-dataset/Supermarket shelves/Supermarket shelves/annotations"

TARGET_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 3

def analyze_dataset_distribution(image_files, annotations_dir):
    """Анализ распределения классов по датасету"""
    class_counts = {0: 0, 1: 0, 2: 0}

    for fname in image_files:
        ann = os.path.join(annotations_dir, fname + ".json")
        if not os.path.exists(ann):
            continue

        with open(ann, "r") as f:
            data = json.load(f)

        for obj in data.get("objects", []):
            if obj["classTitle"] == "Product":
                class_counts[1] += 1
            elif obj["classTitle"] == "Price":
                class_counts[2] += 1
            else:
                class_counts[0] += 1

    print("\nDATASET DISTRIBUTION:")
    print(f"  Empty:   {class_counts[0]}")
    print(f"  Product: {class_counts[1]}")
    print(f"  Price:   {class_counts[2]}\n")



def get_train_transform(size):
    return A.Compose([
        A.Resize(height=size[0], width=size[1]),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(size):
    return A.Compose([
        A.Resize(height=size[0], width=size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])



class FixedShelfDataset(Dataset):
    def __init__(self, image_files, images_dir, annotations_dir,
                 target_size=(512, 512), is_training=True):

        self.image_files = image_files
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.is_training = is_training

        self.transform = (
            get_train_transform(target_size)
            if is_training else get_val_transform(target_size)
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        # Load image
        image = cv2.imread(os.path.join(self.images_dir, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load annotation
        with open(os.path.join(self.annotations_dir, image_name + ".json")) as f:
            ann = json.load(f)

        mask = self.create_mask(ann, image.shape[:2])

        transformed = self.transform(image=image, mask=mask)
        return transformed["image"], transformed["mask"].long()

    def create_mask(self, ann, size):
        mask = np.zeros((size[0], size[1]), dtype=np.uint8)

        for obj in ann["objects"]:
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(size[1]-1, x2), min(size[0]-1, y2)

            cls = 1 if obj["classTitle"] == "Product" else