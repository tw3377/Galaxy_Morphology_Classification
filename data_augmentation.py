#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:01:54 2024


"""

import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

#%
#%% Importing the dataset 
path = "/Users/tanyawalia/Desktop/Tanya Project/filtered_data.npz"
dataset = np.load(path)

images = dataset['data']
labels = dataset['labels']

#%% Turning dataset into tensors 
images_tensor = torch.tensor(images, dtype = torch.float32).unsqueeze(1)
labels_tensor = torch.tensor(labels)

print(images_tensor.shape)
#%% 
# Assuming x is your original 4D tensor (N samples, C, size, size)
# and y is your class label array (N samples)
x_augmented = []
y_augmented = []

# Find unique classes and their counts
class_counts = Counter(labels_tensor.tolist())
print(class_counts)
max_count = max(class_counts.values())

for cls in class_counts:
    if class_counts[cls] < max_count:
        needed_samples = max_count - class_counts[cls]
        for _ in range(needed_samples):
            img = images_tensor[torch.randperm(len(images_tensor))[0]]  # Pick a random image from the dataset
            rotation = transforms.RandomRotation(degrees=(-15, 15))
            shift = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # 10% of image size
            augmented_img = rotation(img)
            augmented_img = shift(augmented_img)
            x_augmented.append(augmented_img.unsqueeze(0))
            y_augmented.append(torch.tensor([cls]))

# Combine original data with augmented data
x_augmented = torch.cat([images_tensor] + x_augmented)
y_augmented = torch.cat([labels_tensor] + y_augmented)


#%% Saving Filess 
path = "/Users/tanyawalia/Desktop/Tanya Project/" + "classbalanced_data.npz"
np.savez_compressed(path, data = x_augmented, labels = y_augmented)
