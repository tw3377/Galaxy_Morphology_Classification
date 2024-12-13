#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 12:53:33 2024
"""

# Data Processing Script 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
from PIL import Image

#%% Helper Functions 

def is_uniform_image(image):
    return np.all(image == image[0, 0])

def filter_dataset(X, y):
    filtered_X = []
    filtered_y = []
    for img, label in zip(X, y):
        if not is_uniform_image(img):
            filtered_X.append(img)
            filtered_y.append(label)
    return np.array(filtered_X), np.array(filtered_y)

#%% 
# Callling Data Files  

image_dir = '/Users/tanyawalia/Desktop/Tanya Project/png_images'
catalog = pd.read_csv('/Users/tanyawalia/Desktop/Tanya Project/catalog.csv')

images = []
labels = []

for _, row in catalog.iterrows():
    img_path = os.path.join(image_dir, row['ID'])
    try:
        img = Image.open(img_path)
        images.append(np.array(img) / 255.0)
        labels.append(row['Encoded Class'])
    except FileNotFoundError:
        print(f"File not found and skipped: {img_path}")
    except Exception as e:
        print(f"Error loading file {img_path}: {e}")

X = np.array(images)
y = np.array(labels)

#%% Code

skipped_indices = []
for i, img in enumerate(X):
    if is_uniform_image(img):
        skipped_indices.append(i)
        
        
#%% Define Filtered out image set k
mask = np.ones(X.shape[0], dtype = bool)
mask[skipped_indices] = False

#%% 
filtered_X = X[mask, :, :]
filtered_y = y[mask]

#%% Saving the Data as a npz array 

# Create a new folder 
path = "/Users/tanyawalia/Desktop/Tanya Project/" + "filtered_data.npz"
np.savez_compressed(path, data = filtered_X, labels = filtered_y)




