#Load different libraries

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

#Load the data from helpers.py
from helpers import load_focus_data, show_curves, fraction_correct

# Load the data and skip the noisy 0-0.2mm region
data = load_focus_data(offset=20)

#Establishing a baseline for the model with 0.1mm tolerance
tol = 0.1
for i, metric in enumerate(data.metrics):
    curves = data.scores[:, :, i]
    fc = fraction_correct(curves, data.true_depths, data.depths, tol)

#For this model, I will be using a margin/ranking loss function, which is commonly used for learning to rank problems. The idea is to train the model to assign higher scores to the correct focus depths compared to the incorrect ones, with a margin that encourages a certain level of separation between the scores.
def focus_loss(outputs, true_depth_indices, margin=0.1):
    """
    outputs: shape (num_images, num_depths)
    true_depth_indices: shape (num_images,) — index of true depth for each image
    """
    loss = 0
    for i in range(outputs.shape[0]):
        true_score = outputs[i, true_depth_indices[i]]
        all_scores = outputs[i]
        # Penalise wherever true_score is not lower than others by margin
        loss += torch.mean(torch.clamp(margin + true_score - all_scores, min=0))
    return loss / outputs.shape[0]

#Splitting of the test data and training data. For this, I will be using the sklearn train_test_split function, which is a convenient way to split the data into training and testing sets. I will use 80% of the data for training and 20% for testing.
num_images = data.scores.shape[0]
image_indices = np.arange(num_images)

train_idx, test_idx = train_test_split(image_indices, test_size=0.2, random_state=42, stratify=data.image_types)
#Stratisfy is used to keep the same distribution of image types in both training and testing sets, which is important for ensuring that the model learns to generalise well across different types of images.

train_scores    = data.scores[train_idx]
test_scores     = data.scores[test_idx]
train_depths    = data.true_depths[train_idx]
test_depths     = data.true_depths[test_idx]