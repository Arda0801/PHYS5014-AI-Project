""" 
A simple example of how to load and view data from the Bundle Holography Refocus
Metrics Dataset
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from helpers import load_focus_data, show_curves


# Load the data
filename = "focus_score_curves_dataset.pkl"
data = load_focus_data(filename)


print(f"Loaded data from {data.num_images} images for {data.num_metrics} metrics applied at {data.num_depths} numerical refocus depths. \n")
print(f"Metrics are: {data.metrics}.")
print(f"Each metric was evaluated at {data.num_depths} depths between {np.min(data.depths) * 1000} and {np.max(data.depths) * 1000} mm. \n")


# Pull out information about a specific image
idx = 17
print(f"Image {idx} is from sample: {data.image_types[idx]}")
print(f"Image {idx} has a ground truth depth of: {data.true_depths[idx]} mm. \n")


# Display an example focus score curve using helper function
idx = 674   # Show data for this image index
show_curves(data, idx)

