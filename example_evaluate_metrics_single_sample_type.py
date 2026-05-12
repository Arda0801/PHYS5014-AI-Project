""" 
A simple example of how to evaluate the individual metrics on individual image types

For each image type, for each metric, calculates the fraction of the images 
for which it correctly predicts the ground truth focus within a tolerance of 0.05 mm.
"""

import pickle

from helpers import load_focus_data, fraction_correct


filename = "focus_score_curves_dataset.pkl"
types = ["paramecium stained", "lillium anther", "usaf_lr", "bee wing", "ipomoea leaf"]
tol = 0.05 # tolerance to count as correct focus position, mm

# Load the data once to get a list of image types
data = load_focus_data(filename, 
                       offset=20,)


for im_type in types:

    print(f"\n-- Sample: {im_type} --")
   
    # Load the data just from this image type
    data = load_focus_data(filename, 
                       offset=20,
                       types=im_type)
                           
    # Evaluate each metric
    for idx, metric in enumerate(data.metrics):
        acc = fraction_correct(data.scores[:,:,idx], data.true_depths, data.depths, tol)
        print(f"{metric:14} : {round(100 * acc)}%")
              
              
          
                                 
    