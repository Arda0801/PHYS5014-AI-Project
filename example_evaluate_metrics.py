""" 
A simple example of how to evaluate the individual metrics. 

For each metric, calculates the fraction of the images for which it correctly predicts
the ground truth focus within a tolerances of 0.025, 0.050 and 0.075 mm.
"""

import pickle

from helpers import load_focus_data, fraction_correct, rms_error


# Load the data
filename = "focus_score_curves_dataset.pkl"
data = load_focus_data(filename, 
                       offset=20,)
                       

for idx, metric in enumerate(data.metrics):
    print(f"\n--- Metric: {metric} ---")

    rms = rms_error(data.scores[:,:,idx], data.true_depths, data.depths)
    print(f"RMS Error: {1000 * rms:.0f} um")
                    
    for tol in [0.025, 0.050, 0.075]:
        acc = fraction_correct(data.scores[:,:,idx], data.true_depths, data.depths, tol)
          
        print(f"Within {tol * 1000} um: {round(100 * acc)}%") 
      
                             
    