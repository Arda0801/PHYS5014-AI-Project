# Helper funtions for working with fibre bundle holographic focusing dataset.

import pickle

import numpy as np
import matplotlib.pyplot as plt


class FocusData:
    """Class to hold the focus score curves and associated data for a 
    set of holographic images."""
    def __init__(self, scores, depths, metrics, true_depths, image_types, depth_groups):
        self.scores = scores
        self.depths = depths 
        self.metrics = metrics
        self.true_depths = true_depths
        self.image_types = image_types
        self.depth_groups = depth_groups
        self.num_images, self.num_depths, self.num_metrics = np.shape(scores)
             
        self.norm_scores = self.scores - np.min(self.scores, 0)
        self.norm_scores = self.norm_scores / np.max(self.norm_scores, 0)


def load_focus_data(filename, offset = 0, types = None):
    """Load the focus score curves and associated data from a pickle file.
    
    Arguments:
        filename: str
                  The path to the pickle file containing the focus score curves and 
                  associated data.
    Optional Keywords Arguments:
        offset: int
                The number of initial depth values to skip when loading the data. 
                This can be used to remove the initial part of the curves where all 
                metrics peform poorly. (default: 0) 
    
    Returns:
        FocusData: An object containing the focus score curves and associated data.
    """

    with open(filename, "rb") as f:
        data = pickle.load(f)   
    
    # Data is loaded as a dict, pull out the things we need
    scores = data["scores"]              # Array of shape (num_images, num_depths, num_metrics)
    depths = data["metric_depths"] * 1000      # Array of shape (num_depths,) containing the depth values corresponding to the scores
    metrics = data["metrics"]            # A list of metric names corresponding to the last dimension of scores    
    true_depths = data["true_depths"]    # A list of image sample names corresponding to the first dimension of scores
    image_types = data["image_types"]    # A list of image sample names corresponding to the first dimension of scores
    depth_groups = data["depth_groups"]  # 

    # Note we convert depths to mm here for easier interpretation, true depths is already in mm.
    
    # Remove items from scores where all metrics are 0, some datasets have some empty
    # items due to the way folders are parsed
    non_zero_indices = np.where(np.any(scores != 0, axis=(1, 2)))[0]
    scores = scores[non_zero_indices]

    # Remove items where we have specified we don't want them
    if types is not None:
        valid_indices = np.where(np.isin(image_types, types))[0]
        scores = scores[valid_indices]
        true_depths = np.array(true_depths)[valid_indices]
        image_types = np.array(image_types)[valid_indices]
        depth_groups = np.array(depth_groups)[valid_indices]

    # Apply depth offset
    scores = scores[:, offset:]
    depths = depths[offset:]

    # Remove data points that have a true depth that is outside of values in depth_range
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    valid_indices = np.where((true_depths >= min_depth) & (true_depths <= max_depth))[0]
    scores = scores[valid_indices]
    true_depths = np.array(true_depths)[valid_indices]
    image_types = np.array(image_types)[valid_indices]
    depth_groups = np.array(depth_groups)[valid_indices]

    return FocusData(scores, depths, metrics, true_depths, image_types, depth_groups)


def min_pos_error(curve, true_depth, depths):
    """ Calculate the error between the predicted depth (the depth corresponding 
    to the minimum value in the curve) and the ground truth depth.
    
    Arguments:
        curve:      np.ndarray
                    A 1D array of shape (num_depths,) containing the focus scores for a single image 
                    at each numerical refocus depth.
        true_depth: float
                    The ground truth depth value in mm corresponding to the curve.
        depths:     np.ndarray
                    A 1D array of shape (num_depths,) containing the numerical refocus depth values in mm
                    corresponding to the focus scores in curve.
    Returns:
        float: The absolute error between the predicted depth and the ground truth 
             depth in mm.
    """

    
    predicted_depth = best_focus(curve, depths)
    
    # Calculate the error as the absolute difference between the predicted depth and the true depth
    error = abs(predicted_depth - true_depth)
   
    return error    


def rms_error(curves, true_depths, depths):
    """ Calculate the root mean square error between the predicted depths 
    (the depth corresponding to the minimum value in each curve) and the ground
    truth depths.
    
    Arguments:
        curves: np.ndarray
                A 2D array of shape (num_images, num_depths) containing the focus 
                scores for each image at each numerical refocus depth.
        true_depths: np.ndarray
                A 1D array of shape (num_images,) containing the ground truth depth
                values in mm corresponding to each curve.
        depths: np.ndarray
                A 1D array of shape (num_depths,) containing the numerical refocus depth    
                values in mm corresponding to the focus scores in curves.

    Returns:
        float: The root mean square error between the predicted depths and the ground truth depths. 
    """
 
    errors = []

    for curve, true_depth in zip(curves, true_depths):
        error = min_pos_error(curve, true_depth, depths)
        errors.append(error)
    
    rms_error = np.sqrt(np.mean(np.array(errors)**2))
    
    return rms_error


def fraction_correct(curves, true_depths, depths, tol):
    """ Calculate the fraction of curves for which the minimum position is 
    within a certain tolerance of the true depth.
    
    Arguments:
        curve:  np.ndarray
                A 2D array of shape (num_images, num_depths) containing the focus scores 
        true_depths: np.ndarray
                A 1D array of shape (num_images,) containing the ground truth depth 
                values in mm corresponding to each curve.
        depths: np.ndarray
                A 1D array of shape (num_depths,) containing the numerical refocus depth
                values in mm corresponding to the focus scores in curves.
        tol: float
                The tolerance in mm within which the predicted depth must be to the true depth
                focus scores for each image at each numerical refocus depth.

    Returns:
        float: The fraction of curves for which the minimum position is within the specified 
               tolerance of the true depth.
    """    

    correct = 0
    for curve, true_depth in zip(curves, true_depths):
        predicted_depth = depths[np.argmin(curve)]
        if abs(predicted_depth - true_depth) < tol:
            correct += 1

    fraction = correct / len(curves)
    
    return fraction


def show_curves(data, idx, norm=True):
    """ Plot the focus score curves for a single image, along with a vertical line 
    indicating the ground true depth.
    Arguments:
        data: FocusData
            An object containing the focus score curves and associated data.
            idx: int
            The index of the image for which to plot the curves.
        norm: bool
            Whether to normalise the focus scores to be between 0 and 1 across the whole dataset.
            (default: True)  
    """

    if norm:              
        # Normalise each metric to be between 0 and 1 across the whole dataset
        norm_scores = data.scores - np.min(data.scores, 0)
        norm_scores = norm_scores / np.max(norm_scores, 0)

    else:
        norm_scores = data.scores
   
    plt.figure()
    plt.plot(data.depths,  norm_scores[idx])
    plt.xlabel("Depth (mm)")
    plt.ylabel("Focus Score")
    plt.title(f"Example focus score curves for image of {data.image_types[idx]} ")
    # Add a vertical line at the true depth
    plt.axvline(x=data.true_depths[idx], color='r', linestyle='--', label='True Depth')
    labels = data.metrics
    labels.append("True Depth")
    plt.legend(data.metrics)
    plt.show()


def best_focus(curve, depths):
    """ Find the best focus depth (the depth corresponding to the minimum value 
    in the curve) for a single curve.
    
    Arguments:
        curve: np.ndarray
            A 1D array of shape (num_depths,) containing the focus scores for a single image 
            at each numerical refocus depth.
        depths: np.ndarray
            A 1D array of shape (num_depths,) containing the numerical refocus depth 
            values in mm corresponding to the focus scores in curve.

    Returns:
        float: The predicted depth corresponding to the minimum value in the curve. 
    """
    min_index = np.argmin(curve)
    predicted_depth = depths[min_index]
    
    return predicted_depth


def show_curve(data, curve, idx):
    """ Plot a single focus score curve, along with a vertical line
    indicating the ground true depth. The score curve is provided as an argument, 
    while data is used only to extract other parametrs such as the depth value
    for each point.

    Arguments:
        data: FocusData
            An object containing the focus score curves and associated data.
        curve: np.ndarray
            A 1D array of shape (num_depths,) containing the focus scores for a single image 
            at each numerical refocus depth.
        idx: int
            The index of the image for which to plot the curve.
    """
    
    plt.figure()
    plt.plot(data.depths * 1000,  curve[idx])
    plt.xlabel("Depth (mm)")
    plt.ylabel("Focus Score")
    plt.title(f"Example focus score curves for image ({data.image_types[idx]}) ")
    # Add a vertical line at the true depth
    plt.axvline(x=data.true_depths[idx], color='r', linestyle='--', label='True Depth')
    plt.show()