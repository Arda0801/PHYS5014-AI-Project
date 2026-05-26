# PHYS5014-AI-Project
Combining Focus Metrics in Inline Holographic Microscopy with machine learning

This project was created for the PHYS5014 Machine Learning for Natural Sciences module.

## How To Run

### Clone the repository
```bash
git clone https://github.com/Arda0801/PHYS5014-AI-Project
cd PHYS5014-AI-Project
```
### Run the Project
```
python MLP.py
```
or
```
python ridge_regression.py
```

## Background

In holographic microscopy, we don't generally know the distance of the object from the camera (or in our case, the fibre bundle), so we need to perform trial numerical refocusing of the image at different distances (depths) and evaluate the quality of the images until we find the best one. This evaluation requires some metric - a calculation we perform on the image - that tells us how well-focused it is, giving us a score. By applying focus metrics to the reconstructed images at different distances, we can determine which reconstruction is the best and thus find the correct distance to the object. Focus metrics are tricky to apply to inline holographic microscopy. This is partly because th reconstructed images contain artefacts, but also because the interference pattern can result in seemingly sharp image features even when the reconstruction is not correct. Various metrics to assess the focus of the reconstructed images have been proposed, but they often fail to provide a clear indication of the best reconstruction. In this mini-project, you will explore whether we can use machine learning to combine the outputs of multiple focus metrics to obtain a single metric that works better than any of the individual ones. Note that you are not learning to predict the reconstruction distance directly, but rather to predict a combination of focus metrics that could then be used to find the best reconstruction either by brute force search or more sophisticated gradient-descent algorithms. You are not being asked to work on how to find the best reconstruction, only to learn a combination of the focus metrics that provides a better indication of how in-focus the image is. In practice this means that the metric should give a low value when the image is in-focus, and higher values when it is out-of-focus.

## The Task

Development of a machine learning model that will take 7 different metric outputs as its input and outputs one single metric as a measuring system to determine the best distance from the sample for focus.

## Why not Standard Linear Regression?

First attempt at this, which is available at earlier versions of this Github page was with a simple linear regression. This attempt did not work as some of the outputs have different "weights" to them and standard linear regression tries to minimise  the mean squared error between the predictions and the targets.