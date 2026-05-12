
## Introduction

Holographic microscopy is an optical imaging technique that allows us to capture three-dimensional information about microscopic objects. Instead of imaging using a lens, we illuminate the sample with a laser and then collect an interference pattern on a camera placed on the other side of the sample, with no lens in between. We then use computer software to numerically propagate the interference pattern back to the plane of the sample, allowing us to calculate an in-focus image.

At Kent we have developed a technique for collecting holograms through optical fibre bundles, which allows us to perform holographic microscopy in hard-to-reach places. You don't need to know the details of how this works to complete this mini-project, but if you are interested you can read this paper: https://doi.org/10.1364/BOE.516030. 

## The Focus Distance Problem

In holographic microscopy, we don't generally know the distance of the object from the camera (or in our case, the fibre bundle), so we need to perform trial numerical refocusing of the image at different distances (depths) and evaluate the quality of the images until we find the best one. This evaluation requires some metric - a calculation we perform on the image - that tells us how well-focused it is, giving us a score. By applying focus metrics to the reconstructed images at different distances, we can determine which reconstruction is the best and thus find the correct distance to the object. 

Focus metrics are tricky to apply to inline holographic microscopy. This is partly because the reconstructed images contain artefacts, but also because the interference pattern can result in seemingly sharp image features even when the reconstruction is not correct. Various metrics to assess the focus of the reconstructed images have been proposed, but they often fail to provide a clear indication of the best reconstruction. In this mini-project, you will explore whether we can use machine learning to combine the outputs of multiple focus metrics to obtain a single metric that works better than any of the individual ones.

Note that you are *not* learning to predict the reconstruction distance directly, but rather to predict a combination of focus metrics that could then be used to find the best reconstruction either by brute force search or more sophisticated gradient-descent algorithms. You are **not** being asked to work on how to find the best reconstruction, only to learn a combination of the focus metrics that provides a better indication of how in-focus the image is. In practice this means that the metric should give a low value when the image is in-focus, and higher values when it is out-of-focus.

## What you will do

Your task is to develop a machine learning model that takes the outputs of seven focus metrics as its input and produces a single output that serves as a better measure of how in-focus the image is. 

## What you will be given

You will work with a dataset that provides the output of 7 different focus metrics for 750 holograms, each reconstructed at 200 different refocus distances (depths) in 10 micron steps over a range of 0 to 2 mm. (So you have 900,000 data points to work with).

The dataset also includes the manually-determined best refocus distance for each hologram, which serves as the ground truth for evaluating the performance of the focus metrics. However, this is dependent on the judgement of the person who did the manual focusing; we would never expect it to exactly match the output of an algorithm, particularly as the samples have some 3D structure, so we cannot have all of them in focus at the same time.

The holograms are from an internal dataset that was captured here at Kent using our fibre bundle holographic microscope. The effect of the fibre bundle was removed using the PyFibreBundle Python package (https://pyfibrebundle.readthedocs.io), and the numerical refocusing and metric scoring was performed using the PyHoloscope Python package (https://pyholoscope.readthedocs.io), both developed at the University of Kent. You don't need to know about or use the packages to complete this project, but if your project is successful, your approach could later even be integrated into the PyHoloscope package to help users of the package find the best reconstruction of their holograms.

The images were taken from five different samples:
* Paramecium (a single-celled organism) - 150 holograms
* Lilium anther (the pollen-producing part of a flower) - 150 holograms
* Bee wing - 150 holograms
* Ipomoea leaf (a type of plant) - 150 holograms
* USAF resolution target (a standard test target used in microscopy) - 150 holograms

The metrics are:
- sum
- peak
- sobel
- sobel_variance
- brenner
- dark_focus
- norm_var

You don't really need to know how they work, but if you are interested you can find out more about them here: https://pyholoscope.readthedocs.io/en/latest/autofocus.html#focus-metrics

The metrics work better with some of these samples than others, so you may want to consider whether to train a single model for all of the data, or separate models for each sample type or some combination of the two.

The dataset is stored as a `.pkl` file, which is a Python pickle file - a collection of data. To help you work with the dataset, some helper functions are provided in the `helpers.py` file.

There are some example scripts to show you how to use these functions. You can also read the docstrings in the `helpers.py` file and the functions themselves to understand what they do. The most important helper functions are:
* `load_focus_data`: This function loads the focus metric data from the dataset and returns it in a convenient way for your work. You can specify which sample type(s) to load, and you can also specify an offset to remove the data at small distances if you want to.
* `fraction_correct`: This function evaluates the performance of your model by comparing the predicted best focus distance with the known best focus distance for each hologram, and computes the percentage within a certain tolerance. You can use this to evaluate your solution.
* `show_curves`: Can be used to plot the focus score value(s) at different depths.

For your interest, below are some example images in Fig. 1 (which you will not have) and example plots of focus scores in Fig 2. (which you have and can generate).

![[examples.png]]
*Fig 1. Example images. For the USAF target, the top row shows a raw hologram, a hologram with the fibre pattern removed, a hologram refocused to the wrong depth, and a hologram refocused to the correct depth. The bottom row shows correctly refocused examples from the other four samples. (Note that the refocused images have the brightness inverted).*



![[example_curve.png]]
*Fig 2. Example focus scores with refocus depth for one of the USAF resolution target images. Most of the metrics have at least local minima at the ground truth depth, as this is an easy object to identify the focus for. Here the focus scores were all normalised to lie between 0 and 1 across the entire dataset.

## Challenges

There are several potential trip-points in this project that you should think carefully about.

1. **Data Preprocessing**: The holograms do not refocus well over short distances of up to around 0.2 mm. All of the focus metrics therefore produce very noisy outputs in this region. A simple solution is to only work with the data at larger distances. You can do this simply by providing a value to the `offset` argument of the `load_focus_data` function in the `helpers.py` file, which will remove the data at small distances. For example, `offset = 10` removes the first 10 depths, and since the depth interval is 10 microns, this removes the first 0.1 mm. The metrics also have very different value ranges, although linear regression can handle this, other approaches may require some normalisation.

2. **Sample Types**: The different sample types have different characteristics, and the focus metrics work better for some than others. You may want to consider whether to train a single model for all of the data, or separate models for each sample type or some combination of them. The former is more general, but the latter will give better performance.

3. **Model Selection**: There are many different machine learning algorithms that you could use for this problem. You can experiment with different approaches, but start with something simple like linear regression or Ridge.

4. **Evaluation Metrics**: An evaluation metric is provided in the `helpers.py` to assess the performance of your model. This simply finds the point that your model predicts to be the best focus (i.e. the lowest value of the metric for each image) and compares this with the known depth for each image, and computes the RMS error as well as the percentage  of the images where the estimated depth is within some tolerance of the true depth. An example of applying this to the individual focus metrics is provided.

5. **Loss Function**: You will need to choose an appropriate loss function for training your model. You cannot use the simple evaluation metric suggested above, because this is not differentiable (the minimum value of the metric across all distances will jump about in a way that cannot be used with gradient descent). You need something you can apply to the output of the model at each distance. You want to train your model to give a low value at the correct distance and higher values at other distances, but you need to find a way to express this as a loss function that can be used for training.

6. **Testing and Validation**: As always, you should split your dataset into training and test sets to evaluate the performance of your model. Think carefully about how to do this, because the data is derived from 750 different images, each with 200 different depths. You should split the data at the level of the images, so that all of the refocus depths for a given image are either in the training set or the testing set.

