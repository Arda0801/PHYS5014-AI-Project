# PHYS5014-AI-Project
Combining Focus Metrics in Inline Holographic Microscopy with Machine Learning

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

---

## Background

In holographic microscopy, we don't generally know the distance of the object from the camera (or in our case, the fibre bundle), so we need to perform trial numerical refocusing of the image at different distances (depths) and evaluate the quality of the images until we find the best one. This evaluation requires some metric, a calculation we perform on the image that tells us how well-focused it is, giving us a score. By applying focus metrics to the reconstructed images at different distances, we can determine which reconstruction is the best and thus find the correct distance to the object.

Focus metrics are tricky to apply to inline holographic microscopy. This is partly because the reconstructed images contain artefacts, but also because the interference pattern can result in seemingly sharp image features even when the reconstruction is not correct. Various metrics to assess the focus of the reconstructed images have been proposed, but they often fail to provide a clear indication of the best reconstruction. In this project, we explore whether machine learning can be used to combine the outputs of multiple focus metrics to obtain a single metric that works better than any of the individual ones.

Note that we are not learning to predict the reconstruction distance directly, but rather learning a combination of focus metrics that could then be used to find the best reconstruction either by brute force search or more sophisticated gradient-descent algorithms. In practice this means that the combined metric should give a low value when the image is in-focus, and higher values when it is out-of-focus.

---

## The Task

Development of a machine learning model that takes 7 different focus metric outputs as its input and produces a single combined metric as a measure to determine the best focus distance from the sample.

---

## Why Not Standard Linear Regression?

The first attempt at this project used standard linear regression, available in earlier versions of this repository. This approach did not perform well because the seven focus metrics have very different value ranges and are correlated with one another  metrics like Sobel variance, Brenner, and Tenengrad all measure image sharpness using slightly different formulas and therefore tend to rise and fall together. Standard linear regression minimises the mean squared error between predictions and targets without any constraint on the size of the learned weights. When input features are correlated, this leads to numerical instability: the model may assign an extremely large positive weight to one metric and an equally large negative weight to a correlated one, effectively cancelling them out in a way that does not generalise to new images.

Ridge regression addresses this by adding a penalty term to the loss function that discourages large weights. Instead of minimising prediction error alone, it minimises prediction error plus alpha times the sum of squared weights. This forces the model to spread weight more evenly across correlated features rather than assigning extreme values that cancel out. The result is a model that generalises far better to unseen images. With seven metrics that measure overlapping properties of the same image, Ridge regression is almost always the right choice over plain linear regression, producing weights that are more trustworthy and stable on new data.

---

## Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) was also explored as a more powerful alternative to Ridge regression. An MLP is a neural network that works by passing the 7 metric inputs through an artificial set of "neurons". Each neuron takes all the numbers from the previous layer, multiplies each by a learned weight, adds them all up, then passes the result through an activation function (in this case, ReLU). ReLU simply sets any negative values to zero: max(0, x). This sounds trivial, but stacking many layers of these neurons with ReLU activations allows the network to approximate almost any mathematical function. The weights are what get learned during training.

The reason an MLP is appropriate here  rather than a convolutional neural network (CNN) or recurrent network (RNN)  is that the model sees each depth independently. The input is simply seven numbers at a single depth, with no spatial or sequential structure. MLPs are specifically designed for this kind of fixed-size numerical input.

---

## Use of Ridge Regression to Combine Focus Metrics in Inline Holographic Microscopy

### 1. The Big Picture

The physical problem is autofocus in inline holographic microscopy: given a hologram of an unknown sample, find the correct numerical refocus distance. Seven different focus metrics are computed at 200 depth values for each of 750 holograms spanning five biological and optical sample types. Each metric provides a partial, noisy signal about which depth is in focus, and no single metric reliably identifies the correct depth across all sample types and conditions.

Success is defined using the `fraction_correct` metric: the proportion of test images for which the model's predicted best-focus depth falls within a tolerance window of the manually-determined ground truth depth. The primary tolerance used is 0.1 mm, which corresponds to ten depth steps of 10 microns each. A secondary evaluation is also performed at tolerances of 0.05, 0.2, and 0.5 mm to give a fuller picture of model precision. The model is considered successful if it achieves a higher `fraction_correct` than the best individual metric across the test set.

### 2. Get Data

The dataset is provided as a Python pickle file (`focus_score_curves_dataset.pkl`) and is loaded using the `load_focus_data` helper function from `helpers.py`. The dataset contains:

- **scores**: an array of shape (750 images, 200 depths, 7 metrics) containing the raw focus metric values
- **depths**: the 200 numerical refocus distances in millimetres, spanning 0 to 2 mm in 10 micron steps
- **true_depths**: the manually-determined ground truth focus depth for each image (in mm)
- **image_types**: the sample type label for each image (paramecium, lilium, bee wing, ipomoea, or USAF target)
- **metrics**: the names of the seven focus metrics

A pristine test set is isolated before any preprocessing or model fitting. The data is split at the image level and all 200 depth values for a given image go entirely into either the training set or the test set, never both. This is critical: splitting at the depth level would allow the same image to appear in both sets, causing data leakage and falsely inflating test performance. 80% of images are used for training (600 images) and 20% for testing (150 images). Stratification by `image_types` ensures each sample type is proportionally represented in both sets, so the test set fairly represents all conditions the model will encounter.

```python
train_idx, test_idx = train_test_split(
    image_indices, test_size=0.2, random_state=42, stratify=data.image_types
)
```

### 3. Explore

Before any model is trained, a baseline is established by evaluating each of the seven raw metrics individually using `fraction_correct` at tol=0.1mm. This reveals which metrics are already most informative and sets a concrete performance target for the combined model to beat.

Focus score curves for individual images are visualised using the `show_curves` helper function, which plots all seven metric curves against depth for a single image and marks the ground truth depth with a vertical line. This visualisation reveals several important properties of the data:

- Metrics tend to be noisy at short distances (below ~0.2 mm) because holographic reconstruction is unreliable at close range
- Different sample types produce very different curve shapes. The USAF resolution target has clean, well-defined minima while biological samples like paramecium or bee wing produce noisier, flatter curves
- Multiple metrics often have local minima at or near the true depth, suggesting that a combination should outperform any individual metric

These observations directly motivate both the preprocessing decisions (depth offset) and the choice of a combined model.

### 4. Prepare

**Depth offset:** The first 20 depth values (covering 0 to 0.2 mm) are removed by passing `offset=20` to `load_focus_data`. In this region, all metrics produce noisy, unreliable outputs because holograms do not reconstruct well at very short distances. Removing this region improves both training stability and evaluation accuracy. This reduces the depth range from 200 to 180 values per image.

**Per-image normalisation:** Each of the seven metrics is independently normalised to the range [0, 1] within each image:

```python
s_min = scores.min(axis=1, keepdims=True)   # minimum over depth axis
s_max = scores.max(axis=1, keepdims=True)   # maximum over depth axis
scores_norm = (scores - s_min) / (s_max - s_min + 1e-8)
```

This is done per-image rather than globally because the absolute values of metrics vary considerably between images depending on sample brightness and contrast. Per-image normalisation ensures the model responds to the *shape* of the curve where the metric is low or high relative to its own range, rather than its absolute magnitude. The small epsilon (1e-8) prevents division by zero in edge cases.

**Gaussian target labels:** Ridge regression requires a scalar target value for every (image, depth) data point. A Gaussian-shaped soft label is constructed for each image, centred on its true depth:

```
target(d) = exp(-0.5 * ((d - true_depth) / sigma)^2)
```

This assigns a target value of 1.0 at the true focus depth, falling smoothly to 0.0 at depths far from focus. A sigma of 0.1 mm is used, which is wide enough for the model to learn from neighbouring depths but narrow enough to produce a precise minimum. The result is a well-posed regression problem: the model learns to predict high values near focus and low values far from it.

**StandardScaler:** After per-image normalisation, a `StandardScaler` from sklearn is applied to standardise each of the seven features to zero mean and unit variance across the training set. This is important for Ridge regression because the L2 penalty treats all features equally without standardisation, a feature with a slightly larger scale would be penalised more heavily than intended. The scaler is fit only on the training set and applied (without refitting) to the test set to prevent data leakage.

**Dataset construction:** The data is flattened from shape (N_images, 180 depths, 7 metrics) into shape (N_images × 180, 7) for sklearn compatibility, with corresponding Gaussian targets of shape (N_images × 180,). This produces approximately 108,000 training samples.

### 5. Train

A Ridge regression model is trained using sklearn's `Ridge` class with regularisation strength `alpha=1.0`:

```python
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)
```

Ridge regression finds the weight vector **w** and bias *b* that minimise:

```
Loss = MSE(y_pred, y_true) + alpha * ||w||^2
```

The first term is the standard mean squared error between the predicted Gaussian values and the true Gaussian targets. The second term is the L2 penalty: alpha times the sum of squared weights. This penalty discourages any single metric from receiving a disproportionately large weight, which is important because the seven metrics are correlated.

Unlike the MLP, Ridge regression has a closed-form solution. Sklearn solves it in a single step using linear algebra rather than iterative gradient descent. This makes training near-instantaneous and the result completely deterministic. The trained model produces seven weights (one per metric) and a bias, which can be directly inspected to understand which metrics the model found most informative.

### 6. Fine-Tune

The key tuning parameter is `alpha`, the regularisation strength. A larger alpha pushes all weights further toward zero, producing a simpler, more conservative model that is less likely to overfit but may underfit if set too high. A smaller alpha allows larger weights and more expressive fits, but risks instability when metrics are correlated.

To find the best alpha, a search over a logarithmic grid is recommended:

```python
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    # evaluate fraction_correct on validation set
```

The `sigma` parameter of the Gaussian target (default 0.1 mm) is a second tuning variable. A smaller sigma (e.g. 0.05 mm) creates sharper, more demanding targets — only depths very close to true focus receive high labels. A larger sigma (e.g. 0.2 mm) is more forgiving but may produce broader output curves with less precise predicted minima. Both alpha and sigma should be tuned using a held-out validation set carved from the training data, not the final test set.

The `offset` parameter (default 20) can also be varied to investigate how much of the noisy short-distance region needs to be removed. Values between 10 and 30 are reasonable to explore.

### 7. Present

**Learned weights:** The most interpretable output of the Ridge model is its seven learned weights, visualised as a bar chart. A large positive weight means that metric's value being high is associated with the image being in focus. A negative or near-zero weight means the model found that metric unreliable or redundant given the others. This directly tells you something about the physics of the problem; which mathematical properties of a holographic image genuinely correlate with focus quality across different sample types.

**Comparison with baseline:** A bar chart comparing `fraction_correct` for each individual metric alongside the Ridge model on the test set shows whether the combination adds value. The model is only useful if it clearly exceeds the best individual metric.

**Per-sample-type breakdown:** Performance is reported separately for each of the five sample types. The USAF resolution target typically scores highest because it has clean, sharp features. Biological samples are more challenging. If one sample type scores significantly lower than the others, it suggests that a separate model trained only on that type might be warranted.

**Single-image curve inspection:** For individual test images, the model's output curve (negated Gaussian prediction) is plotted alongside all seven raw normalised metric curves and the true depth line. A good result shows the model producing a clean, unambiguous minimum near the true depth even when the raw metrics are noisy or contradictory. This is the most diagnostic visualisation available.

**Limitations:** The model is a linear combination of the seven metrics and cannot capture non-linear interactions between them. It is also trained on images from five specific sample types captured with one particular fibre-bundle holographic microscope; performance on different instruments or new sample types may be lower. The ground truth true depths were determined manually and carry some inherent subjectivity, particularly for three-dimensional samples where no single plane is perfectly in focus. Accuracy is therefore bounded by the quality of the ground truth labels.

### 8. Launch

To use the trained model on new holographic data, the same preprocessing pipeline must be applied in the same order: remove the initial 20 depth values, apply per-image normalisation, apply the same `StandardScaler` (already fit on training data), and then call `model.predict()`. The output is then negated so that the minimum corresponds to the predicted best-focus depth.

```python
# Preprocessing new data
new_scores_norm = normalise_per_image(new_scores[:, 20:, :])
new_scores_flat = new_scores_norm.reshape(-1, 7)
new_scores_scaled = scaler.transform(new_scores_flat)   # use the already-fitted scaler

# Predict and find best focus depth
predictions = model.predict(new_scores_scaled).reshape(num_new_images, -1)
best_focus_indices = np.argmax(predictions, axis=1)   # highest prediction = best focus
best_focus_depths = data.depths[best_focus_indices]
```

For continuous monitoring, `fraction_correct` should be evaluated periodically on new labelled images as they become available. If performance degrades (for example when imaging a new sample type or after instrument recalibration) the model should be retrained on an expanded dataset that includes the new data. Because Ridge regression trains in milliseconds, retraining is cheap. The `alpha` and `sigma` hyperparameters should be re-validated whenever the training set changes substantially, as the optimal values may shift with a different data distribution.