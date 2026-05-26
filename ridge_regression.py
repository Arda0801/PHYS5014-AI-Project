#Linear regression model for focus prediction. This is a simpler, more interpretable

# --- Imports ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge      
from sklearn.preprocessing import StandardScaler
from helpers import load_focus_data, fraction_correct

# --- Load data ----------------------------------------------------------------
# Same as the MLP version. offset=20 removes the noisy first 0.2mm of depths.

print("Loading data...")
data = load_focus_data(offset=20)

print(f"  Images:  {data.num_images}")
print(f"  Depths:  {data.num_depths}  ({data.depths[0]:.2f}mm to {data.depths[-1]:.2f}mm)")
print(f"  Metrics: {data.num_metrics} → {data.metrics}")
print(f"  Types:   {np.unique(data.image_types)}")

# --- Baseline performance of individual metrics --------------------------------
# Before training the model, let's see how well each individual metric performs on its own.
# This gives us a baseline to compare against. If the Ridge model can't beat the best
# individual metric, then it's not really learning anything useful from combining them.

print("\n--- Baseline (individual metrics, tol=0.1mm) ---")
tol = 0.1
baseline_scores = {}

for i, metric in enumerate(data.metrics):
    curves = data.scores[:, :, i]
    fc = fraction_correct(curves, data.true_depths, data.depths, tol)
    baseline_scores[metric] = fc
    print(f"  {metric:<20s}: {fc:.3f}")

# ---Train-test split ----------------------------------------------------------------
# Identical to the MLP version — split at image level, stratify by type.

num_images    = data.scores.shape[0]
image_indices = np.arange(num_images)

train_idx, test_idx = train_test_split(
    image_indices,
    test_size=0.2,
    random_state=42,
    stratify=data.image_types
)

train_scores = data.scores[train_idx]     
test_scores  = data.scores[test_idx]       
train_depths = data.true_depths[train_idx]  
test_depths  = data.true_depths[test_idx]

print(f"\nTrain images: {len(train_idx)}, Test images: {len(test_idx)}")

# ---Normalise Inputs----------------------------------------------------------------
# Same normalisation as the MLP version. Each metric is scaled 0→1 within
# each image so that metrics with large absolute values don't dominate.

def normalise_per_image(scores):
    s = scores.copy()
    s_min = s.min(axis=1, keepdims=True)
    s_max = s.max(axis=1, keepdims=True)
    s = (s - s_min) / (s_max - s_min + 1e-8)
    return s

train_scores_norm = normalise_per_image(train_scores)
test_scores_norm  = normalise_per_image(test_scores)

# ---Build Gausian Target Model----------------------------------------------------------------------
# This is the key difference from the MLP version. Because sklearn's Ridge
# regression is a standard supervised learning model, it needs a target value
# (a label) for every single data point.
#
# Each "data point" here is one (image, depth) pair. The input is 7 metric
# values. The target is a number representing "how in-focus is this depth?"
#
# We use a Gaussian centred on the true depth:
#   - At the true depth:      target ≈ 1.0  (very in-focus)
#   - Far from true depth:    target ≈ 0.0  (out of focus)
#
# The model then learns to predict these Gaussian values from the 7 metrics.
# At test time, we find the depth where the model's output is highest — that
# is our focus prediction. (Note: unlike the MLP which predicts LOW=focused,
# here HIGH=focused because the Gaussian peaks at the true depth. We negate
# the output before passing to fraction_correct which expects LOW=focused.)
#
# sigma controls the width of the Gaussian. A smaller sigma (e.g. 0.05) makes
# the target very sharp and precise — only depths very close to the true depth
# get a high label. A larger sigma (e.g. 0.2) is more forgiving and easier to
# learn, but may produce broader, less precise output curves.

def gaussian_target(depths, true_depth, sigma=0.1):
    return np.exp(-0.5 * ((depths - true_depth) / sigma) ** 2)

def build_dataset(scores_norm, true_depths_mm, depths_array, sigma=0.1):
    num_images, num_depths, num_metrics = scores_norm.shape
    X_list = []
    y_list = []

    for i in range(num_images):
        X_list.append(scores_norm[i])                                 
        y_list.append(gaussian_target(depths_array, true_depths_mm[i], sigma))
    X = np.vstack(X_list)  
    y = np.hstack(y_list)   
    return X, y

X_train, y_train = build_dataset(train_scores_norm, train_depths, data.depths, sigma=0.1)
X_test,  y_test  = build_dataset(test_scores_norm,  test_depths,  data.depths, sigma=0.1)

print(f"\nTraining set shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test set shape:     X={X_test.shape},  y={y_test.shape}")

# ---Ridge Regression ----------------------------------------------------------------
# Ridge regression is sensitive to the scale of features even after per-image
# normalisation, because the regularisation penalty treats all features equally.
# StandardScaler rescales each of the 7 features to have mean=0 and std=1
# across all training samples. We fit the scaler ONLY on training data and
# then apply the same transformation to test data — never fit on test data,
# as that would be data leakage.

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled  = scaler.transform(X_test)     

# ---Train the Ridge Regression model----------------------------------------------------------------
# Ridge regression is linear regression with L2 regularisation. This means
# it adds a penalty to the loss function that discourages large weights:
#
#   Loss = mean_squared_error(predictions, targets) + alpha * sum(weights^2)
#
# The alpha parameter controls how strong the regularisation is:
#   - alpha=0:    pure linear regression, no regularisation (can overfit)
#   - alpha=1:    mild regularisation (good default)
#   - alpha=100:  strong regularisation (weights are pushed towards zero)
#
# Regularisation helps when the 7 metrics are correlated with each other
# (which they likely are — most focus metrics respond to similar image
# features). Without it, the model might assign very large positive weight
# to one metric and equally large negative weight to a correlated one,
# which is numerically unstable and generalises poorly.

print("\nTraining Ridge Regression model...")

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

print("Training complete.")
print(f"\nLearned weights (one per metric):")
for metric, weight in zip(data.metrics, model.coef_):
    print(f"  {metric:<20s}: {weight:+.4f}")
print(f"  {'bias':<20s}: {model.intercept_:+.4f}")

# The weights tell you how much each metric contributes to the combined score.
# A large positive weight means that metric strongly indicates focus when high.
# A near-zero weight means that metric is being mostly ignored.

# --- Generate predictions on the test set ----------------------------------------------------------------
# At test time, we run the model on each (image, depth) pair and reshape the
# outputs back into curves of shape (num_images, num_depths).
#
# The model outputs HIGH values where it thinks the image is in focus
# (because the Gaussian target was high at the true depth).
# fraction_correct expects LOW values at the true depth, so we negate.

y_pred = model.predict(X_test_scaled)
predicted_curves_pos = y_pred.reshape(len(test_idx), data.num_depths)

predicted_curves = -predicted_curves_pos

# --- Evaluate performance ----------------------------------------------------------------

fc_model = fraction_correct(predicted_curves, test_depths, data.depths, tol=0.1)
print(f"\n--- Results (tol=0.1mm) ---")
print(f"  Ridge model: {fc_model:.3f}")
for metric, fc in baseline_scores.items():
    print(f"  {metric:<20s}: {fc:.3f}")

print("\n--- Model at multiple tolerances ---")
for t in [0.05, 0.1, 0.2, 0.5]:
    fc = fraction_correct(predicted_curves, test_depths, data.depths, tol=t)
    print(f"  tol={t:.2f}mm: {fc:.3f}")

print("\n--- Per sample type (tol=0.1mm) ---")
for sample_type in np.unique(data.image_types):
    mask = data.image_types[test_idx] == sample_type
    if mask.sum() == 0:
        continue
    fc = fraction_correct(predicted_curves[mask], test_depths[mask], data.depths, tol=0.1)
    print(f"  {sample_type:<20s}: {fc:.3f}  (n={mask.sum()})")

# --- Plots ----------------------------------------------------------------------

# --- Plot 1: Learned Weights Bar Chart --------------------------------------
# This is unique to the linear model — you can directly inspect what the model
# learned. The MLP has hundreds of weights that are hard to interpret; here
# you have exactly 7, one per metric.

plt.figure(figsize=(9, 4))
colours = ['#E07B54' if w > 0 else '#7BAFD4' for w in model.coef_]
plt.bar(data.metrics, model.coef_, color=colours, edgecolor='white')
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel("Weight")
plt.title("Learned Metric Weights (Ridge Regression)\nOrange = positive contribution, Blue = negative")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("plot_linear_weights.png", dpi=150)
plt.show()

# --- Plot 2: Model vs Baseline Bar Chart ------------------------------------

labels  = list(baseline_scores.keys()) + ["Ridge Model"]
values  = list(baseline_scores.values()) + [fc_model]
colours = ["#7BAFD4"] * len(baseline_scores) + ["#E07B54"]

plt.figure(figsize=(10, 5))
plt.bar(labels, values, color=colours, edgecolor='white')
plt.axhline(y=fc_model, color='#E07B54', linestyle='--', linewidth=1.2, label=f"Ridge ({fc_model:.3f})")
plt.ylabel("Fraction Correct")
plt.title("Ridge Model vs Individual Metrics (tol=0.1mm, test set)")
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("plot_linear_baseline_comparison.png", dpi=150)
plt.show()

# --- Plot 3: Per-Sample-Type Performance ------------------------------------

sample_types   = np.unique(data.image_types)
model_per_type = []
for sample_type in sample_types:
    mask = data.image_types[test_idx] == sample_type
    fc = fraction_correct(predicted_curves[mask], test_depths[mask], data.depths, tol=0.1) if mask.sum() > 0 else 0
    model_per_type.append(fc)

plt.figure(figsize=(8, 5))
plt.bar(sample_types, model_per_type, color="#E07B54", edgecolor='white')
plt.ylabel("Fraction Correct")
plt.title("Ridge Model Performance by Sample Type (tol=0.1mm)")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig("plot_linear_per_type.png", dpi=150)
plt.show()

# --- Plot 4: Single Image — Model Curve vs Raw Metrics ----------------------
# Same diagnostic plot as the MLP version. The model output curve (black line)
# should ideally have a clear minimum near the red true-depth line.

example_idx = 0

raw_norm  = test_scores_norm[example_idx]              # (num_depths, 7)
mlp_curve = predicted_curves[example_idx]              # (num_depths,)
mlp_norm  = (mlp_curve - mlp_curve.min()) / (mlp_curve.max() - mlp_curve.min() + 1e-8)

plt.figure(figsize=(10, 5))
for m in range(raw_norm.shape[1]):
    plt.plot(data.depths, raw_norm[:, m], alpha=0.4, linestyle='--', label=data.metrics[m])

plt.plot(data.depths, mlp_norm, color='black', linewidth=2.5, label='Ridge output')
plt.axvline(x=test_depths[example_idx], color='red', linestyle='--', linewidth=1.5, label='True depth')
plt.xlabel("Depth (mm)")
plt.ylabel("Normalised Score")
plt.title(f"Ridge Model vs Raw Metrics — {data.image_types[test_idx[example_idx]]}")
plt.legend(loc='upper right', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_linear_single_image_curve.png", dpi=150)
plt.show()

# --- Plot 5: Gaussian Targets for a Few Training Images ---------------------
# This plot is unique to the linear version and shows what the target labels
# look like — the soft Gaussian curves the model is trained to predict.
# It helps you understand what the model is actually learning to do.

plt.figure(figsize=(10, 4))
for i in range(5):
    true_depth = train_depths[i]
    target = gaussian_target(data.depths, true_depth, sigma=0.1)
    plt.plot(data.depths, target, label=f"True depth: {true_depth:.2f}mm")

plt.xlabel("Depth (mm)")
plt.ylabel("Target Value")
plt.title("Gaussian Target Labels for 5 Training Images\n(model learns to predict these curves)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_linear_gaussian_targets.png", dpi=150)
plt.show()

print("\nAll plots saved.")