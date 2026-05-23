# --- Imports ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from helpers import load_focus_data, fraction_correct

# - - Load Data ----------------------------------------------------------------
# offset=20 skips the first 20 depth values (0-0.2mm), which are noisy because

print("Loading data...")
data = load_focus_data(offset=20)

print(f"  Images:  {data.num_images}")
print(f"  Depths:  {data.num_depths}  ({data.depths[0]:.2f}mm to {data.depths[-1]:.2f}mm)")
print(f"  Metrics: {data.num_metrics} → {data.metrics}")
print(f"  Types:   {np.unique(data.image_types)}")

# - - Baseline Performance ------------------------------------------------------
# This checks how well each of the 7 metrics performs on its own. This sets the 
# target metric to beat. The model should learn to combine the strengths of all
# metrics to outperform. fraction_correct() finds the depth where each metric 
# is lowest (its prediction) and checks whether it falls within `tol` mm of the 
# true depth.

print("\n--- Baseline (individual metrics, tol=0.1mm) ---")
tol = 0.1
baseline_scores = {}

for i, metric in enumerate(data.metrics):
    curves = data.scores[:, :, i]
    fc = fraction_correct(curves, data.true_depths, data.depths, tol)
    baseline_scores[metric] = fc
    print(f"  {metric:<20s}: {fc:.3f}")


# -- - Train/Test Split ---------------------------------------------------------
# IMPORTANT: We split at the IMAGE level, not the depth level. If we split by
# depth, the same image could appear in both train and test set, leaking information.
# stratify=data.image_types ensures each sample type (bee wing, paramecium etc.)
# is proportionally represented in both sets.

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

# -- - Normalisation -------------------------------------------------------------
# Each metric has a very different value range. We normalise per-image so that
# every metric is scaled 0→1 for that image. This stops metrics with large
# absolute values from dominating, and helps the model generalise across images
# with different brightness/contrast.

def normalise_per_image(scores):
    s = scores.copy()
    s_min = s.min(axis=1, keepdims=True)
    s_max = s.max(axis=1, keepdims=True)
    s = (s - s_min) / (s_max - s_min + 1e-8) #add small epsilon to avoid division by zero
    return s

train_scores_norm = normalise_per_image(train_scores)
test_scores_norm  = normalise_per_image(test_scores)

# Convert to PyTorch tensors for use with the neural network
X_train = torch.tensor(train_scores_norm, dtype=torch.float32)
X_test  = torch.tensor(test_scores_norm,  dtype=torch.float32)

# -- - True Depth Indices for Loss Function ------------------------------------------------
# The loss function needs to know *which index* in the depths array corresponds
# to the true depth for each image (not the mm value itself).

def get_true_depth_indices(true_depths_mm, depths_array):
    indices = [np.argmin(np.abs(depths_array - td)) for td in true_depths_mm]
    return np.array(indices)

train_depth_indices = get_true_depth_indices(train_depths, data.depths)
test_depth_indices  = get_true_depth_indices(test_depths,  data.depths)

# Convert to tensor for use in loss function
train_idx_tensor = torch.tensor(train_depth_indices, dtype=torch.long)

# - - Loss Function: Margin Ranking Loss ------------------------------------------------
# We want the model to output a LOW score at the true focus depth and HIGH
# scores everywhere else. This is a ranking/ordering problem.
#
# For each image, for every depth d that is NOT the true depth, we penalise
# the model if:
#     model_output[true_depth] >= model_output[d] - margin
# i.e. the true depth score is not sufficiently lower than other scores.
#
# torch.clamp(..., min=0) acts like max(0, ...) — only penalises violations.
# margin controls how much separation we demand between true and false depths.

def focus_loss(outputs, true_depth_indices, margin=0.2):
    total_loss = torch.tensor(0.0, requires_grad=True)

    for i in range(outputs.shape[0]):
        true_score = outputs[i, true_depth_indices[i]]   # scalar
        all_scores = outputs[i]                           # (num_depths,)

        # Penalise: margin + true_score - other_score > 0 means violation
        penalties = torch.clamp(margin + true_score - all_scores, min=0.0)
        total_loss = total_loss + torch.mean(penalties)

    return total_loss / outputs.shape[0]

# - - - Neural Network Definition ---------------------------------------------------------------
# Simple MLP (Multi-Layer Perceptron):
#   Input:  7 metric values at a single depth
#   Output: 1 focus score for that depth
#
# The network is applied independently to each depth, then the loss is computed
# across all depths for each image. This makes the model depth-agnostic — it
# only sees the metric values, not the depth itself.

class FocusMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),    # 7 metrics → 32 hidden units
            nn.ReLU(),
            nn.Linear(32, 16),   # 32 → 16 hidden units
            nn.ReLU(),
            nn.Linear(16, 1)     # 16 → 1 output score
        )

    def forward(self, x):
        # x shape: (num_images, num_depths, 7)
        # self.net applied to last dim → (num_images, num_depths, 1)
        # squeeze(-1) removes the trailing 1 → (num_images, num_depths)
        return self.net(x).squeeze(-1)

# - - - Training Loop ---------------------------------------------------------------

model     = FocusMLP()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 150
losses     = []

print("\nTraining MLP...")
for epoch in range(num_epochs):
    model.train()
    optimiser.zero_grad()               # clear gradients from last step

    outputs = model(X_train)            # forward pass → (N_train, num_depths)
    loss    = focus_loss(outputs, train_idx_tensor, margin=0.2)

    loss.backward()                     # compute gradients
    optimiser.step()                    # update weights

    losses.append(loss.item())

    if epoch % 15 == 0:
        print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {loss.item():.4f}")

print("Training complete.")

# - - - Evaluation ----------------------------------------------------------------

model.eval()
with torch.no_grad():
    predicted_curves = model(X_test).numpy()

fc_model = fraction_correct(predicted_curves, test_depths, data.depths, tol=0.1)
print(f"\n--- Results (tol=0.1mm) ---")
print(f"  MLP model: {fc_model:.3f}")
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

# - - - Plots ----------------------------------------------------------------------

# --- Plot 1: Training Loss Curve -------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(losses, color='steelblue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_training_loss.png", dpi=150)
plt.show()

# --- Plot 2: Model vs Baseline Bar Chart ------------------------------------
labels = list(baseline_scores.keys()) + ["MLP Model"]
values = list(baseline_scores.values()) + [fc_model]
colours = ["#7BAFD4"] * len(baseline_scores) + ["#E07B54"]

plt.figure(figsize=(10, 5))
bars = plt.bar(labels, values, color=colours, edgecolor='white')
plt.axhline(y=fc_model, color='#E07B54', linestyle='--', linewidth=1.2, label=f"Model ({fc_model:.3f})")
plt.ylabel("Fraction Correct")
plt.title("MLP Model vs Individual Metrics (tol=0.1mm, test set)")
plt.xticks(rotation=30, ha='right')
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.savefig("plot_baseline_comparison.png", dpi=150)
plt.show()

# --- Plot 3: Per-Sample-Type Performance ------------------------------------
sample_types  = np.unique(data.image_types)
model_per_type = []
for sample_type in sample_types:
    mask = data.image_types[test_idx] == sample_type
    fc = fraction_correct(predicted_curves[mask], test_depths[mask], data.depths, tol=0.1) if mask.sum() > 0 else 0
    model_per_type.append(fc)

plt.figure(figsize=(8, 5))
plt.bar(sample_types, model_per_type, color="#E07B54", edgecolor='white')
plt.ylabel("Fraction Correct")
plt.title("MLP Model Performance by Sample Type (tol=0.1mm)")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig("plot_per_type.png", dpi=150)
plt.show()

# --- Plot 4: Single Image — Model Curve vs Raw Metrics ----------------------
# Pick one test image and overlay all raw metrics alongside the model output.
# This is the most informative diagnostic plot — you can see whether the model
# produces a cleaner, sharper minimum near the true depth.

example_idx = 0

raw_norm = test_scores_norm[example_idx]
mlp_curve = predicted_curves[example_idx]
mlp_norm  = (mlp_curve - mlp_curve.min()) / (mlp_curve.max() - mlp_curve.min() + 1e-8)

plt.figure(figsize=(10, 5))
for m in range(raw_norm.shape[1]):
    plt.plot(data.depths, raw_norm[:, m], alpha=0.4, linestyle='--', label=data.metrics[m])

plt.plot(data.depths, mlp_norm, color='black', linewidth=2.5, label='MLP output')
plt.axvline(x=test_depths[example_idx], color='red', linestyle='--', linewidth=1.5, label='True depth')
plt.xlabel("Depth (mm)")
plt.ylabel("Normalised Score")
plt.title(f"Model vs Raw Metrics — {data.image_types[test_idx[example_idx]]}")
plt.legend(loc='upper right', fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_single_image_curve.png", dpi=150)
plt.show()

print("\nAll plots saved.")