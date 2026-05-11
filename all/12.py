"""
Hierarchical (Agglomerative) Clustering in Python
==================================================
Demonstrates:
  - Generating sample data
  - Fitting agglomerative clustering
  - Plotting the dendrogram
  - Visualising the cluster assignments
  - Choosing the number of clusters from the dendrogram
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# ── Save folder: saves PNGs in the same folder as this script ─────────────
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── 1. Generate sample data ────────────────────────────────────────────────
np.random.seed(42)
X, y_true = make_blobs(
    n_samples=150,
    centers=4,
    cluster_std=0.8,
    random_state=42,
)

# Standardise features (good practice before clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 2. Build linkage matrix for the dendrogram ────────────────────────────
Z = linkage(X_scaled, method="ward", metric="euclidean")


# ── 3. Plot the dendrogram + cluster assignments ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
dendrogram(
    Z,
    ax=ax,
    truncate_mode="lastp",
    p=20,
    leaf_rotation=90,
    leaf_font_size=9,
    show_contracted=True,
    color_threshold=6,
)
ax.set_title("Dendrogram (Ward linkage)", fontsize=13)
ax.set_xlabel("Cluster index (sample count)")
ax.set_ylabel("Merge distance")
ax.axhline(y=6, color="red", linestyle="--", linewidth=1.2,
           label="Cut-off -> 4 clusters")
ax.legend(fontsize=9)


# ── 4. Fit agglomerative clustering ──────────────────────────────────────
N_CLUSTERS = 4

model = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    linkage="ward",
    metric="euclidean",
)
labels = model.fit_predict(X_scaled)


# ── 5. Plot cluster assignments ───────────────────────────────────────────
ax2 = axes[1]
colours = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
for k in range(N_CLUSTERS):
    mask = labels == k
    ax2.scatter(
        X_scaled[mask, 0], X_scaled[mask, 1],
        c=colours[k], s=50, alpha=0.8, edgecolors="white",
        linewidths=0.4, label=f"Cluster {k}",
    )
ax2.set_title(f"Agglomerative clustering (k = {N_CLUSTERS})", fontsize=13)
ax2.set_xlabel("Feature 1 (scaled)")
ax2.set_ylabel("Feature 2 (scaled)")
ax2.legend(fontsize=9)

plt.tight_layout()
out1 = os.path.join(SAVE_DIR, "hierarchical_clustering.png")  # ✅ FIXED PATH
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {out1}")


# ── 6. Linkage methods comparison ────────────────────────────────────────
methods = ["ward", "complete", "average", "single"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, method in zip(axes, methods):
    Z_m = linkage(X_scaled, method=method, metric="euclidean")
    dendrogram(Z_m, ax=ax, truncate_mode="lastp", p=15,
               leaf_rotation=90, leaf_font_size=8, no_labels=True)
    ax.set_title(f"{method.capitalize()} linkage", fontsize=11)
    ax.set_ylabel("Merge distance")

plt.suptitle("Linkage method comparison", fontsize=13, y=1.02)
plt.tight_layout()
out2 = os.path.join(SAVE_DIR, "linkage_comparison.png")       # ✅ FIXED PATH
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {out2}")


# ── 7. Print cluster summary ──────────────────────────────────────────────
print("\nCluster sizes:")
for k in range(N_CLUSTERS):
    print(f"  Cluster {k}: {np.sum(labels == k)} samples")

print("\nKey hyperparameters to tune:")
print("  n_clusters  - use dendrogram cut-off to decide")
print("  linkage     - 'ward' usually best; 'single' sensitive to outliers")
print("  metric      - 'euclidean' default; try 'cosine' for text/sparse data")