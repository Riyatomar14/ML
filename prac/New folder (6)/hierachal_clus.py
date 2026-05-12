import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ─────────────────────────────────────────
# 1. LOAD EXCEL FILE
# ─────────────────────────────────────────

file_name = "data.xlsx"

df = pd.read_excel(file_name)

print("\n📘 Dataset")
print(df.head())

print("\nShape:", df.shape)

# ─────────────────────────────────────────
# 2. SELECT FEATURES
# (No target needed)
# ─────────────────────────────────────────

X = df.select_dtypes(
    include=np.number
)

print("\nFeatures Used:")
print(X.columns.tolist())

# ─────────────────────────────────────────
# 3. FEATURE SCALING
# ─────────────────────────────────────────

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n✅ Feature Scaling Done")

# ─────────────────────────────────────────
# 4. DENDROGRAM
# ─────────────────────────────────────────

Z = linkage(
    X_scaled,
    method='ward',
    metric='euclidean'
)

plt.figure(figsize=(10,5))

dendrogram(
    Z,
    truncate_mode='lastp',
    p=20,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel("Clusters")
plt.ylabel("Distance")

plt.show()

# ─────────────────────────────────────────
# 5. CHOOSE NUMBER OF CLUSTERS
# ─────────────────────────────────────────

n_clusters = int(
    input(
        "\nEnter number of clusters: "
    )
)

# ─────────────────────────────────────────
# 6. TRAIN MODEL
# ─────────────────────────────────────────

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage='ward',
    metric='euclidean'
)

labels = model.fit_predict(
    X_scaled
)

print("\n✅ Hierarchical Model Trained")

# ─────────────────────────────────────────
# 7. CLUSTER DISTRIBUTION
# ─────────────────────────────────────────

print("\nCluster Sizes")

unique, counts = np.unique(
    labels,
    return_counts=True
)

for u, c in zip(
    unique,
    counts
):
    print(
        f"Cluster {u}: {c}"
    )

# ─────────────────────────────────────────
# 8. PCA VISUALIZATION
# ─────────────────────────────────────────

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(figsize=(8,5))

for i in range(n_clusters):

    mask = labels == i

    plt.scatter(
        X_pca[mask,0],
        X_pca[mask,1],
        label=f'Cluster {i}'
    )

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Hierarchical Clustering"
)

plt.legend()

plt.show()

# ─────────────────────────────────────────
# 9. ADD CLUSTER COLUMN
# ─────────────────────────────────────────

df['Cluster'] = labels

print("\nClustered Data")
print(df.head())

# ─────────────────────────────────────────
# 10. CLUSTER CENTERS
# (Mean values of clusters)
# ─────────────────────────────────────────

cluster_centers = df.groupby(
    'Cluster'
).mean()

print("\nCluster Centers")
print(
    cluster_centers.round(3)
)

# ─────────────────────────────────────────
# 11. HEATMAP
# ─────────────────────────────────────────

plt.figure(figsize=(8,5))

sns.heatmap(
    cluster_centers,
    annot=True,
    fmt='.2f',
    cmap='coolwarm'
)

plt.title(
    "Cluster Centers Heatmap"
)

plt.show()

# ─────────────────────────────────────────
# 12. PAIRPLOT
# ─────────────────────────────────────────

sns.pairplot(
    df,
    hue='Cluster'
)

plt.show()

# ─────────────────────────────────────────
# 13. LINKAGE COMPARISON
# ─────────────────────────────────────────

methods = [
    'ward',
    'complete',
    'average',
    'single'
]

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18,5)
)

for ax, method in zip(
    axes,
    methods
):

    Z_method = linkage(
        X_scaled,
        method=method
    )

    dendrogram(
        Z_method,
        ax=ax,
        truncate_mode='lastp',
        p=15,
        no_labels=True
    )

    ax.set_title(
        method.capitalize()
    )

plt.tight_layout()

plt.show()

# ─────────────────────────────────────────
# 14. SAVE OUTPUT FILE
# ─────────────────────────────────────────

df.to_excel(
    "clustered_output.xlsx",
    index=False
)

print(
    "\n✅ Output saved as "
    "'clustered_output.xlsx'"
)

# ─────────────────────────────────────────
# 15. HYPERPARAMETERS
# ─────────────────────────────────────────

print("\nKey Hyperparameters")

print(
    "n_clusters → "
    "Number of clusters"
)

print(
    "linkage → "
    "ward, complete, "
    "average, single"
)

print(
    "metric → "
    "euclidean, cosine"
)