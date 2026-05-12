import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA

# ─────────────────────────────────────────
# 1. LOAD EXCEL FILE
# ─────────────────────────────────────────

file_name = "data.xlsx"

df = pd.read_excel(file_name)

print("\n📘 Dataset")
print(df.head())

print("\nShape:", df.shape)

# ─────────────────────────────────────────
# 2. FEATURES
# (No target in clustering)
# ─────────────────────────────────────────

# Select only numeric columns
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
# 4. ELBOW METHOD
# ─────────────────────────────────────────

inertia = []

K_range = range(1, 11)

for k in K_range:

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)

    inertia.append(
        km.inertia_
    )

plt.figure(figsize=(8,5))

plt.plot(
    K_range,
    inertia,
    marker='o'
)

plt.xlabel("K")
plt.ylabel("WCSS")

plt.title(
    "Elbow Method"
)

plt.show()

# ─────────────────────────────────────────
# 5. SILHOUETTE SCORE
# ─────────────────────────────────────────

sil_scores = []

for k in range(2, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    sil_scores.append(score)

    print(
        f"K={k} | "
        f"Silhouette="
        f"{round(score,4)}"
    )

plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    sil_scores,
    marker='s'
)

plt.xlabel("K")
plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score vs K"
)

plt.show()

best_k = list(
    range(2,11)
)[
    np.argmax(
        sil_scores
    )
]

print(
    "\n✅ Best K:",
    best_k
)

# ─────────────────────────────────────────
# 6. FINAL K-MEANS MODEL
# ─────────────────────────────────────────

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
    max_iter=300
)

kmeans.fit(X_scaled)

labels = kmeans.labels_

print(
    "\n✅ KMeans Trained"
)

# ─────────────────────────────────────────
# 7. CLUSTERING METRICS
# ─────────────────────────────────────────

inertia_val = kmeans.inertia_

sil = silhouette_score(
    X_scaled,
    labels
)

db = davies_bouldin_score(
    X_scaled,
    labels
)

print(
    "\n📘 Evaluation"
)

print(
    "Inertia:",
    round(inertia_val,4)
)

print(
    "Silhouette Score:",
    round(sil,4)
)

print(
    "Davies Bouldin:",
    round(db,4)
)

# ─────────────────────────────────────────
# 8. CLUSTER CENTERS
# ─────────────────────────────────────────

centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

centers_df = pd.DataFrame(
    centers,
    columns=X.columns
)

print(
    "\nCluster Centers"
)

print(
    centers_df.round(3)
)

# ─────────────────────────────────────────
# 9. CLUSTER DISTRIBUTION
# ─────────────────────────────────────────

unique, counts = np.unique(
    labels,
    return_counts=True
)

print(
    "\nCluster Distribution"
)

for u, c in zip(
    unique,
    counts
):

    print(
        f"Cluster {u}: {c}"
    )

# ─────────────────────────────────────────
# 10. PCA VISUALIZATION
# ─────────────────────────────────────────

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(figsize=(8,5))

for i in range(best_k):

    mask = labels == i

    plt.scatter(
        X_pca[mask,0],
        X_pca[mask,1],
        label=f'Cluster {i}'
    )

centers_pca = pca.transform(
    kmeans.cluster_centers_
)

plt.scatter(
    centers_pca[:,0],
    centers_pca[:,1],
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "KMeans Clusters"
)

plt.legend()

plt.show()

# ─────────────────────────────────────────
# 11. PAIRPLOT
# ─────────────────────────────────────────

df['Cluster'] = labels

sns.pairplot(
    df,
    hue='Cluster'
)

plt.show()

# ─────────────────────────────────────────
# 12. HEATMAP
# ─────────────────────────────────────────

plt.figure(figsize=(8,5))

sns.heatmap(
    centers_df,
    annot=True,
    fmt='.2f',
    cmap='coolwarm'
)

plt.title(
    "Cluster Centers"
)

plt.show()

# ─────────────────────────────────────────
# 13. INITIALIZATION METHOD
# ─────────────────────────────────────────

print(
    "\nInitialization Comparison"
)

for init in [
    'k-means++',
    'random'
]:

    km = KMeans(
        n_clusters=best_k,
        init=init,
        random_state=42,
        n_init=10
    )

    km.fit(
        X_scaled
    )

    lbl = km.labels_

    sil = silhouette_score(
        X_scaled,
        lbl
    )

    db = davies_bouldin_score(
        X_scaled,
        lbl
    )

    print(
        init,
        "| Silhouette:",
        round(sil,4),
        "| DB:",
        round(db,4)
    )

# ─────────────────────────────────────────
# 14. ITERATIONS
# ─────────────────────────────────────────

print(
    "\nIterations:",
    kmeans.n_iter_
)

print(
    "Final Inertia:",
    round(
        kmeans.inertia_,
        4
    )
)