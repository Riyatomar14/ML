import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,
                              davies_bouldin_score,
                              confusion_matrix,
                              adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Shape:", df.shape)
print("\nActual Class Distribution:")
print(df['target'].value_counts())
print("Classes:", data.target_names)

# 2. Features (no target for clustering - unsupervised)
X = df.drop('target', axis=1).values
y_true = df['target'].values  # only for evaluation

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Elbow Method - find optimal K
inertia = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k,
                random_state=42,
                n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, color='steelblue',
         marker='o', linewidth=2)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method - Optimal K")
plt.xticks(K_range)
plt.tight_layout()
plt.savefig("elbow_method.png")
plt.show()

# 5. Silhouette Score for each K
sil_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k,
                random_state=42,
                n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"  K={k} | Silhouette Score: "
          f"{round(silhouette_score(X_scaled, labels), 4)}")

plt.figure(figsize=(8, 4))
plt.plot(range(2, 11), sil_scores, color='red',
         marker='s', linewidth=2)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig("silhouette_scores.png")
plt.show()

best_k = range(2, 11)[np.argmax(sil_scores)]
print(f"\nBest K (Silhouette): {best_k}")

# 6. Train final KMeans with best K
kmeans = KMeans(n_clusters=3,
                random_state=42,
                n_init=10,
                max_iter=300)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# 7. Evaluation Metrics
inertia_val = kmeans.inertia_
sil         = silhouette_score(X_scaled, labels)
db          = davies_bouldin_score(X_scaled, labels)
ari         = adjusted_rand_score(y_true, labels)

print("\n--- Clustering Evaluation Metrics ---")
print(f"Inertia (WCSS)      : {round(inertia_val, 4)}")
print(f"Silhouette Score    : {round(sil, 4)}")
print(f"Davies-Bouldin Index: {round(db, 4)}")
print(f"Adjusted Rand Index : {round(ari, 4)}")

# 8. Cluster Centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers,
             columns=data.feature_names)
centers_df.index.name = 'Cluster'
print("\n--- Cluster Centers ---")
print(centers_df.round(3))

# 9. Cluster distribution
print("\n--- Cluster Distribution ---")
unique, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Cluster {u}: {c} samples")

# 10. Visualize clusters using PCA (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))
colors = ['steelblue', 'red', 'green']
for i in range(3):
    mask = labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=colors[i], s=50, alpha=0.7,
                label=f'Cluster {i}')

# Plot centroids in PCA space
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            color='black', marker='X', s=200,
            zorder=5, label='Centroids')

plt.xlabel(f"PC1 ({round(pca.explained_variance_ratio_[0]*100, 1)}%)")
plt.ylabel(f"PC2 ({round(pca.explained_variance_ratio_[1]*100, 1)}%)")
plt.title("K-Means Clustering (PCA 2D View)")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_clusters_pca.png")
plt.show()

# 11. Actual vs Predicted clusters comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i in range(3):
    mask = y_true == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=colors[i], s=50, alpha=0.7,
                label=data.target_names[i])
plt.title("Actual Classes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.subplot(1, 2, 2)
for i in range(3):
    mask = labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=colors[i], s=50, alpha=0.7,
                label=f'Cluster {i}')
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            color='black', marker='X', s=200,
            zorder=5, label='Centroids')
plt.title("K-Means Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()

plt.suptitle("Actual vs K-Means Clustering", fontsize=13)
plt.tight_layout()
plt.savefig("actual_vs_clusters.png")
plt.show()

# 12. Pairplot of features colored by cluster
df['Cluster'] = labels
plt.figure(figsize=(10, 8))
sns.pairplot(df.drop('target', axis=1),
             hue='Cluster',
             palette={0:'steelblue',
                      1:'red',
                      2:'green'})
plt.suptitle("Pairplot - K-Means Clusters", y=1.02)
plt.savefig("pairplot_clusters.png")
plt.show()

# 13. Heatmap of cluster centers
plt.figure(figsize=(8, 4))
sns.heatmap(centers_df,
            annot=True, fmt='.2f',
            cmap='coolwarm',
            linewidths=0.5)
plt.title("Cluster Centers Heatmap")
plt.tight_layout()
plt.savefig("cluster_centers_heatmap.png")
plt.show()

# 14. Effect of initialization method
print("\n--- Initialization Method Comparison ---")
for init in ['k-means++', 'random']:
    km = KMeans(n_clusters=3,
                init=init,
                random_state=42,
                n_init=10)
    km.fit(X_scaled)
    lbl = km.labels_
    sil = silhouette_score(X_scaled, lbl)
    db  = davies_bouldin_score(X_scaled, lbl)
    print(f"  {init:12s} -> Silhouette: "
          f"{round(sil, 4)}  DB Index: {round(db, 4)}")

# 15. Number of iterations to converge
print(f"\nIterations to converge: {kmeans.n_iter_}")
print(f"Final Inertia        : {round(kmeans.inertia_, 4)}")