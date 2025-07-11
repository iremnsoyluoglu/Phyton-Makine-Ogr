from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Ornek Veri")

linkage_methods = ["ward", "single", "average", "complete"]

plt.figure(figsize=(16, 8))
for i, method in enumerate(linkage_methods, 1):
    model = AgglomerativeClustering(n_clusters=4, linkage=method)
    cluster_labels = model.fit_predict(X)

    plt.subplot(2, 4, i)
    plt.title(f"{method.capitalize()} Linkage Dendrogram")
    Z = linkage(X, method=method)
    dendrogram(Z, no_labels=True)
    plt.xlabel("Veri Noktalari")
    plt.ylabel("Uzaklık")

    plt.subplot(2, 4, i + 4)
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis")
    plt.title(f"{method.capitalize()} Linkage Clustering")
    plt.xlabel("X")
    plt.ylabel("Y")

plt.tight_layout()
plt.savefig("Hierarch Clustering ", dpi=300)
plt.show()
