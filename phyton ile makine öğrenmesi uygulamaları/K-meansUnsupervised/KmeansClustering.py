from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
X, _ =make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.figure()
plt.scatter(X[:, 0], X[:,1])
plt.title("Ornek veri")

KMeans = KMeans(n_clusters= 4)
KMeans.fit(X)

labels = KMeans.labels_

plt.figure()
plt.scatter(X[:,0], X[:,1], c= labels, cmap="viridis")
centers = KMeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c="red", marker="X")
plt.title("K-means")

plt.show()