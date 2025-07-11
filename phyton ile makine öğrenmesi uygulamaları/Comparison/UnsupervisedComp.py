from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples)
no_structure = (np.random.rand(n_samples, 2), None)

datasets_list = [noisy_circles, noisy_moons, blobs, no_structure]

plt.figure(figsize=(18, 10))
plot_num = 1

for i_dataset, dataset in enumerate(datasets_list):
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    two_means = cluster.MiniBatchKMeans(n_clusters=2)
    ward = cluster.AgglomerativeClustering(n_clusters=2, linkage="ward")
    spectral = cluster.SpectralClustering(n_clusters=2, assign_labels="discretize")
    dbscan = cluster.DBSCAN(eps=0.2)
    average_linkage = cluster.AgglomerativeClustering(n_clusters=2, linkage="average")
    birch = cluster.Birch(n_clusters=2)

    clustering_algorithms = [
        ("MiniBatchKMeans", two_means),
        ("Ward", ward),
        ("SpectralClustering", spectral),
        ("DBSCAN", dbscan),
        ("AverageLinkage", average_linkage),
        ("Birch", birch)
    ]

    for name, algorithm in clustering_algorithms:
        algorithm.fit(X)

        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets_list), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", s=5)
        plt.xticks([])
        plt.yticks([])
        plot_num += 1

plt.tight_layout()
plt.savefig("clustering_algorithms_output.png")
plt.show()
