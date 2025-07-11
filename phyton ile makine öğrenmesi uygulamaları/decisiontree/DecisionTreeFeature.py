from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import warnings
warnings.filterwarnings("ignore")

iris = load_iris()
n_classes = len(iris.target_names)
plt_colors = "ryb"

plt.figure(figsize=(14, 8))  

# Özellik çiftleri üzerinden dön
for pairidx, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    print(f"Index: {pairidx}, Pair: {pair}")

    X = iris.data[:, pair]
    y = iris.target

    clf = DecisionTreeClassifier().fit(X, y)

    ax = plt.subplot(2, 3, pairidx + 1)

    DecisionBoundaryDisplay.from_estimator(
        clf, X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]]
    )

    for i, color in zip(range(n_classes), plt_colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color,
                   edgecolors="black", s=30, label=iris.target_names[i])

    ax.legend(fontsize="small", loc="lower right")

plt.tight_layout(pad=3.0)

#  Görseli kaydet
plt.savefig("iris_decision_boundaries.png", dpi=300)

plt.show()
