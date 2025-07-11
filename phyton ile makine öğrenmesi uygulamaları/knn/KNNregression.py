import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Veri oluşturma
X = np.sort(5 * np.random.rand(40, 1))  # 0 ile 5 arasında 40 tane veri
y = np.sin(X).ravel()  # hedef değişken

# Gürültü ekleme (her 5. elemana rastgele bir sapma)
y[::5] += 1 * (0.5 - np.random.rand(8))

# Tahmin için kullanılacak test verisi
T = np.linspace(0, 5, 500)[:, np.newaxis]

# Her ağırlık tipi için döngü
for i, weight in enumerate(["uniform", "distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)

    # Grafik çizimi
    plt.figure(figsize=(8, 4))
    plt.scatter(X, y, color="green", label="Veri")
    plt.plot(T, y_pred, color="blue", label="Tahmin")
    plt.title(f"KNN Regressor (weights = '{weight}')")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"knn_regression_{weight}.png")

    plt.show()
