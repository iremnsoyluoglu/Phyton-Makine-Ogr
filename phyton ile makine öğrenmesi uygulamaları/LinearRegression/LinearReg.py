from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Veri oluştur
X = np.random.rand(100, 1)
y = 3 + 4 * X  # y = 3 + 4x

# Model oluştur ve eğit
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Katsayılar
a1 = lin_reg.coef_[0][0]
a0 = lin_reg.intercept_[0]
print("a1: ", a1)
print("a0: ", a0)

# Grafik
plt.figure()
plt.scatter(X, y, label="Veriler")
plt.plot(X, lin_reg.predict(X), color="red", alpha=0.7, label="Regresyon Doğrusu")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lineer Regresyon")
plt.legend()
plt.grid(True)
plt.savefig("LinearReg", dpi=300)
plt.show()
