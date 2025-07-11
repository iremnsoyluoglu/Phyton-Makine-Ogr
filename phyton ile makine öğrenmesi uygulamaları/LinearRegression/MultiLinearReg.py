import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# y = a0 + a1*x1 + a2*x2 -> multi-variable linear regression

# Veri oluştur
X = np.random.rand(100, 2)
coef = np.array([3, 5])
y = 0 + np.dot(X, coef)  # y = 3*x1 + 5*x2

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

# Lineer regresyon modeli
model = LinearRegression()
model.fit(X, y)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape))

plt.title("multi variable linear regression")
plt.savefig("Multi variable linear regression", dpi=300)
plt.show()