from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

#create data set
X = np.sort(5 * np.random.rand(80,1), axis=0 )
y = np.sin(X).ravel()
y[::5] += 0.5 * (0.5 - np.random.rand(16))

plt.scatter(X,y)

reg_1 = DecisionTreeRegressor(max_depth=2)
reg_2 = DecisionTreeRegressor(max_depth=5)
reg_1.fit(X,y)
reg_2.fit(X,y)

X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred1 = reg_1.predict(X_test)
y_pred2 = reg_2.predict(X_test)

plt.figure()
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X_test,y_pred1 , color = "blue", label = " Max depth: 2", linewidth = 2)
plt.plot(X_test,y_pred2 , color = "green", label = " Max depth: 5", linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
plt.savefig("DT model ", dpi=300)
plt.show()