from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Veri seti
iris = load_iris()
X = iris.data #features
y = iris.target #target

# Eğitim ve test verisi
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2, random_state=42)

# Decision Tree modeli oluştur ve eğit
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# Tahmin ve değerlendirme
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("İris veri seti ile eğitilen DT modeli doğruluğu:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Karar ağacı çizimi ve kaydı
plt.figure(figsize=(15, 10))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.savefig("DT_Model.png", dpi=300)  
plt.show()

# Özellik önem dereceleri
features_importances = tree_clf.feature_importances_
feature_names = iris.feature_names
features_importances_sorted = sorted(zip(features_importances, feature_names), reverse=True)

print("Özellik Önem Dereceleri:")
for importance, name in features_importances_sorted:
    print(f"{name}: {importance:.4f}")