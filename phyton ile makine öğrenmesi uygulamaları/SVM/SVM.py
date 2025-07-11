from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Veri setini yükle
digits = load_digits()

# Görsel olarak ilk 10 görüntüyü çiz
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")
    ax.set_title(f"Label: {digits.target[i]}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# Veri ve hedef değişkenleri
X = digits.data
y = digits.target

# Eğitim ve test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM modeli oluştur ve eğit
svm_clf = SVC(kernel="linear", random_state=42)
svm_clf.fit(X_train, y_train)

# Tahmin ve performans
y_pred = svm_clf.predict(X_test)
print(classification_report(y_test, y_pred))

