# sklearn : ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
# *1 Veri Seti incelenmesi
cancer = load_breast_cancer()
df = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
print(df.head())
df["target"] = cancer.target

# *2 Makine Öğrenmesi modelinin seçilmesi -KNN Sınıflandırıcı
# *3 Modelin train edilmesi
X = cancer.data
y = cancer.target
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X,y) #fit fonksiyonu verimizi (samples + target) kullanarak knn algoritmasını eğitir

#train test split
X_train, X_test, y_train,y_test = train_test_split(X , y, test_size= 0.3, random_state= 42)


scaler = StandardScaler()  # <-- 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#knn modeli oluştur ve train et


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
# *4 Sonuçların Değerlendirilmesi :Test
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk", accuracy)

conf_matrix = confusion_matrix(y_test,y_pred)
print("confusion matrix:")
print(conf_matrix)


# *5 Hiperparametre Ayarlanması
"""
KNN : Hyperparamer : K
     K: 1,2,3...N
     Accuracy : %A,%B....
    
"""
accuracy_values = []
k_values = []

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)  # doğru değişken adı
    accuracy_values.append(accuracy)           # listeye ekliyoruz
    k_values.append(k)

plt.figure()
plt.plot(k_values,accuracy_values, marker = "o", linestyle = "-")
plt.title("K değerine göre doğruluk") 
plt.xlabel("K değeri") 
plt.ylabel("Doğruluk") 
plt.grid(True)
plt.savefig("knn_dogruluk_grafigi.png", dpi=300)
plt.show()
