import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ✅ UCI'den heart disease veri setini ID ile al (ID: 45)
heart_disease = fetch_ucirepo(id=45)

# Özellikler ve hedef
df = pd.DataFrame(data=heart_disease.data.features)
target = heart_disease.data.targets

# Eksik veri kontrolü ve temizleme
if df.isna().any().any():
    df.dropna(inplace=True)
    print("NaN değerler silindi.")

# Hedef değişkeni (class label)
df["Target"] = target.values.ravel()

# Özellik ve hedef ayrımı
X = df.drop("Target", axis=1).values
y = df["Target"].values

# Eğitim ve test seti
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Lojistik regresyon modeli
log_reg = LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=1000)
log_reg.fit(X_train, y_train)

# Doğruluk skoru
accuracy = log_reg.score(X_test, y_test)
print("Logistic Regression Accuracy:", accuracy)



