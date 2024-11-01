import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Veri seti eklenir
file = "kredi_veriseti.xlsx"
data = pd.read_excel(file)
dfRaw = pd.DataFrame(data=data)

# Cinsiyet sütununa gruplama yapıldı
dfNew = pd.get_dummies(dfRaw, columns=["cinsiyet"])

# X ve Y eksen atamaları yapıldı
x = dfNew.drop(["kredi"], axis=1)  # Bağımsız değişkenler buradadır
y = dfNew['kredi']  # Bağımlı değişkenler buradadır

# Test ve eğitim yapıldı
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.33, random_state=42)

# Ölçeklendirme yapılır
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)  # Test verilerini aynı ölçekleyiciyle ayrı ölçeklendirmek

# Modelleme ve Cross-validation ile değerlendirme
model = LogisticRegression(random_state=0)
cross_val_scores = cross_val_score(model, xTrain, yTrain, cv=5)
print(f"Cross-validation doğruluk oranları: {cross_val_scores}")
print(f"Cross-validation ortalama doğruluk oranı: {cross_val_scores.mean()}")

model.fit(xTrain, yTrain)
yPredict = model.predict(xTest)

# Sonuçlar yazdırılır
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)
print("Predictions: ", yPredict)
print("True Values: ", yTest.values)
print("Confusion Matrix: \n", confusion_matrix(y_true=yTest, y_pred=yPredict))
print("Accuracy Score: ", accuracy_score(y_true=yTest, y_pred=yPredict))
print("Classification Report: \n", classification_report(y_true=yTest, y_pred=yPredict))

# Kullanıcıdan veri alma ve tahmin yapma
def kullanici_verisi_al():
    # Kullanıcıdan verileri al
    yas = float(input("Yaş: "))
    gelir = float(input("Gelir: "))
    cinsiyet = int(input("Cinsiyet (Erkek: 1, Kadın: 0): "))

    # Yeni veriyi DataFrame olarak oluştur
    yeni_veri = pd.DataFrame({
        'yas': [yas],
        'gelir': [gelir],
        'cinsiyet_Erkek': [cinsiyet]
    })

    # Veri setindeki diğer dummy değişkenleri de ekleyelim
    for col in x.columns:
        if col not in yeni_veri.columns:
            yeni_veri[col] = 0
    
    # Sütunları aynı sıraya getir
    yeni_veri = yeni_veri[x.columns]

    return yeni_veri

# Kullanıcıdan veri al ve tahmin yap
yeni_veri_df = kullanici_verisi_al()
yeni_veri_df_scaled = sc.transform(yeni_veri_df)

tahmin = model.predict(yeni_veri_df_scaled)
print(f"Tahmin edilen kredi durumu: {tahmin[0]}")
