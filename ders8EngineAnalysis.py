from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Veri dosyası eklenir ve öğretilecek kolonlar seçilir.
file="engine_data.csv"
data=pd.read_csv(file)
dfRaw=pd.DataFrame(data=data)

# Veri seti bilgileri alınır.
print(dfRaw.describe(),"**",dfRaw.info())
print(dfRaw.columns)

# Kullanılmayacak sütunlar çıkarılır.
df=dfRaw.drop(['Lub oil pressure', 'Fuel pressure', 'Coolant pressure',
       'lub oil temp','Engine Condition'],axis=1)

# Nokta dağılımı grafik oluşturulur.
x=df['Engine rpm'].head(200)
y=df['Coolant temp'].head(200)
'''
plt.xlabel("Engine rpm")
plt.ylabel("Coolant temp")
plt.title("Engine RPM/Coolant Temperature")
plt.legend()
plt.scatter(x=x,y=y,c = x,cmap = 'turbo',alpha = 0.7)
plt.show()
'''
# Dizi  boyutlu hale getirilir.
engineRPMraw=np.array(x).reshape(-1,1)
coolanTEMP=np.array(y).reshape(-1,1)
MinMaXScale=MinMaxScaler()
engineRPM=MinMaXScale.fit_transform(engineRPMraw)

# Veriler lineer regresyona sokulur.
linearModel=LinearRegression()
linearModel.fit(engineRPM,coolanTEMP)
coolanTEMPprdct=linearModel.predict(engineRPM)
print(f"formülizasyon=beta0+beta1.(X) beta0: {linearModel.intercept_}, beta1: {linearModel.coef_}")
coolanTEMPpredicted=linearModel.predict(engineRPM)

#1 e yakın olması iyi bir uyum olduğunu belirtir.
print(f"r^2 skoru: {r2_score(coolanTEMP,coolanTEMPpredicted)}")

# Hataları yazdırma.
print(f"ortalama mutlak hata: {mean_absolute_error(coolanTEMP,coolanTEMPpredicted)}, ortalama karesel hata: {mean_squared_error(coolanTEMP,coolanTEMPpredicted)}")


# Tahmin ve gerçek değer kıyaslaması grafiği aşağıdaki gibidir.
plt.plot(coolanTEMPpredicted, color="red", label="Tahmin edilen CoolanTEMP") 
plt.plot(coolanTEMP, color="blue", label="Gerçek CoolanTEMP") 




plt.legend()
plt.show()