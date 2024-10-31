from sklearn.model_selection import train_test_split # verilere öğretme yaparken kullanılan metod.
from sklearn.preprocessing import MinMaxScaler # bağımsız değişkenleri skala eden metod.
from sklearn.linear_model import LinearRegression # regresyon metodu.
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error # r^2 ve hatalar için metod.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class EngineDataAnalysis():
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

       def rawDataScatter(self):
              plt.xlabel("Engine rpm")
              plt.ylabel("Coolant temp")
              plt.title("Engine RPM/Coolant Temperature")
              plt.legend()
              plt.scatter(x=self.x,y=self.y,c = self.x,cmap = 'turbo',alpha = 0.7)
              plt.show()
       
       def learnedData(self):
              # Diziler 2 boyutlu hale getirilir. ve öğretme/test işlemine sokulur.
              engineRPMraw=np.array(self.x).reshape(-1,1)
              coolanTEMP=np.array(self.y).reshape(-1,1)
              engineRpmTrain,engineRpmTest,coolantTrain,coolantTest=train_test_split(engineRPMraw,coolanTEMP,
                                                                                    test_size=0.20,random_state=123445)

              # bağımsız değişken 0-1 arası skala edilir.
              MinMaXScale=MinMaxScaler()
              engineRPM=MinMaXScale.fit_transform(engineRpmTrain)

              # Veriler lineer regresyona sokulur.
              linearModel=LinearRegression()
              linearModel.fit(engineRPM,coolantTrain)
              print(f"formülizasyon=beta0+beta1.(X) beta0: {linearModel.intercept_}, beta1: {linearModel.coef_}")
              engineRPMTestScaled=MinMaXScale.transform(engineRpmTest)
              coolanTEMPpredicted=linearModel.predict(engineRPMTestScaled)

              # r^2 skoru 1 e yakın olması iyi bir uyum olduğunu belirtir. Hataları yazdırır.
              print(f"r^2 skoru: {r2_score(coolantTest,coolanTEMPpredicted)}")
              print(f"ortalama mutlak hata: {mean_absolute_error(coolantTest,coolanTEMPpredicted)},ortalama karesel hata: {mean_squared_error(coolantTest,coolanTEMPpredicted)}")

              # Tahmin ve gerçek değer kıyaslaması grafiği aşağıdaki gibidir.
              plt.plot(coolanTEMPpredicted, color="red", label="Tahmin edilen Coolant TEMP") 
              plt.plot(coolanTEMP, color="blue", label="Gerçek Coolant TEMP") 
              plt.legend()
              plt.show()

engineData=EngineDataAnalysis()
engineData.rawDataScatter()
engineData.learnedData()
