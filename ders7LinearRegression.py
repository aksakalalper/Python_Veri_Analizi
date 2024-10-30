from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

reklamGiderler=np.array([7,11,15,22,26,28,31])
satislar=np.array([223,215,233,264,305,316,320])
index=['2001','2002','2003','2004','2005','2006','2007']
df=pd.DataFrame(data={'reklamGiderleri':reklamGiderler,'satislar':satislar},index=index)
print(df)
#plt.show()
linearModel=LinearRegression()
reklamGiderler=reklamGiderler.reshape(-1,1)
#beta0 ve beta1 belirlendir
linearModel.fit(reklamGiderler,satislar)
print(f"beta0: {linearModel.intercept_}, beta1: {linearModel.coef_}")
print(f"coef: {linearModel.score(reklamGiderler,satislar)} ") #1 e yakın olması iyi bir uyum olduğunu belirtir.
print(linearModel.predict(reklamGiderler))
satislarTahmin=linearModel.predict(reklamGiderler)
hatalar=satislarTahmin-satislar
print(hatalar.sum())
hatalarinKaresi=np.square(hatalar)
print(hatalarinKaresi)
##scikit ile yapalım
print(mean_squared_error(satislar,satislarTahmin))
varyans=np.var(hatalar)
print(varyans) #gerçek değerlerin ortalamasının tahminden ne kadar saptığını gösterir.
plt.plot(satislarTahmin,color="red",label="satislarTahmin")
plt.plot(satislar,color="blue",label="satislar")
plt.legend()
plt.show()