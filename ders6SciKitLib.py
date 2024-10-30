from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#hazır toy veriseti eklendi
iris=datasets.load_iris()
print(type(iris),iris.keys())
x=iris.data #verilerin bulunduğu kısımdır.
y=iris.target #verilerin çıktı kısmıdır. hedef değişkenidir.

#pd.options.display.float_format='{:.1f}'.format #ondalık kısım 1 karakterli.
df=pd.DataFrame(x,columns=iris.feature_names) #veriseti dataframe haline getirilir.
print(df.head())

#x özellik matrisi, y hedef değişkeni, test_size eğitim ve test oranını ayırır. 
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.20,random_state=12345)
MinMaXScale=MinMaxScaler()
xTrainScaled=MinMaXScale.fit_transform(xTrain)
print(MinMaXScale.min_)
print(MinMaXScale.scale_)
print(x.shape,xTrain.shape)
np.set_printoptions(precision=3,suppress=True)
print(xTrainScaled[:5])

#korelasyon matrisi / ısı haritası oluşturuldu.
fig=plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),center=0,vmin=-1,vmax=1,square=True,annot=True,cbar_kws={'shrink':0.8})
plt.show()

