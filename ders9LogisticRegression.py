import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,OneHotEncoder
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# veri seti eklenir.
file="kredi_veriseti.xlsx"
data=pd.read_excel(file)
df=pd.DataFrame(data=data)
# cinsiyet sütununa gruplama yapıldı
dfNew=pd.get_dummies(df,columns=["cinsiyet"])
# x ve y eksen atamaları yapıldı.
x=dfNew.drop(["kredi"],axis=1)
y=dfNew['kredi']
# test ve eğitim yapıldı.
xTrain,xTest,yTrain,yTest=train_test_split(x,y,test_size=0.33,random_state=42)
# ölçeklendirme yapılır.
sc=StandardScaler()
xTrain=sc.fit_transform(xTrain)
xTest=sc.fit_transform(xTest)
# modelleme
model=LogisticRegression(random_state=0)
model.fit(xTrain,yTrain)
yPredict=model.predict(xTest)
# sonuclar yazdirilir.
print(model.intercept_,model.coef_)
print(yPredict)
print(yTest.values)
print(confusion_matrix(y_true=yTest,y_pred=yPredict))
print(accuracy_score(y_true=yTest,y_pred=yPredict))
print(classification_report(y_true=yTest,y_pred=yPredict))