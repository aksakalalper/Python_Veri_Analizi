import pandas as pd

file="titanic.csv"
data=pd.read_csv(file)
df=pd.DataFrame(data)
print(df["Age"].isnull().sum()) # yaş sütununda 177 adet boş öğe var.
df["Age"].fillna(value=df["Age"].mean(),inplace=True) #boş değerler ortalama ile dolduruldu.
print(df["Age"].isnull().sum())
print(df.describe(),df.info())

#normalizasyon: burada amaç verileri 0-1 arasında normalize etmektir. scikit kütüphanesinde bunu yapan hazır fonksiyonlar var.
def normalize():
    print(df["Age"].min(),df["Age"].max())
    normalize=(df["Age"]-df["Age"].min())/(df["Age"].max()-df["Age"].min())
    print(normalize)
normalize()

#standardizasyon: -3 ile +3 arasında ölçeklendirilir. verilerin %99,7 lik kısmı 6sigma arasında toplanır. 
def standardize():
    standardizeRes=(df["Age"]-df["Age"].mean())/df["Age"].std()
    print(standardizeRes)
standardize()

#kesikli hale getirme: verileri gruplandırmaya yarar.
def discretization():
    yasAraligi=[0,20,50,100]
    kategoriler=['çocuk','genç','yaşlı']
    df["Age"]=pd.cut(x=df["Age"],bins=yasAraligi,labels=kategoriler,right=False)
    print(df["Age"])
discretization()

#ikili hale getirme: verileri sadece iki gruba ayırır.
def getDummies():
    print(pd.get_dummies(df["Sex"],drop_first=True))
getDummies()