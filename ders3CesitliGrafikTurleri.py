import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file="employees.csv"
data=pd.read_csv(file)
df=pd.DataFrame(data)

'''

'''
#Histogram
def histogram():
    plt.bar(x=xAxis,height=yAxis) #oran olarak gösterildi.
    plt.xlabel("Tezgah kodu")
    plt.ylabel("Tezgah adeti")
    plt.legend()
    plt.show()
#scatter grafik olusturulur.
def scatter():
    sns.relplot(data=df,y="Sex",x="Age",hue="Age",style="Age",col="Sex")
    plt.show()
#bir parametreyle diğer bütün parametreler ilişkilendirilir.
def pairplot():
    sns.pairplot(df,hue="Team")
    plt.show()

df["Team"].dropna()
print(df["Team"].value_counts())
x=df["Team"].dropna().unique()
xLabel=[]
for i in x:
    xLabel.append(i)

y=df["Team"].value_counts()
yLabel=[]
for k in y:
    yLabel.append(k)

print(xLabel,yLabel)

sns.scatterplot(x=xLabel,y=yLabel,size=yLabel,hue=xLabel,sizes=(50,200))
plt.show()