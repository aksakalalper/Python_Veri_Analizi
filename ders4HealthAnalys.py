import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StudentData():
    def cleanData(self):
        file="studentmentalhealth.csv"
        data=pd.read_csv(file)
        df=pd.DataFrame(data)
        print(df.info(),df.describe())
        print(df["Timestamp"])
        df=df.drop("Timestamp",axis=1)
        print(df.info(),df.describe())
        dfColumns=[]
        for i in df.columns:
            dfColumns.append(i)
        dfColumns[0]="Index"
        dfColumns[1]="Gender"
        dfColumns[3]="Course"
        dfColumns[4]="Study year"
        dfColumns[5]="CGPA"
        dfColumns[7]="Depression"
        dfColumns[8]="Anxiety"
        dfColumns[9]="Panic attack"
        dfColumns[10]="Treatment"
        df=df.rename(columns={'Unnamed: 0':dfColumns[0], 'Choose your gender':dfColumns[1], 'What is your course?':dfColumns[3], 
                            'Your current year of Study':dfColumns[4], 'What is your CGPA?':dfColumns[5],
                            'Do you have Depression?' :dfColumns[7], 'Do you have Anxiety?' :dfColumns[8] ,
                                'Do you have Panic attack?' :dfColumns[9], 'Did you seek any specialist for a treatment?':dfColumns[10]})
        df.to_csv("cleandata.csv")

    def analysis1(self):
        #Anksiyete yaşayan cinsiyetler analiz edildiğinde ansiyete yaşama oranı erkeklerde daha yüksektir.
        file="cleandata.csv" 
        data=pd.read_csv(file)
        df=pd.DataFrame(data)
        dfColumnsName=df.columns
        # Anksiyete yaşayan kadın ve erkeklerin sayısını ayrı ayrı hesaplama 
        anxietyCounts = df[df['Anxiety'] == 'Yes'].groupby('Gender').size().reset_index(name='Count') 
        # Kadın ve erkekleri ayrı ayrı ekrana yazdırma 
        womanCount = anxietyCounts[anxietyCounts['Gender'] == 'Female']['Count'].sum() 
        menCount = anxietyCounts[anxietyCounts['Gender'] == 'Male']['Count'].sum() 
        label=f"{womanCount} womanCount",f"{menCount} menCount"
        value=[womanCount,menCount]
        colors=["blue","red","orange"]
        plt.pie(value,labels=label,colors=colors,autopct="%0.1f%%")
        #plt.show()
        ax=sns.displot(data=df.query('Anxiety=="Yes"'), x="Course", hue="Anxiety", palette='viridis')
        sns.set_context("notebook", font_scale=0.5)
        # Barların üzerine adet ekleme 
        for p in ax.ax.patches: 
            ax.ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plt.tight_layout()
        plt.show()

student=StudentData()
student.analysis1()

