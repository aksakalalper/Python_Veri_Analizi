import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Örnek veri seti
data = {
    'Math': [85, 78, 90, 88, 76, 95, 89, 77, 92, 80],
    'Science': [88, 76, 93, 85, 79, 90, 87, 81, 94, 77],
    'English': [90, 82, 88, 91, 75, 92, 85, 80, 89, 84],
    'History': [78, 85, 86, 87, 79, 85, 88, 76, 90, 82]
}

df = pd.DataFrame(data)
# Korelasyon matrisini hesapla
corr = df.corr()
print(corr)
# Isı haritası (heatmap) oluşturma
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Subjects Correlation Heatmap')
plt.show()
