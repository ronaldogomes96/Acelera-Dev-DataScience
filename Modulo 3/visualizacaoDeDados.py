import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('houses_to_rent_v2.csv')
df.rename(columns = {'rent amount (R$)' : 'valor_aluguel'}, inplace = True)

sns.countplot(x='city', data=df)

##Grafico de barras normal
plt.figure(figsize = (12 , 6))
sns.barplot(x = 'city',y = 'valor_aluguel', data = df.groupby('city')['valor_aluguel'].mean().reset_index())
plt.title('Media do valor do aluguel por cidade')
plt.show()

#Grafico de densidade
sns.distplot(df["valor_aluguel"])
plt.show()

#Grafico de scaterplt
sns.scatterplot(x = 'valor_aluguel', y = 'bathroom', hue = 'city', size = 'floor', data = df)
plt.show()

#Grafico de correlacao
sns.heatmap(df.corr() , annot = True)
plt.show()