import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')

# Criando um dataframe auxliar para analisar a consistencia das variaveis
cons = pd.DataFrame({'colunas': df.columns,
                     'tipo': df.dtypes,
                     'missing': df.isna().sum(),
                     'size': df.shape[0],
                     'unicos': df.nunique()})

# Procentagem de valores faltantes
cons['percentual'] = round(cons['missing'] / cons['size'], 2)

# Grafico de valores faltantes
cons.percentual.plot.hist(bins=5)
plt.show()

# Mostrando as colunas com dados faltantes
print('Contagem de colunas com ATÉ 20% de dados faltantes', cons[cons.percentual < 0.2].shape[0])
print('Contagem de colunas com 0% de dados faltantes', cons[cons.percentual == 0].shape[0])

# Mantendo somente as colunas sem valores faltantes
cons[cons.percentual == 0]['tipo'].value_counts()
cons['completa'] = ['completa' if x == 0 else 'faltante' for x in cons['percentual']]
mantem = list(cons[cons['completa'] == 'completa']['colunas'])
df = df[mantem]

#Analisando as colunas numericas
colunas_numericas = list(cons[((cons['tipo'] != 'object') &
                              (cons['completa'] == 'completa'))]['colunas'])

# Analise univariavel
for coluna in colunas_numericas:
    print(coluna)
    df[coluna].plot.hist(bins = 50, log= True)
    plt.show()

#Analisando a correlacao entre as variaveis númericas
plt.figure(figsize = (20,20))
sns.heatmap(df[colunas_numericas].corr().round(2), annot= True)

#Variaveis que sao relacionadas com a variavel principal
correlacionadas = ['GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual']
