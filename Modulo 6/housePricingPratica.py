import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('train.csv')

#Criando um dataframe auxliar para analisar a consistencia das variaveis
cons = pd.DataFrame({'colunas' : df.columns,
                    'tipo': df.dtypes,
                    'missing' : df.isna().sum(),
                    'size' : df.shape[0],
                    'unicos': df.nunique()})

#Procentagem de valores faltantes
cons['percentual'] = round(cons['missing'] / cons['size'],2)

#Grafico de valores faltantes
cons.percentual.plot.hist( bins = 5)
plt.show()

