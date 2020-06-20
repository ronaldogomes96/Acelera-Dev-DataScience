
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

#Criando um data frame auxiliar
aux = pd.DataFrame( { 'colunas': df.columns,
                      'tipos': df.dtypes,
                      'percentual_faltante': df.isna().sum() / df.shape[0] })
print(aux)

#Trantando os dados numericos faltantes, com a media ou mediana
df['Age'] = df['Age'].fillna(df['Age'].mean())

#Tratando dados categoricos faltantes, Unknown ou moda, ou apaga mesmo
df['Cabin'] = df['Cabin'].fillna('Unknown')

#Mostrando os valores de cada variavel, o numero de vez que ela aparece
print(df['Cabin'].value_counts())