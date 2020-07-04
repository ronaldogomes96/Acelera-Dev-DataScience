
import pandas as pd

df = pd.read_csv('houses_to_rent_v2.csv')

#Mostra os 6 prieiros registros do data frame
print(df.head(6))

#Verificando os tipos
print(df.dtypes)

#Verificando quantos nulos existem
print(df.isna().sum())

#Vendo as informacoes do data frame
print(df.info())