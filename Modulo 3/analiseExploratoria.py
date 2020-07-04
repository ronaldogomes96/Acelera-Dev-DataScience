
import pandas as pd
import matplotlib

df = pd.read_csv('houses_to_rent_v2.csv')

#Mostra os 6 prieiros registros do data frame
print(df.head(6))

#Verificando os tipos
print(df.dtypes)

#Verificando quantos nulos existem
print(df.isna().sum())

#Vendo as informacoes do data frame
print(df.info())

#Estatistica Univariada

#Mudando o nome do target
df.rename(columns = {'rent amount (R$)' : 'valor_aluguel'}, inplace = True)
print(df['valor_aluguel'])

#Algumas medidas estatisticas
print(df['valor_aluguel'].mean())
print(df['valor_aluguel'].median())
print(df['valor_aluguel'].std())
print(df['valor_aluguel'].describe())

#Plot dos dados com o pandas
print(df['valor_aluguel'].plot(kind = 'hist', bins = 10))