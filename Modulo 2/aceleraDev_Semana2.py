
#importando pacotes
import pandas as pd
import numpy as np

#Criando um dicionario com os dados
dados = {'canal_de_venda' : ['Facebook', 'Twitter', 'Instagram', 'Linkedin', 'Facebook'],
         'acessos': [100,200,300,400,500],
         'site': ['site1','site1', 'site2', 'site2', 'site3' ],
         'vendas': [1000.52, 1052.34, 2002, 5000, 300]}

#Printa os dados do dicionario na tela
print(dados)

#Verificando o tipo de dicionario
print(type(dados))

#Acessando as chaves do meu dicionario
print(dados.keys())

#Acessando chave especifica
print(dados['canal_de_venda'])

#Acessando posicao especifica de um dicionario
print(dados['canal_de_venda'][2])

#Acessando posicoes especificas de um dicionario com intervalo
print(dados['canal_de_venda'][:2])


#Listas

#Verificando se é uma lista 
print(type([1,2,3]))

#Criando uma lista
lista = [200, 200 , 300 ,800, 200]

#Printando a lista
print(lista)

#Vendo valores especificos
print(lista[1])

#Fatia de lista
print(lista[:2])

#Adicionando a lista ao dicionario
dados['lista'] = lista

print(dados)

#Data Frames

#Criando um data frame a partir de um dicionario
dataframe = pd.DataFrame(dados)

#printando o data frame
print(dataframe)

#Printando os 2 primeiros casos do data frame
print(dataframe.head(2))

#Verificando o formato do data frame
print(dataframe.shape)

#Verificando o indice do dataframe
print(dataframe.index)

#Verificando os tipos dos dados do dataframe
print(dataframe.dtypes)
print(dataframe.dtypes.value_counts())

#Verificando se existem valores faltantes, mostrando a tabela
print(dataframe.isna())

#Mostra o numero de acordo com a varival
print(dataframe.isna().sum())

#printando os nomes das colunas
print(dataframe.columns)

#Acessando uma coluna especifica
print(dataframe['canal_venda'])

# Criando uma nova coluna
dataframe['nova_coluna'] = [1, 2, 3, 4, 5]

#Mostrando novamente as colunas
print(dataframe.columns)

# Removendo colunas
dataframe.drop(columns = ['acessos','site', 'canal_venda'])
#Ele so elemina no data frame originial se usar o parametro inplace = true


