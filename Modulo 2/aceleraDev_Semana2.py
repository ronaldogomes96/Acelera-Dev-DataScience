
#importando pacotes
import pandas as pd
import numpy as np

#Criando um dicionario com os dados
dados = {'canal_venda' : ['Facebook', 'Twitter', 'Instagram', 'Linkedin', 'Facebook'],
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
print(dados['canal_venda'])

#Acessando posicao especifica de um dicionario
print(dados['canal_venda'][2])

#Acessando posicoes especificas de um dicionario com intervalo
print(dados['canal_venda'][:2])

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

#Acessando valores especificos
print(dataframe['acessos'][1])

#Acessando fatia de coluna especifica
print(dataframe['canal_venda'][:2])

#Fatiando dados com iloc: Linhas/Colunas
print(dataframe.iloc[1:2, 1:3])

#Fatiar os dados usando o loc (indice), que so pega o indice
print(dataframe.loc[:3])

#Selecionando colunas especificas
print(dataframe[['canal_venda', 'vendas']])

#Podemos passar os valores da coluna atraves de uma lista
filtro = ['canal_venda', 'acessos']
print(dataframe[filtro])

#Metodo info diz as caracteristicas do dataframe
print(dataframe.info())

#Pivotando os dados (coluna), funciona como uma juncao de colunas
aux = dataframe.pivot(index = 'canal_venda', columns='site', values='acessos')
print(aux)

#Completando os valores faltantes usando fillna, dizendo que os faltantes vao ser igualados a 0
#Pivotando os dados (coluna)
aux = dataframe.pivot(index = 'canal_venda', columns='site', values='acessos').fillna(0)
print(dataframe.pivot(index = 'canal_venda', columns='site', values='acessos').fillna(0))

# Mudando as colunas usando o comando melt
print(dataframe.melt(id_vars='site', value_vars=['canal_venda']))

#Resetando o indice do dataframe, voltando ao original
print(aux.columns)
aux = dataframe.reset_index()
print(aux.columns)

aux.reset_index()

#Exemplo do comando melt
#print(aux.melt(id_vars='canal_venda', value_vars=['site1', 'site2', 'site3']))

#Somando as colunas do dataframe
print(dataframe.sum())
print(dataframe.sum(axis = 1))

#Calculando a mediana das colunas numericas
print('Por linha: ',dataframe.median(axis= 1) )
print('Por coluna: ', dataframe.median())

#So pega os valores de numeros, nao pega as strings
#Calculando a media das colunas númericas
print(dataframe.mean())

#Calculando o desvio padrão das colunas numericas
print(dataframe.std())

#Usando o comando describe que calcula estatisticas descritivas para colunas numericas
print(dataframe.describe())

#Calculando a moda
print(dataframe.mode())

#Mostrando os valores max e min
print(dataframe.max() , dataframe.min())

#Printando o numero de unicos
print(dataframe.nunique())

#Contando valores unicos de uma coluna
print(dataframe['canal_venda'].value_counts())

#String com o nome dos valores unicos de uma coluna
print(dataframe['canal_venda'].unique())

# Usando o groupby com valores numericos
print(dataframe.groupby('site')['acessos'].sum())

# Usando o groupby com valores numericos
print(dataframe.groupby('canal_venda')['acessos'].median())

# Usando o groupby com categoricos
print(dataframe.groupby('site')['canal_venda'].first())

#Usando o groupby com a função agg
print(dataframe.groupby('canal_venda').agg({'site': 'unique',
                                     'acessos': 'sum'}))

#Correlações entre variaveis
print(dataframe.corr(method = 'spearman'))

#Criando variaveis categoricas por fatia de variavel numerica
dataframe['categoria_vendas'] = pd.cut(dataframe['vendas'],
                                       bins= (0, 1500, 2000, 8000), 
                                       labels = ('0 a 1500', '1500 a 2000', '2000 a 8000'))

#Criando variavel categorica usando compressao de lista
dataframe['categoria_acessos'] = ['maior_que_300' if x > 300 else 'menor_que_300' for x in dataframe['acessos']]



