
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

#Estatistica multivariada
"""
    Como cientista de dados precisamos fazer pergntas sobre nossa base de dados e gerar hipoteses a partir do resultado
    Perguntas:
        Qual a cidade com a media de aluguel mais cara?
        Quantos banheiros existem nas casas do media de aluguel mais caro?
        Os imoveis mais caros aceitam animais?
        Os imoveis mais caros sao mobiliados?
"""

#Respondendo as perguntas

#Qual a cidade com a media de aluguel mais cara?
cidadeComAluguelMaisCaro = df.groupby('city')['valor_aluguel'].mean().reset_index()
print(cidadeComAluguelMaisCaro)

#Quantos banheiros existem nas casas do media de aluguel mais caro?
df['aluguel_alto'] = [ 'Alto' if x > 5000 else 'Baixo' for x in df['valor_aluguel']]
banheirosEmAlugueisAltos = df.groupby('aluguel_alto')['bathroom'].mean()
print(banheirosEmAlugueisAltos)

#Correlacao

#Correlacao entre valor do aluguel e o banheiro

correlacaoBanheiroAluguel = df[['valor_aluguel', 'bathroom']].corr()
print(correlacaoBanheiroAluguel)

#Criando uma lista com valor int64
aux = pd.DataFrame({'colunas' : df.columns, 'tipos' : df.dtypes})
listaInteiros = list(aux[aux['tipos'] == 'int64']['colunas'])
for coluna in listaInteiros:
    print(coluna)
    print( df[['valor_aluguel', coluna]].corr())







