
import pandas as pd
import json

df = pd.read_csv('baseDesafio.csv')

#Selecionando as colunas para o uso
dadosImportantes = { 'pontuacao_credito' : df['pontuacao_credito'] ,
                     'estado_residencia' : df['estado_residencia']}

#Criando um novo datframe com essa colunas
data = pd.DataFrame(dadosImportantes)

#Lendo o arquivo json
with open ('submission.json','r') as arquivo:
    texto = arquivo.read()
    submissionJSON = json.loads(texto)
submissionJSON['SC']['moda'] = 10

#Recebendo os valores necessarios
moda = df.groupby('estado_residencia')['pontuacao_credito'].apply(lambda x: x.mode())
media = df.groupby('estado_residencia')['pontuacao_credito'].mean().reset_index()
mediana = df.groupby('estado_residencia')['pontuacao_credito'].median().reset_index()
desvioPadrao = df.groupby('estado_residencia')['pontuacao_credito'].std().reset_index()

#Analisando os resultados
print(moda)
print(mediana)
print(media)
print(desvioPadrao)
