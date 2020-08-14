
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

treino = pd.read_csv("train.csv")
teste = pd.read_csv("test.csv")

#Buscando as colunas que existem no treino mais nao existem no teste e apagando do treino
colunasInexistentesTeste = []
for coluna in treino.columns:
    result = list(filter(lambda x:  x==coluna ,teste.columns))
    if not result:
        colunasInexistentesTeste.append(coluna)
del(colunasInexistentesTeste[9]) #Deletando a coluna da nota de matematica
treino = treino.drop(colunasInexistentesTeste, axis=1)

#Parametrizacao
copia = treino['IN_TREINEIRO']
treino = treino.drop('IN_TREINEIRO', axis=1)
treino["IN_TREINEIRO"] = copia

#Tipos das colunas
print(treino.info())

#Retirando colunas sem correlacao com o target e valores faltantes
colunasNaoImportantes = ["SG_UF_RESIDENCIA", "CO_UF_RESIDENCIA", "TP_SEXO",
                         "TP_COR_RACA", "TP_NACIONALIDADE", "TP_ENSINO",
                         "TP_DEPENDENCIA_ADM_ESC", "IN_BAIXA_VISAO", "IN_CEGUEIRA",
                         "IN_SURDEZ", "IN_DISLEXIA", "IN_DISCALCULIA", "IN_SABATISTA",
                         "IN_GESTANTE", "IN_IDOSO", "TP_PRESENCA_CN", "TP_PRESENCA_CH",
                         "TP_PRESENCA_LC", "TP_PRESENCA_MT", "TP_LINGUA",
                         "TP_STATUS_REDACAO", "NU_NOTA_COMP1", "NU_NOTA_COMP2",
                         "NU_NOTA_COMP3", "NU_NOTA_COMP4", "NU_NOTA_COMP5","Q027","Q001",
                         "Q002", "Q006", "Q024", "Q025", "Q026", "Q047"
                         ]
treino = treino.drop(colunasNaoImportantes, axis = 1)
teste = teste.drop(colunasNaoImportantes, axis = 1)

#Substituindo os valores faltantes por 0
treino = treino.replace(np.NAN, 0)
teste = teste.replace(np.NAN, 0)
numeroInscricaoTeste = list(teste["NU_INSCRICAO"])
treino = treino.drop('NU_INSCRICAO', axis=1)
teste = teste.drop('NU_INSCRICAO', axis=1)


#Divisao da base de dados
base = treino
_, a = base.shape
atributos = base.iloc[:, 0:a - 1].values
classe = base.iloc[:, a - 1].values

#Balanceando os dados
smote = SMOTE(sampling_strategy="minority")
x_treino, y_treino = smote.fit_resample(atributos, classe)

#Modelo ML Random Forrest com 500 estimadores
svm = RandomForestClassifier(n_estimators=500)
svm.fit(x_treino, y_treino)

#Predicao
previsores = svm.predict(teste)

#Criacao da tabela de Resultados
resposta = pd.DataFrame()
resposta['NU_INSCRICAO'] = numeroInscricaoTeste
resposta['IN_TREINEIRO'] = previsores
resposta.to_csv('answer.csv', index=False, header=True)