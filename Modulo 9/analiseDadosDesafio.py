
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.preprocessing import (
    OneHotEncoder,StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC

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
treino = treino.drop('NU_INSCRICAO', axis=1)
treino["IN_TREINEIRO"] = copia

#Graficos de correlacao
"""
#Analisando a correlacao entre as variaveis númericas
plt.figure(figsize = (20,20))
sns.heatmap(treino.corr().round(2), annot= True)
plt.show()
#Plotagem dos dados e suas correlacoes
corr = treino.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(treino.columns),0.75)
ax.set_yticks(ticks)
ax.set_xticklabels(treino.columns)
ax.set_yticklabels(treino.columns)
plt.show()
"""

#Verificando a diferanca entre os valores da variavel target
zero = 0
um = 0
for i in treino['IN_TREINEIRO']:
    if i == 0:
        zero += 1
    else:
        um += 1
print( zero, um )

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
treino = treino.dropna()
print(treino.isna().sum())

#Divisao da base de dados
base = treino
_, a = base.shape
atributos = base.iloc[:, 0:a - 1].values
classe = base.iloc[:, a - 1].values


#Dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split( atributos, classe, test_size=0.30, random_state=42)

#Modelo ML SVM
regressor = SVC(kernel = 'linear', random_state=1)
regressor.fit(X_train, y_train)
previsores = regressor.predict(X_test)

#Resultados
print(previsores)
print(accuracy_score(y_test, previsores))
print(confusion_matrix(y_test, previsores))