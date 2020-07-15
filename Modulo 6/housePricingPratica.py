import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.features import Rank1D
from yellowbrick.features import PCA
from yellowbrick.target import FeatureCorrelation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from yellowbrick.model_selection import RFECV
from sklearn.decomposition import PCA




df = pd.read_csv('train.csv')

# Criando um dataframe auxliar para analisar a consistencia das variaveis
cons = pd.DataFrame({'colunas': df.columns,
                     'tipo': df.dtypes,
                     'missing': df.isna().sum(),
                     'size': df.shape[0],
                     'unicos': df.nunique()})

# Procentagem de valores faltantes
cons['percentual'] = round(cons['missing'] / cons['size'], 2)

# Grafico de valores faltantes
cons.percentual.plot.hist(bins=5)
plt.show()

# Mostrando as colunas com dados faltantes
print('Contagem de colunas com ATÉ 20% de dados faltantes', cons[cons.percentual < 0.2].shape[0])
print('Contagem de colunas com 0% de dados faltantes', cons[cons.percentual == 0].shape[0])

# Mantendo somente as colunas sem valores faltantes
cons[cons.percentual == 0]['tipo'].value_counts()
cons['completa'] = ['completa' if x == 0 else 'faltante' for x in cons['percentual']]
mantem = list(cons[cons['completa'] == 'completa']['colunas'])
df = df[mantem]

#Analisando as colunas numericas
colunas_numericas = list(cons[((cons['tipo'] != 'object') &
                              (cons['completa'] == 'completa'))]['colunas'])

# Analise univariavel
for coluna in colunas_numericas:
    print(coluna)
    df[coluna].plot.hist(bins = 50, log= True)
    plt.show()

#Analisando a correlacao entre as variaveis númericas
plt.figure(figsize = (20,20))
sns.heatmap(df[colunas_numericas].corr().round(2), annot= True)

#Variaveis que sao relacionadas com a variavel principal
correlacionadas = ['GarageArea', 'GarageCars', 'GrLivArea', 'OverallQual']

#Removendo a coluna ID
colunas_numericas.remove('Id')
df = df[colunas_numericas]

y_train = df['SalePrice']
X_train = df.drop(columns = 'SalePrice')

#Usando o yelowbrick para visualizacao de dados e suas correlacoes
visualizer = Rank1D(algorithm='shapiro')
visualizer.fit(X_train, y_train)
visualizer.transform(X_train)
visualizer.show()

#Visualizando a correlacoes de features
#visualizer = PCA(scale=True, proj_features=True, projection=2)
#visualizer.fit_transform(X_train[correlacionadas], y_train)
#visualizer.show()
features = list(X_train.columns)


visualizer = FeatureCorrelation(labels=features)
visualizer.fit(X_train, y_train)
visualizer.show()

#Aplicando o modelo machine learning
reg= LinearRegression()
reg.fit(X_train, y_train)
colunas_treinamento = X_train.columns

#Pegando a base de dados de teste
X_test = pd.read_csv('test.csv')
y_test = pd.read_csv('sample_submission.csv')
y_test = y_test['SalePrice']

#Tirando as colunas nulas da base de teste
X_test= X_test[colunas_treinamento].fillna(df[colunas_treinamento].mean())

#Fazendo a predicao
y_pred = reg.predict(X_test)

#Mostrando o erro
erro_normal = mean_squared_error(y_pred=y_pred, y_true=y_test)

# Aplicando o Feature Selection

rfe = RFE(reg)
rfe.fit(X_train, y_train)

#Criando um dataframe com o feature selection
pd.DataFrame({'coluna':X_train.columns,
              'bool': rfe.get_support(),
              'coeficientes': pd.Series(reg.coef_)})

X_train_importante = rfe.transform(X_train)
X_test_importante = rfe.transform(X_test)

#Fazendo o modelo e predicao com os valores apos o feature selection
reg.fit(X_train_importante, y_train)
y_pred_imp = reg.predict(X_test_importante)
erro_imp = mean_squared_error(y_pred=y_pred_imp, y_true=y_test)

# Instantiate RFECV visualizer with a linear SVM classifier
visualizer = RFECV(reg)

visualizer.fit(X_train, y_train) # Fit the data to the visualizer
visualizer.show() # Finalize and render the figure

# Aplicando PCA

pca = PCA(0.95)
pca.fit(X_train)
print(pca.explained_variance_ratio_)

#Fazendo o modelo e predicao com os valores apos o PCA

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
reg = LinearRegression()
reg.fit(X_train_pca, y_train)

y_pred_pca = reg.predict(X_test_pca)
erro_pca = mean_squared_error(y_pred=y_pred_pca, y_true=y_test)

print(pd.DataFrame({'erro' : [erro_normal, erro_imp, erro_pca]}).plot(kind = 'bar', log = True))
