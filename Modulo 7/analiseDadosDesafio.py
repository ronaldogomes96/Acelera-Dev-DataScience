
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)
from sklearn.feature_extraction.text import( TfidfVectorizer, CountVectorizer)
from sklearn.datasets import fetch_20newsgroups

countries = pd.read_csv("countries.csv")

new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)

#Retirando os espacos nas variaveis Country e Region
imputer = SimpleImputer(strategy='constant', fill_value = "-1")
countries["Climate"] = imputer.fit_transform(countries[["Climate"]])
for i in range(countries.shape[0]):
    countries["Country"][i] = countries["Country"][i].strip()
    countries["Region"][i] = countries["Region"][i].strip()

#Trocando os separadores decimais e tipos de valores
variaveisFloat = ["Pop_density", "Coastline_ratio",
                 "Net_migration","Infant_mortality",
                "Literacy", "Phones_per_1000",
                "Arable", "Crops", "Other", "Climate",
                 "Birthrate", "Deathrate", "Agriculture",
                "Industry", "Service"]
countries[variaveisFloat] = countries[variaveisFloat].apply(lambda x: x.str.replace(',','.')).astype(float)


print(countries.info())

"""
    Quais são as regiões (variável Region) presentes no data set_? 
    Retorne uma lista com as regiões únicas do _data set com os espaços à frente 
    e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e 
    ordenadas em ordem alfabética.
"""

def q1():
    # Retorne aqui o resultado da questão 1.

    regioesUnicas = countries['Region'].unique()
    return sorted(regioesUnicas)
print(q1())

"""
    Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` 
    e estratégia `quantile`, quantos países se encontram acima do 90º percentil? 
    Responda como um único escalar inteiro.
"""

def q2():
    # Retorne aqui o resultado da questão 2
    discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")

    bins = discretizer.fit_transform(countries[["Pop_density"]])
    percentil90 = np.quantile(bins, 0.9)
    paisesAcimaPercentil  = ( bins > percentil90).sum()
    return int(paisesAcimaPercentil)


print(q2())

"""
    Se codificarmos as variáveis Region e "Climate" usando _one-hot encoding_, 
    quantos novos atributos seriam criados? 
    Responda como um único escalar.
"""

def q3():

    antesPadronizacaoDeVariaveis = len(countries.columns)
    one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
    climateDepoisEncoder = one_hot_encoder.fit_transform(countries[["Climate"]])
    regionDepoisEncoder = one_hot_encoder.fit_transform(countries[["Region"]])
    depoisPadronizacaoDeVariaveis = climateDepoisEncoder[0][0] + regionDepoisEncoder[0][0] + 1
    return antesPadronizacaoDeVariaveis - depoisPadronizacaoDeVariaveis
print(q3())

"""
    Aplique o seguinte _pipeline_:
    Preencha as variáveis do tipo int64 e float64 com suas respectivas medianas.
    Padronize essas variáveis.
    Após aplicado o pipeline descrito acima aos dados (somente nas variáveis dos tipos especificados), 
    aplique o mesmo pipeline (ou ColumnTransformer) ao dado abaixo. 
    Qual o valor da variável Arable após o _pipeline_? 
    Responda como um único float arredondado para três casas decimais.
"""
test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

def q4():

    pipeline = Pipeline(steps = [('input_median', SimpleImputer(strategy = 'median')),
                             ('standard_scaler', StandardScaler())])
    pipeline.fit(countries._get_numeric_data())
    n_arable = pipeline.transform([test_country[2:]])[0][countries.columns.get_loc('Arable')-2]
    return n_arable.round(3)

print(q4())


"""
    Descubra o número de outliers da variável Net_migration segundo o método do _boxplot_
    que se encontram no grupo inferior e no grupo superior.
    Você deveria remover da análise as observações consideradas outliers segundo esse método? 
    Responda como uma tupla de três elementos (outliers_abaixo, outliers_acima, removeria?) ((int, int, bool)).
"""

def q5():

    q1 = countries["Net_migration"].quantile(0.25)
    q3 = countries["Net_migration"].quantile(0.75)
    iqr = q3-q1

    outliersAbaixo = []
    outliersAcima = []

    limiteInferior = q1 - ( 1.5 * iqr )
    limiteSuperior = q3 + ( 1.5 * iqr )

    for i in range(countries.shape[0]):
        if countries["Net_migration"][i] < limiteInferior:
            outliersAbaixo.append(countries["Net_migration"][i])
        elif countries["Net_migration"][i] > limiteSuperior:
            outliersAcima.append(countries["Net_migration"][i])

    quantidadeOutliers = (len(outliersAbaixo), len(outliersAcima), False)


    return quantidadeOutliers

print(q5())

"""
    Para as questões 6 e 7 utilize a biblioteca fetch_20newsgroups de datasets de test do sklearn
    Considere carregar as seguintes categorias e o dataset newsgroups:
    categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
    newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    Aplique CountVectorizer ao data set newsgroups e descubra o número de vezes que a palavra phone aparece no corpus. 
    Responda como um único escalar.
"""

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)

def q6():

    vetorize = CountVectorizer()
    phoneAparece = vetorize.fit_transform(newsgroup.data)[:,vetorize.vocabulary_['phone']].sum()
    return phoneAparece

print(q6())

"""
    Aplique TfidfVectorizer ao data set newsgroups e descubra o TF-IDF da palavra phone.
    Responda como um único escalar arredondado para três casas decimais.
"""

def q7():

    tfVetorize = TfidfVectorizer()
    tfIdPhone = tfVetorize.fit_transform(newsgroup.data)[:,tfVetorize.vocabulary_['phone']].sum().round(3)
    return tfIdPhone

print(q7())