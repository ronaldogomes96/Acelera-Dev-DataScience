from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA
from loguru import logger
from IPython.core.pylabtools import figsize
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

#Analise inicial
fifa = pd.read_csv("fifa.csv")
columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")

fifa.dropna(inplace= True)

"""
    Qual fração da variância consegue ser explicada pelo primeiro componente principal de fifa? 
    Responda como um único float (entre 0 e 1) arredondado para três casas decimais.
"""

def q1():

    pca = PCA(n_components = 1)
    project = pca.fit(fifa)
    varianciaExplicada = project.explained_variance_ratio_[0]
    return varianciaExplicada.round(3)

q1()

"""
    Quantos componentes principais precisamos para explicar 95% da variância total?
    Responda como un único escalar inteiro.
"""

def q2():

    pca095 = PCA(n_components= 0.95)
    project = pca095.fit_transform(fifa)
    numeroComponentesPrincipais = project.shape[1]
    return numeroComponentesPrincipais

q2()

"""
    Qual são as coordenadas (primeiro e segundo componentes principais) do ponto x abaixo? 
    O vetor abaixo já está centralizado. Cuidado para não centralizar o vetor novamente
    (por exemplo, invocando PCA.transform() nele).
    Responda como uma tupla de float arredondados para três casas decimais.
"""

x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]

def q3():

    pca = PCA().fit(fifa)
    c1,c2 = pca.components_.dot(x)[0:2].round(3)
    return c1,c2

q3()


"""
    Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma.
    Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.
"""

def q4():

    x = fifa.drop(columns="Overall")
    y = fifa["Overall"]

    rfe = RFE(estimator= LinearRegression(), n_features_to_select= 5)
    rfe.fit(x,y)

    indexFeatureSelect = rfe.get_support(indices=True)

    featureSelect = list(x.columns[indexFeatureSelect])

    return featureSelect

q4()



