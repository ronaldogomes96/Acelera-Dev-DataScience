
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from IPython.core.pylabtools import figsize

#Configuracoes do matplot
figsize(12, 8)

sns.set()

np.random.seed(42)

#Criacao do data frame
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                          "binomial": sct.binom.rvs(100, 0.2, size=10000)})

#Plot dos dois graficos
sns.distplot(dataframe["normal"])
plt.show()

sns.boxplot(dataframe["binomial"])
plt.show()


#Respostas das questoes do desafio

"""
    Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis normal e binomial de dataframe? 
    Responda como uma tupla de três elementos arredondados para três casas decimais.
    Em outra palavras, sejam q1_norm, q2_norm e q3_norm os quantis da variável normal e q1_binom, q2_binom e q3_binom os quantis da variável binomial,
    qual a diferença (q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)?
"""
def q1():
    q1_norm = np.quantile(dataframe["normal"], .25)
    q2_normal = np.quantile(dataframe["normal"], .50)
    q3_normal = np.quantile(dataframe["normal"], .75)

    q1_binom = np.quantile(dataframe["binomial"], .25)
    q2_binom = np.quantile(dataframe["binomial"], .50)
    q3_binom = np.quantile(dataframe["binomial"], .75)

    q1 = (q1_norm-q1_binom).round(3)
    q2 = (q2_normal-q2_binom).round(3)
    q3 = (q3_normal-q3_binom).round(3)

    return (q1, q2, q3)

print(q1())

"""
    Considere o intervalo  [𝑥¯−𝑠,𝑥¯+𝑠] , onde  𝑥¯ é a média amostral e  𝑠 s é o desvio padrão. 
    Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável normal? 
    Responda como uma único escalar arredondado para três casas decimais.
"""

def q2():

    intervaloQuartilicoNormal = [dataframe["normal"].mean() - dataframe["normal"].std(), dataframe["normal"].mean() + dataframe["normal"].std()]
    probabilidadeMinima = sct.norm.cdf( intervaloQuartilicoNormal[0], 20, 4)
    probabilidadeMaxima = sct.norm.cdf( intervaloQuartilicoNormal[1], 20, 4)
    probabilidadeEntre = probabilidadeMaxima - probabilidadeMinima

    return probabilidadeEntre.round(3)

print(q2())


"""
    Qual é a diferença entre as médias e as variâncias das variáveis binomial e normal? 
    Responda como uma tupla de dois elementos arredondados para três casas decimais.
    Em outras palavras, sejam m_binom e v_binom a média e a variância da variável binomial, e m_norm e v_norm a média e 
    a variância da variável normal. Quais as diferenças (m_binom - m_norm, v_binom - v_norm)?
"""

def q3():
    m_binom = dataframe["binomial"].mean()
    v_binom = dataframe["binomial"].var()

    m_norm = dataframe["normal"].mean()
    v_norm = dataframe["normal"].var()

    media = m_binom - m_norm
    variancia = v_binom - v_norm

    return (media.round(3), variancia.round(3))

print(q3())

#Parte 2

#Configuracao do dataset
stars = pd.read_csv("pulsar_stars.csv")
stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)
stars.loc[:, "target"] = stars.target.astype(bool)
false_pulsar_mean_profile_standardized = []

"""
Considerando a variável mean_profile de stars:
Filtre apenas os valores de mean_profile onde target == 0 (ou seja, onde a estrela não é um pulsar).
Padronize a variável mean_profile filtrada anteriormente para ter média 0 e variância 1.
Chamaremos a variável resultante de false_pulsar_mean_profile_standardized.
Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função norm.ppf() disponível em scipy.stats.
Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável false_pulsar_mean_profile_standardized? 
Responda como uma tupla de três elementos arredondados para três casas decimais.
"""

def q4():

    #Filtre apenas os valores de mean_profile onde target == 0
    mean_profile = []
    for i in range(stars.shape[0]):
        if stars["target"][i] == False:
            mean_profile.append(stars["mean_profile"][i])

    #Padronize a variável mean_profile filtrada anteriormente para ter média 0 e variância 1.
    media = np.mean(mean_profile)
    desvioPadrao = np.std(mean_profile)
    for i in range(len(mean_profile)):
        false_pulsar_mean_profile_standardized.append((mean_profile[i] - media) / desvioPadrao)

    #Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
    q80 = sct.norm.ppf(0.80, 0, 1)
    q90 = sct.norm.ppf(0.90, 0, 1)
    q95 = sct.norm.ppf(0.95, 0, 1)

    #Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`?
    #Responda como uma tupla de três elementos arredondados para três casas decimais.
    probabilidadeq80  = sct.norm.cdf(q80, 0, 1)
    probabilidadeq90 = sct.norm.cdf(q90, 0, 1)
    probabilidadeq95 = sct.norm.cdf(q95, 0, 1)

    return  (probabilidadeq80.round(3), probabilidadeq90.round(3), probabilidadeq95.round(3))

print(q4())

"""
    Qual a diferença entre os quantis Q1, Q2 e Q3 de false_pulsar_mean_profile_standardized 
    e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? 
    Responda como uma tupla de três elementos arredondados para três casas decimais.
"""

def q5():

    #Quantis teoricos de distribuicao normal
    qNormal25 = sct.norm.ppf(0.25, 0, 1)
    qNormal50 = sct.norm.ppf(0.50, 0, 1)
    qNormal75 = sct.norm.ppf(0.75, 0, 1)

    #Quantis da variavel false_pulsar_mean_profile_standardized
    qStar25 = np.quantile(false_pulsar_mean_profile_standardized, .25)
    qStar50 = np.quantile(false_pulsar_mean_profile_standardized, .50)
    qStar75 = np.quantile(false_pulsar_mean_profile_standardized, .75)

    #Diferenca
    q25 = qStar25 - qNormal25
    q50 = qStar50 - qNormal50
    q75 = qStar75 - qNormal75

    return (q25.round(3), q50.round(3), q75.round(3))

print(q5())
















