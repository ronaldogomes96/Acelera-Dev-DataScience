import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
athletes = pd.read_csv("athletes.csv")

#Funcao do desafio
def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.

    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.

    Example of numpydoc for those who haven't seen yet.

    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.

    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)

    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)

    return df.loc[random_idx, col_name]

"""
    Considerando uma amostra de tamanho 3000 da coluna height obtida com a função get_sample(), 
    execute o teste de normalidade de Shapiro-Wilk com a função scipy.stats.shapiro(). 
    Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? 
    Responda com um boolean (True ou False).
"""

def q1():

    amostraHeight3000 = get_sample(athletes, "height", 3000)
    testeNormalidadeShapiro = sct.shapiro(amostraHeight3000)
    plt.hist(amostraHeight3000, bins=25)
    plt.show()
    sm.qqplot(amostraHeight3000)
    plt.show()
    print(testeNormalidadeShapiro)
    return False

#q1()

"""
    Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função scipy.stats.jarque_bera().
    Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (True ou False).
"""

def q2():
    amostraHeight3000 = get_sample(athletes, "height", 3000)
    testeNormalidadeBera = sct.jarque_bera(amostraHeight3000)
    print(testeNormalidadeBera)
    return False

#q2()

"""
    Considerando agora uma amostra de tamanho 3000 da coluna weight obtida com a função get_sample(). 
    Faça o teste de normalidade de D'Agostino-Pearson utilizando a função scipy.stats.normaltest(). 
    Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? 
    Responda com um boolean (True ou False).
"""
def q3():
    amostraWeight3000 = get_sample(athletes, "weight", 3000)
    testeNormalidadePearson = sct.normaltest(amostraWeight3000)
    sm.qqplot(amostraWeight3000, fit=True, line="45")
    plt.show()
    print(testeNormalidadePearson)
    return False

#q3()

"""
    Realize uma transformação logarítmica em na amostra de weight da questão 3 e repita o mesmo procedimento. 
    Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? 
    Responda com um boolean (True ou False).
"""

def q4():
    athletes["weight"] = np.log(athletes["weight"])
    amostraWeight3000 = get_sample(athletes, "weight", 3000)
    testeNormalidadePearson = sct.normaltest(amostraWeight3000)
    sm.qqplot(amostraWeight3000, fit=True, line="45")
    plt.show()
    print(testeNormalidadePearson)
    return False

#q4()

"""
    Obtenha todos atletas brasileiros, norte-americanos e canadenses em DataFrames chamados bra, usa e can,respectivamente.
    Realize um teste de hipóteses para comparação das médias das alturas (height) para amostras independentes e 
    variâncias diferentes com a função scipy.stats.ttest_ind() entre bra e usa. 
    Podemos afirmar que as médias são estatisticamente iguais? 
    Responda com um boolean (True ou False).
"""

def q5():
    linhaBra, linhaUsa ,linhaCan = [], [], []
    for i in range(athletes.shape[0]):
        if athletes["nationality"][i] == "BRA":
            linhaBra.append(i)
        elif athletes["nationality"][i] == "USA":
            linhaUsa.append(i)
        elif athletes["nationality"][i] == "CAN":
            linhaCan.append(i)

    bra = athletes.loc[linhaBra]
    usa = athletes.loc[linhaUsa]
    can = athletes.loc[linhaCan]

    testeHipotese = sct.ttest_ind(bra.dropna()["height"], usa.dropna()["height"])
    print(testeHipotese)
    return False

#q5()

"""
    Repita o procedimento da questão 5, mas agora entre as alturas de bra e can.
    Podemos afimar agora que as médias são estatisticamente iguais? 
    Reponda com um boolean (True ou False).
"""

def q6():
    linhaBra, linhaUsa, linhaCan = [], [], []
    for i in range(athletes.shape[0]):
        if athletes["nationality"][i] == "BRA":
            linhaBra.append(i)
        elif athletes["nationality"][i] == "USA":
            linhaUsa.append(i)
        elif athletes["nationality"][i] == "CAN":
            linhaCan.append(i)

    bra = athletes.loc[linhaBra]
    usa = athletes.loc[linhaUsa]
    can = athletes.loc[linhaCan]

    testeHipotese = sct.ttest_ind(bra.dropna()["height"], can.dropna()["height"])
    print(testeHipotese)
    return True

#q6()

"""
    Repita o procedimento da questão 6, mas agora entre as alturas de usa e can.
     Qual o valor do p-valor retornado? 
     Responda como um único escalar arredondado para oito casas decimais.
"""
def q7():
    linhaBra, linhaUsa, linhaCan = [], [], []
    for i in range(athletes.shape[0]):
        if athletes["nationality"][i] == "BRA":
            linhaBra.append(i)
        elif athletes["nationality"][i] == "USA":
            linhaUsa.append(i)
        elif athletes["nationality"][i] == "CAN":
            linhaCan.append(i)

    bra = athletes.loc[linhaBra]
    usa = athletes.loc[linhaUsa]
    can = athletes.loc[linhaCan]

    testeHipotese = sct.ttest_ind(usa.dropna()["height"], can.dropna()["height"])
    return testeHipotese[1].round(8)

print(q7())
