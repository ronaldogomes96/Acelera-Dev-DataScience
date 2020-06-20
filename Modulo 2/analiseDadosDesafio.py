
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


black_friday = pd.read_csv("black_friday.csv")


def q1():
    linhas = black_friday.shape[0]
    colunas = black_friday.shape[1]
    linhasColunas = (linhas, colunas)
    return linhasColunas

def q2():
    mulheres = 0
    for i in range(black_friday.shape[0]):
        if black_friday['Gender'][i] == 'F' and black_friday['Age'][i] == '26-35':
            mulheres += 1
    return mulheres

def q3():

    usuariosUnicos = black_friday['User_ID'].unique()
    return len(usuariosUnicos)

def q4():

    return black_friday.dtypes.nunique()

def q5():

    faltantesPorColuna = black_friday.isna().sum() / black_friday.shape[0]
    porcentagemValoresFaltantes = faltantesPorColuna.sum() / faltantesPorColuna.shape[0]
    return faltantesPorColuna[10]

def q6():

    faltantes = black_friday.isna().sum()
    return faltantes.max()

def q7():

    valorMaisFrequente = black_friday['Product_Category_3'].mode()
    return valorMaisFrequente[0]

def q8():

    scaler = StandardScaler()
    black_friday['Purchase'] = scaler.fit_transform(black_friday['Purchase'].values.reshape(-1,1))
    mediaColunaPurchase = black_friday['Purchase'].mean()
    return mediaColunaPurchase

def q9():

    return black_friday.shape[0]

def q10():

    return False


if __name__ == '__main__':

    black_friday.info()
    #print(black_friday.shape)
    print(q1())

    #print(black_friday)
    #print(q2())

    print(q3())

    print(q4())

    print(q5())

    print(q6())

    print(q7())

    print(q8())

    print(q9())

    print(q10())

