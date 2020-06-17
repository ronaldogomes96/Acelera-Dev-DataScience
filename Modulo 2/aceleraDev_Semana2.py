
#importando pacotes
import pandas as pd
import numpy as np

#Criando um dicionario com os dados
dados = {'canal_de_venda' : ['Facebook', 'Twitter', 'Instagram', 'Linkedin', 'Facebook'],
         'acessos': [100,200,300,400,500],
         'site': ['site1','site1', 'site2', 'site2', 'site3' ],
         'vendas': [1000.52, 1052.34, 2002, 5000, 300]}

#Printa os dados do dicionario na tela
print(dados)

#Verificando o tipo de dicionario
print(type(dados))

#Acessando as chaves do meu dicionario
print(dados.keys())

#Acessando chave especifica
print(dados['canal_de_venda'])

#Acessando posicao especifica de um dicionario
print(dados['canal_de_venda'][2])

#Acessando posicoes especificas de um dicionario com intervalo
print(dados['canal_de_venda'][:2])
