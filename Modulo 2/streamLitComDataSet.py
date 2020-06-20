
import streamlit as st
import pandas as pd

def main():

    st.title('Usando streamlit com pandas')
    #Importando algum datset
    file = st.file_uploader('Uploder your file', type = 'csv')

    #Caso esse arquivo seja valido
    if file is not None:
        #Criando um slider, mostrando o dataframe de acordo com o valor do slider
        slider = st.slider('Valores', 1,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))

        #Mostrando de acordo com o table
        st.markdown("Table")
        st.table(df.head(slider))

        #Escrevendo algo na tela
        st.write(df.columns)
        st.table(df.groupby('species')['pental_width'].mean())



if __name__ == '__main__':
    main()