
import streamlit as st

def main():
    st.title("Hello World") #Titulo
    st.header("This is a header")
    st.subheader("Isso é uma subheader")
    st.text("Isso é um texto")
    #st.image("Nome da imagem")
    #st.audio("nome do audio")
    #st.video("nome do video")
    
def comandosDeInterface():
    
    #criando um botao
    st.markdown("Botao")
    botao = st.button("Botao")
    if botao:
        st.markdown("Clicado")
        
    #Criando uma caixa de checkbox
    st.markdown("Checkbox")
    check = st.checkbox("Checkbox")
    if check:
        st.markdown("Clicado")
        
    #Criando opcoes radios
    st.markdown("Radius")
    radio = st.radio("Escolha as opcoes", ("Op1", "Op2"))
    if radio == "Op1":
        st.markdown("Op1")
    if radio == "Op2":
        st.markdown("Op2")
        
    #Criando uma caixa de selecao
    st.markdown("Select box")
    select = st.selectbox("Escolha a box", ("Op1", "Op2"))
    if select == "Op1":
        st.markdown("Op1")
    if select == "Op2":
        st.markdown("Op2")
    
    #Criando um selecionador de arquivos
    st.markdown("File Uploder")
    file = st.file_uploader("Chose your file", type = "csv")
    if file is not None:
        st.markdown("Arquivo nao vazio")
    
    
if __name__ == "__main__":
    comandosDeInterface()

