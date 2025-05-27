# Importa bibliotecas necessárias
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Configuração da página do aplicativo Streamlit
st.set_page_config(
    page_title = "Classificação de Veiculos",
    layout="wide"
)

# Função que carrega os dados e treina o modelo
@st.cache_data  # Cacheia os dados e modelo para evitar recarregamento desnecessário
def load_data_and_model():
    carros = pd.read_csv("car.csv", sep= ",")  # Lê o dataset de veículos
    enconder = OrdinalEncoder()  # Codificador para transformar categorias em números

    # Converte todas as colunas, exceto a de classe, para tipo categórico
    for col in carros.columns.drop('class'):
        carros[col] = carros[col].astype('category')

    # Codifica os dados (exceto a classe) em valores numéricos
    X_enconded = enconder.fit_transform(carros.drop('class', axis = 1))

    # Codifica os valores da coluna 'class' como números (0, 1, 2, ...)
    y = carros['class'].astype('category').cat.codes

    # Divide os dados em treino (70%) e teste (30%)
    X_train , X_test , y_train , y_teste = train_test_split(X_enconded, y, test_size= 0.3 , random_state = 42 )

    # Cria o modelo Naive Bayes para dados categóricos
    modelo = CategoricalNB()
    modelo.fit(X_train, y_train)  # Treina o modelo com os dados de treino

    # Faz previsões com os dados de teste
    y_pred = modelo.predict(X_test)

    # Calcula a acurácia do modelo
    accuracia = accuracy_score(y_teste , y_pred)

    # Retorna o codificador, modelo treinado, acurácia e o dataframe original
    return enconder, modelo , accuracia , carros 


# Executa a função acima e obtém os resultados
enconder, modelo , accuracia , carros = load_data_and_model()

# Título da página
st.title(" Previsão de qualidade de Veiculo ")

# Mostra a acurácia do modelo na interface
st.write(f"Acuracia do modelo: {accuracia: .2f}")

# Coleta as entradas do usuário com base nos valores únicos de cada atributo
input_features = [
    st.selectbox("Preço: ", carros['buying'].unique()),
    st.selectbox("Manutenção: ", carros['maint'].unique()),
    st.selectbox("Portas: ", carros['doors'].unique()),
    st.selectbox("Capacidade de Passageiros: ", carros['persons'].unique()),
    st.selectbox("Porta Malas: ", carros['lug_boot'].unique()),
    st.selectbox("Segurança: ", carros['safety'].unique()),
]

# Quando o botão "Processar" é clicado
if st.button("Processar"):
    # Cria um dataframe com os dados inseridos pelo usuário
    input_df = pd.DataFrame([input_features], columns=carros.columns.drop('class'))

    # Codifica os dados com o mesmo encoder usado no treino
    input_enconded = enconder.transform(input_df)

    # Realiza a previsão com o modelo treinado
    predict_enconded = modelo.predict(input_enconded)

    # Recupera o nome da classe prevista a partir do valor codificado
    previsao = carros['class'].astype('category').cat.categories[predict_enconded][0]

    # Exibe o resultado da previsão
    st.header(f"Resultado da previsão: {previsao}")
