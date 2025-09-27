import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset
url_movies = "https://drive.google.com/uc?export=download&id=17dWfqGAtdKZAR0rTCT6cv7weiIcrNycZ"
movies = pd.read_csv(url_movies)

# Função para extrair o gênero principal
def main_genre_in_list(genre_list):
    if isinstance(genre_list, str):
        try:
            genre_list = json.loads(genre_list)
        except json.JSONDecodeError:
            return None
    if genre_list and isinstance(genre_list, list) and len(genre_list) > 0:
        return genre_list[0].get('name')
    return None

movies['main_genre'] = movies['genres'].apply(main_genre_in_list)

# Função de visualização das distribuições numéricas
def plot_histograms(movies, numeric_cols):
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(movies[col], bins=40, kde=True)
        plt.title(f"Distribuição de {col}")
    plt.tight_layout()
    st.pyplot(plt)

# Função de gráficos de dispersão
def plot_scatter(movies):
    plt.figure(figsize=(14, 8))
    sns.scatterplot(data=movies, x='budget', y='revenue')
    plt.title('Gráfico de Dispersão: Orçamento vs. Receita (Seaborn)')
    plt.xlabel('Orçamento')
    plt.ylabel('Receita')
    st.pyplot(plt)

# Função de boxplots
def plot_boxplots(movies, numeric_cols):
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(y=movies[col])
        plt.title(f"Boxplot de {col}")
    plt.tight_layout()
    st.pyplot(plt)

# Função para o boxplot do gênero principal
def plot_genre_boxplot(movies):
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=movies, x='main_genre', y='vote_average')
    plt.xticks(rotation=90)
    plt.title('Boxplot da Média de Votos por Gênero Principal')
    plt.xlabel('Gênero Principal')
    plt.ylabel('Média de Votos')
    plt.tight_layout()
    st.pyplot(plt)

# Função para o countplot de línguas originais
def plot_language_count(movies):
    plt.figure(figsize=(10, 10))
    sns.countplot(data=movies, y='original_language', order=movies['original_language'].value_counts().index)
    plt.title('Contagem de Filmes por Língua Original')
    plt.xlabel('Contagem')
    plt.ylabel('Língua Original')
    plt.tight_layout()
    st.pyplot(plt)

# Função para treinar o modelo de regressão linear
def train_model(movies):
    X = movies[['budget', 'revenue', 'popularity', 'vote_count']]
    y = movies['vote_average']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2

# Criar a estrutura do app Streamlit
st.set_page_config(page_title="Análise de Filmes", layout="wide")
st.title('Análise de Filmes - TMDB Dataset')

# Sidebar com a navegação
sidebar = st.sidebar.radio('Escolha uma seção', ['Cenário', 'Análises', 'Modelos', 'Conclusões'])

if sidebar == 'Cenário':
    st.subheader('Cenário')
    st.write("""
        Este é um conjunto de dados do TMDB com informações sobre 5000 filmes. O conjunto inclui dados sobre orçamento, receita, popularidade, 
        voto médio, gêneros, língua original e outros atributos relacionados aos filmes. O objetivo é explorar esses dados para identificar padrões
        e criar um modelo preditivo para a média de votos dos filmes.
    """)

elif sidebar == 'Análises':
    st.subheader('Análises')
    
    st.write("### Distribuições das variáveis numéricas")
    plot_histograms(movies, ["budget", "revenue", "popularity", "vote_average"])

    st.write("### Gráfico de Dispersão: Orçamento vs. Receita")
    plot_scatter(movies)

    st.write("### Boxplots das variáveis numéricas")
    plot_boxplots(movies, ["budget", "revenue", "popularity", "vote_average"])

    st.write("### Boxplot da Média de Votos por Gênero Principal")
    plot_genre_boxplot(movies)

    st.write("### Contagem de Filmes por Língua Original")
    plot_language_count(movies)

elif sidebar == 'Modelos':
    st.subheader('Modelos')

    st.write("Treinando um modelo de regressão linear para prever a média de votos com base em características como orçamento, receita, popularidade e contagem de votos.")
    
    mse, rmse, r2 = train_model(movies)
    
    st.write(f"**MSE (Erro Quadrático Médio):** {mse}")
    st.write(f"**RMSE (Raiz do Erro Quadrático Médio):** {rmse}")
    st.write(f"**R² (Coeficiente de Determinação):** {r2}")
    
elif sidebar == 'Conclusões':
    st.subheader('Conclusões')
    st.write("""
        A partir das análises realizadas, obtive as seguintes observações:
        
        Existe uma distribuição bastante concentrada para variáveis como orçamento e popularidade.
        O orçamento e a receita dos filmes possuem uma correlação visível no gráfico de dispersão.
        O gênero do filme tem um impacto na média dos votos, conforme mostrado nos boxplots.
        A língua original dos filmes é predominante em inglês, mas outras línguas como francês e espanhol também estão bem representadas.

        O modelo de regressão linear apresentou um **R²** razoável, indicando que as variáveis preditoras explicam parcialmente a média de votos dos filmes.
    """)
