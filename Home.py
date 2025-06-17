import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import kagglehub

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configuração inicial da página
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
path_dataset = path + '/dataset.csv'
df=pd.read_csv(path_dataset, index_col=0)

# Sidebar - Seleção de seções
section = st.sidebar.radio("Selecione uma etapa da análise:", (
    "1. Visão Geral dos Dados",
    "2. Análise Univariada",
    "3. Correlação entre Variáveis",
    "4. Detecção de Outliers",
    #"5. Pré-processamento",
    #"6. Redução de Dimensionalidade",
    #"7. Clusterização",
    #"8. Avaliação dos Clusters"
))
num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

if section == "1. Visão Geral dos Dados":
    st.subheader("📊 Informações Gerais do Dataset")
    st.write("Nesta etapa vamos ficar mais familiarizados com os dados. Vamos explorar as colunas, tipos de dados, valores ausentes, estatísticas descritivas e visualizar alguns plots.")

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    

    st.markdown("**Visualize o DataFrame com as colunas selecionadas:**")
    cols = st.multiselect("Colunas para exibir:", df.columns.tolist(), default=df.columns[:6].tolist())
    st.dataframe(df[cols].head(15))

    
    st.markdown("**Tipos de dados e valores ausentes:**")
    if st.checkbox("Mostrar dtypes e valores nulos"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(df.dtypes.astype(str))
        with col2:
            st.write(df.isnull().sum())

    st.markdown("**Estatísticas descritivas:**")
    if st.checkbox("Exibir estatísticas descritivas"):
        st.write(df.describe())

    st.markdown("---")
    st.subheader("🧾 Dicionário de Dados: Descrição das Colunas")

    col_descriptions = {
        "track_id": "O ID do Spotify para a faixa.",
        "artists": "Nomes dos artistas que performaram a faixa. Se houver mais de um, são separados por ponto e vírgula.",
        "album_name": "Nome do álbum no qual a faixa aparece.",
        "track_name": "Nome da faixa.",
        "popularity": "Popularidade da faixa (0 a 100), baseada em número e recência de reproduções.",
        "duration_ms": "Duração da faixa em milissegundos.",
        "explicit": "Indica se a faixa possui conteúdo explícito (True = sim, False = não).",
        "danceability": "Quão dançante é a faixa, de 0.0 (menos) a 1.0 (mais dançante).",
        "energy": "Energia percebida da faixa, de 0.0 a 1.0.",
        "key": "Tom da música (0 = Dó, 1 = Dó♯/Ré♭, ..., -1 = indetectável).",
        "loudness": "Volume geral da faixa em decibéis (dB).",
        "mode": "Modalidade: 1 = maior, 0 = menor.",
        "speechiness": "Detecta presença de fala. 1.0 = fala pura; 0.0 = música pura.",
        "acousticness": "Confiança de que a faixa é acústica (0.0 a 1.0).",
        "instrumentalness": "Probabilidade de não conter vocais. Próximo de 1.0 = instrumental.",
        "liveness": "Probabilidade de ter sido gravada ao vivo. Acima de 0.8 = performance ao vivo.",
        "valence": "Quão positiva é a música (0.0 = triste, 1.0 = alegre).",
        "tempo": "Tempo estimado da faixa (batidas por minuto).",
        "time_signature": "Compasso estimado (de 3 a 7).",
        "track_genre": "Gênero musical da faixa."
    }

    # Ordena colunas disponíveis por ordem alfabética
    available_columns = list(col_descriptions.keys())
    selected_columns = st.multiselect("Selecione as colunas que deseja entender melhor:", available_columns)

    if selected_columns:
        for col in selected_columns:
            st.markdown(f"**🔹 {col}**: {col_descriptions[col]}")
    
    st.markdown("---")
    st.subheader("📈 Visualizações Gerais de Distribuição")

    selected_col = st.selectbox("Selecione uma coluna numérica:", num_cols)
    plot_type = st.radio("Tipo de gráfico:", ["Histograma", "Boxplot", "Ambos"])

    fig, axs = plt.subplots(1, 2 if plot_type == "Ambos" else 1, figsize=(14, 4))
    if plot_type == "Histograma" or plot_type == "Ambos":
        ax = axs[0] if plot_type == "Ambos" else axs
        sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue')
        ax.set_title(f"Histograma de {selected_col}")
    if plot_type == "Boxplot" or plot_type == "Ambos":
        ax = axs[1] if plot_type == "Ambos" else axs
        sns.boxplot(x=df[selected_col], ax=ax, color='salmon')
        ax.set_title(f"Boxplot de {selected_col}")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📈 Gráficos de Dispersão entre Variáveis Numéricas")
    if len(num_cols) >= 2:
        x_axis = st.selectbox("Variável no eixo X:", num_cols, index=0)
        y_axis = st.selectbox("Variável no eixo Y:", num_cols, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.5, ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)


elif section == "2. Análise Univariada":
    st.subheader("📈 Análise Univariada Detalhada")

    st.markdown("Selecione uma ou mais variáveis para análise:")
    selected_vars = st.multiselect("Variáveis numéricas:", num_features, default=[num_features[0]])

    scale_option = st.selectbox("Transformação da variável:", ["Nenhuma", "Log (log1p)", "Raiz quadrada (sqrt)"])

    for var in selected_vars:
        st.markdown(f"---\n### 🎯 Análise da variável: `{var}`")

        # Aplicar transformação se necessário
        data = df[var].copy()
        if scale_option == "Log (log1p)":
            data = np.log1p(data)
        elif scale_option == "Raiz quadrada (sqrt)":
            data = np.sqrt(data)

        # Estatísticas descritivas
        st.markdown("**📌 Estatísticas descritivas:**")
        stats = data.describe().to_frame(name=var)
        stats.loc['outliers_above_1.5iqr'] = np.sum(data > (stats.loc['75%'][0] + 1.5 * (stats.loc['75%'][0] - stats.loc['25%'][0])))
        stats.loc['outliers_below_1.5iqr'] = np.sum(data < (stats.loc['25%'][0] - 1.5 * (stats.loc['75%'][0] - stats.loc['25%'][0])))
        st.write(stats)

        # Gráficos lado a lado
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.histplot(data, kde=True, ax=ax1, color='skyblue')
            ax1.set_title(f"Distribuição de {var} ({scale_option})")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=data, ax=ax2, color='lightcoral')
            ax2.set_title(f"Boxplot de {var} ({scale_option})")
            st.pyplot(fig2)

        # Mostrar valores extremos
        st.markdown("**🔎 Valores extremos detectados (1.5 IQR):**")
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(data < lower_bound) | (data > upper_bound)][['track_name', var]]
        st.dataframe(outliers.head(10))

elif section == "3. Correlação entre Variáveis":
    st.subheader("📊 Matriz de Correlação")
    corr_method = st.selectbox("Método de correlação:", ["pearson", "spearman", "kendall"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_features].corr(method=corr_method), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title(f"Correlação - método: {corr_method}")
    st.pyplot(fig)

elif section == "4. Detecção de Outliers":
    st.subheader("🚨 Detecção de Outliers com Isolation Forest")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_features])

    contamination = st.slider("Taxa de contaminação (proporção de outliers):", 0.01, 0.2, 0.05, step=0.01)
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = iso.fit_predict(X_scaled)
    outliers = df[df['anomaly'] == -1]

    st.write(f"Outliers detectados: {len(outliers)} músicas")
    st.dataframe(outliers[['track_name'] + num_features])

    # Gráfico com outliers
    st.subheader("📍 Visualização de Outliers em 2D")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=(df['anomaly'] == -1), cmap='coolwarm', alpha=0.6)
    ax.set_title("Outliers detectados via PCA")
    st.pyplot(fig)
