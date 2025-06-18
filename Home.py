import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import kagglehub

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    path_dataset = path + '/dataset.csv'
    df = pd.read_csv(path_dataset, index_col=0)
    if 'track_genre' not in df.columns:
        df['track_genre'] = 'unknown'
    return df

df = load_data()

num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()


def pagina_1_visao_geral(df):
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
    x_axis = st.selectbox("Variável no eixo X:", num_features, index=0)
    y_axis = st.selectbox("Variável no eixo Y:", num_features, index=1)
    show_trend = st.checkbox("Mostrar linha de tendência (regressão linear)")
    

    df_plot = df[[x_axis, y_axis]].dropna()

    fig, ax = plt.subplots()
    if show_trend:
        sns.regplot(data=df_plot, x=x_axis, y=y_axis, ax=ax,
                    scatter_kws={'alpha': 0.5, 'color': 'red'},
                    line_kws={"color": "blue"})
    else:
        sns.scatterplot(data=df_plot, x=x_axis, y=y_axis, alpha=0.5, color='red', ax=ax)

    ax.set_title(f"Dispersão entre {x_axis} e {y_axis}")
    st.pyplot(fig)
    st.markdown("📌 Danceability vs. Energy")
    st.markdown("- Já esperamnos uma correlação positiva entre 'danceability' e 'energy', pois músicas mais dançantes tendem a ter mais energia.  \n"
                "- A linha de tendência (regressão linear) ajuda a visualizar uma correlação moderadamente positiva. \n"
                "\n" 
                "📌 Acousticness vs. Energy \n"
                "- Correlação negativa forte esperada, pois músicas acústicas são menos energéticas \n"
                "- A linha de tendência decrescemte indica uma relação inversamente proporcional. \n"
                "\n"
                "📌 Loudness vs. Energy \n"
                "- Baixa dispersão e ascendência dos pontos mostram uma correlação fortemente positiva (músicas energéticas costumam ser mais altas) \n")

def pagina_2_analise_univariada(df):
    st.subheader("🔬 Análise Univariada Detalhada")
    st.markdown("Explore a distribuição de cada variável. Use os filtros para comparar diferentes gêneros e ajuste os gráficos para uma análise mais profunda.")

    st.markdown("### 🎭 Comparar Distribuições por Gênero")
    genres_to_compare = st.multiselect(
        "Selecione um ou mais gêneros para comparar (opcional):",
        sorted(df['track_genre'].unique().tolist())
    )

    if genres_to_compare:
        df_filtered = df[df['track_genre'].isin(genres_to_compare)]
        hue_on = 'track_genre'
    else:
        df_filtered = df.copy()
        hue_on = None

    st.markdown("### ⚙️ Controles da Análise")
    selected_var = st.selectbox("Selecione uma variável numérica para análise:", num_features)

    num_bins = st.slider("Número de Bins para o Histograma:", min_value=10, max_value=100, value=30)
    
    st.markdown(f"---\n### 🎯 Análise da variável: `{selected_var}`")

    st.markdown("**📊 Visualização da Distribuição:**")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(data=df_filtered, x=selected_var, hue=hue_on, kde=True, ax=ax1, bins=num_bins, palette='viridis')
        ax1.set_title(f"Distribuição de {selected_var}")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df_filtered, x=selected_var, y=hue_on, ax=ax2, palette='viridis', orient='h')
        ax2.set_title(f"Boxplot de {selected_var}")
        st.pyplot(fig2)

    st.markdown(f"**🔎 Estatísticas e Outliers para `{selected_var}`**")
    if genres_to_compare:
        st.write("Estatísticas descritivas por gênero selecionado:")
        st.write(df_filtered.groupby('track_genre')[selected_var].describe().T)
    else:
        st.write("Estatísticas descritivas gerais:")
        st.write(df_filtered[selected_var].describe())
    
    if st.checkbox(f"Mostrar outliers (músicas com valores extremos) para '{selected_var}'"):
        q1 = df_filtered[selected_var].quantile(0.25)
        q3 = df_filtered[selected_var].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df_filtered[(df_filtered[selected_var] < lower_bound) | (df_filtered[selected_var] > upper_bound)]
        
        if outliers.empty:
            st.success("Não foram encontrados outliers com base no critério de 1.5 * IQR. ✨")
        else:
            st.write(f"Foram encontrados **{len(outliers)}** outliers:")
            st.dataframe(outliers[['track_name', 'artists', 'track_genre', selected_var]].sort_values(by=selected_var, ascending=False))

def pagina_3_correlacao(df):
    st.subheader("↔️ Análise de Correlação")
    st.markdown("Investigue a relação entre as variáveis. Use o filtro de gênero e o slider de intensidade para focar nas correlações mais importantes.")
    
    st.markdown("### 🎵 Filtrar por Gênero")
    genre_list = ['Todos'] + sorted(df['track_genre'].unique().tolist())
    corr_genre = st.selectbox("Selecione um gênero para calcular a correlação:", genre_list)

    if corr_genre == 'Todos':
        df_corr = df[num_features]
    else:
        df_corr = df[df['track_genre'] == corr_genre][num_features]
        st.info(f"Mostrando correlações apenas para o gênero: **{corr_genre}**")

    st.markdown("### ⚙️ Controles do Heatmap")
    corr_method = st.selectbox("Método de correlação:", ["pearson", "spearman", "kendall"])
    
    corr_threshold = st.slider("Ocultar no gráfico e na tabela correlações com valor absoluto abaixo de:", 0.0, 1.0, 0.0, 0.05)
    
    if not df_corr.empty and len(df_corr) > 1:
        corr_matrix = df_corr.corr(method=corr_method)
        
        mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))
        mask_threshold = np.abs(corr_matrix) < corr_threshold
        mask_heatmap = mask_upper | mask_threshold
        
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr_matrix, mask=mask_heatmap, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 8}, vmin=-1, vmax=1)
        ax.set_title(f"Mapa de Calor (Método: {corr_method.capitalize()}, Gênero: {corr_genre})", fontsize=16)
        st.pyplot(fig)

        st.markdown("### ✨ Pares com Maior Correlação")
        st.write(f"Abaixo estão os pares de variáveis com correlação absoluta acima de **{corr_threshold}** (excluindo duplicatas e auto-correlações).")
        
        mask_table = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_unstacked = corr_matrix.where(mask_table).stack()
        
        strong_pairs = corr_unstacked.sort_values(key=abs, ascending=False)
        
        strong_pairs = strong_pairs[abs(strong_pairs) > corr_threshold]

        if strong_pairs.empty:
            st.warning("Nenhum par encontrado acima do threshold. Tente um valor menor.")
        else:
            st.dataframe(strong_pairs.to_frame(name='correlation_value').head(20))

    else:
        st.warning(f"Não há dados suficientes para o gênero '{corr_genre}' para calcular a correlação.")

def pagina_4_outliers(df):
    st.subheader("🚨 Detecção de Outliers com Isolation Forest")
    st.markdown("""
    Nesta seção, vamos identificar **outliers**: músicas que possuem características muito diferentes da maioria. Um outlier pode ser uma música experimental, um erro nos dados ou simplesmente uma faixa única.
    
    Usaremos o algoritmo **Isolation Forest**, que é eficiente em detectar anomalias em dados multidimensionais. Ele funciona isolando observações ao selecionar aleatoriamente uma feature e, em seguida, um valor de divisão aleatório entre os valores máximo e mínimo da feature selecionada.
    """)

    st.markdown("### ⚙️ Controles da Detecção")
    features_for_outliers = st.multiselect(
        "Selecione as features para a detecção de outliers:", 
        num_features, 
        default=['danceability', 'energy', 'loudness', 'acousticness', 'valence']
    )
    
    if not features_for_outliers:
        st.warning("Por favor, selecione ao menos uma feature para a análise.")
        return

    contamination = st.slider(
        "Taxa de contaminação (proporção de outliers):", 
        0.01, 0.2, 0.05, step=0.01,
        help="Este valor representa a proporção esperada de outliers no dataset. Um valor maior resultará em mais músicas sendo classificadas como anomalias."
    )

    st.markdown("---")
    st.markdown("### 🎼 Músicas Anômalas Detectadas")

    df_outlier_analysis = df.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_outlier_analysis[features_for_outliers])

    iso = IsolationForest(contamination=contamination, random_state=42)
    df_outlier_analysis['anomaly'] = iso.fit_predict(X_scaled)
    outliers = df_outlier_analysis[df_outlier_analysis['anomaly'] == -1]

    st.write(f"Com base nas suas configurações, foram detectadas **{len(outliers)}** músicas como outliers.")
    st.dataframe(outliers[['track_name', 'artists', 'track_genre'] + features_for_outliers])

    if len(features_for_outliers) >= 2:
        st.subheader("📍 Visualização de Outliers em 2D (usando PCA)")
        st.markdown("""
        Para visualizar os outliers em um gráfico 2D, reduzimos a dimensionalidade das features selecionadas usando **Análise de Componentes Principais (PCA)**. Os pontos em vermelho representam as músicas marcadas como outliers.
        """)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'], index=df_outlier_analysis.index)
        df_pca['anomaly'] = df_outlier_analysis['anomaly']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=df_pca, x='PC1', y='PC2', hue='anomaly', 
            palette={1: 'blue', -1: 'red'}, style='anomaly',
            markers={1: '.', -1: 'X'}, s=100, alpha=0.7, ax=ax
        )
        ax.set_title("Outliers detectados via PCA")
        ax.legend(['Normal', 'Outlier'])
        st.pyplot(fig)
    else:
        st.info("Selecione 2 ou mais features para visualizar o gráfico de dispersão com PCA.")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def pagina_5_preprocessamento(df):
    st.subheader("⚙️ 5. Pré-processamento dos Dados")
    st.markdown("""
    O pré-processamento é uma etapa fundamental na preparação de dados para modelos de Machine Learning. Aqui, transformaremos nossas features para que os algoritmos possam interpretá-las da melhor forma possível.
    
    Vamos abordar três etapas principais:
    1.  **Imputação de Dados**: Substituir valores ausentes.
    2.  **Feature Scaling**: Padronizar as escalas das nossas variáveis numéricas.
    3.  **One-Hot Encoding**: Converter variáveis categóricas em um formato numérico.
    """)

    df_processed = df.copy()

    st.markdown("---")
    st.markdown("### 1. Tratamento de Valores Ausentes")
    st.markdown("""
    Antes de escalar os dados, precisamos lidar com valores ausentes (NaN). Uma estratégia comum e robusta é a **imputação pela mediana**, onde substituímos os valores ausentes pelo valor central da coluna. Isso é menos sensível a outliers do que usar a média.
    """)
    
    # Preencher NaNs com a mediana das colunas numéricas
    numeric_cols_with_na = df_processed[num_features].columns[df_processed[num_features].isnull().any()].tolist()
    if numeric_cols_with_na:
        st.write("Valores ausentes encontrados nas seguintes colunas e preenchidos com a mediana:")
        st.write(df_processed[numeric_cols_with_na].isnull().sum().to_frame(name='NAs Preenchidos'))
        df_processed.fillna(df_processed.median(numeric_only=True), inplace=True)
    else:
        st.success("Nenhum valor ausente encontrado nas colunas numéricas. ✅")
    

    st.markdown("---")
    st.markdown("### 2. Feature Scaling (Padronização)")
    st.markdown("""
    Algoritmos de clusterização, como o K-Means, são sensíveis à escala das features. Variáveis com escalas maiores (como `duration_ms`) podem dominar o processo de agrupamento.
    Usaremos o **StandardScaler**, que transforma os dados para que tenham média 0 e desvio padrão 1.
    """)

    features_to_scale = st.multiselect(
        "Selecione as features numéricas para padronizar:",
        options=num_features,
        default=num_features
    )
    
    if features_to_scale:
        scaler = StandardScaler()
        df_processed[features_to_scale] = scaler.fit_transform(df_processed[features_to_scale])

        st.markdown("**Comparação: Antes vs. Depois da Padronização** (para a feature `danceability`)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Antes:")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df['danceability'], kde=True, ax=ax, color='blue')
            ax.set_title("Original")
            st.pyplot(fig)
        with col2:
            st.write("Depois:")
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df_processed['danceability'], kde=True, ax=ax, color='green')
            ax.set_title("Padronizado")
            st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### 3. One-Hot Encoding para Gêneros")
    st.markdown("""
    Para usar a feature `track_genre` em nosso modelo, precisamos convertê-la de texto para um formato numérico. O **One-Hot Encoding** cria novas colunas para cada gênero, marcando com `1` se a música pertence àquele gênero e `0` caso contrário.
    """)
    
    if st.checkbox("Aplicar One-Hot Encoding na coluna 'track_genre'?", value=True):
        df_processed = pd.get_dummies(df_processed, columns=['track_genre'], prefix='genre')
        st.success(f"One-Hot Encoding aplicado! Novas colunas de gênero foram criadas.")
    
    st.markdown("---")
    st.markdown("### 🏁 DataFrame Final Pré-processado")
    st.markdown("Abaixo está uma amostra do nosso dataset após as transformações. Este é o conjunto de dados que usaremos para a clusterização.")
    
    final_features = df_processed.select_dtypes(include=np.number).columns.tolist()
    final_df = df_processed[final_features]

    st.dataframe(final_df.head())
    st.write(f"O dataset final possui **{final_df.shape[0]}** linhas e **{final_df.shape[1]}** features.")

    st.markdown("### 💾 Baixar Dados Processados")
    st.markdown("Clique no botão para baixar o DataFrame processado em um arquivo CSV para uso posterior.")
    
    csv = convert_df_to_csv(final_df)
    st.download_button(
       label="Baixar dados como CSV",
       data=csv,
       file_name='processed_spotify_data.csv',
       mime='text/csv',
    )
    if st.button("Salvar DataFrame em cache para próximas etapas"):
        st.session_state['processed_df'] = final_df
        st.success("DataFrame processado salvo na sessão! ✅")

def pagina_6_reducao_dimensionalidade(df):
    st.subheader("📉 6. Redução de Dimensionalidade")
    st.markdown("""
    Com um grande número de features, pode ser difícil visualizar e modelar os dados. A **Redução de Dimensionalidade** nos ajuda a "comprimir" as informações mais importantes em um número menor de componentes.
    
    Usaremos a **Análise de Componentes Principais (PCA)**, uma técnica popular que encontra novas eixos (componentes) que maximizam a variância nos dados.
    """)
    
    if 'processed_df' not in st.session_state or st.session_state['processed_df'] is None:
        st.warning("Por favor, execute o pré-processamento na página '5. Pré-processamento' e clique em 'Salvar DataFrame' antes de continuar.")
        return

    processed_df = st.session_state['processed_df']
    
    st.markdown("### ⚙️ Configurando o PCA")
    n_components = st.slider(
        "Número de componentes principais para gerar:",
        min_value=2, max_value=20, value=10,
        help="Escolha quantos componentes (novas features) você deseja criar. Começar com 10 a 15 é geralmente um bom ponto de partida."
    )
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(processed_df)

    st.markdown("### 📊 Variância Explicada")
    st.markdown("O gráfico abaixo mostra quanta da variância original dos dados é 'capturada' por cada componente principal. O ideal é que os primeiros componentes capturem a maior parte da informação.")

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, n_components + 1), explained_variance, alpha=0.6, color='b', label='Variância Individual')
    ax.plot(range(1, n_components + 1), cumulative_variance, 'r-o', label='Variância Cumulativa')
    ax.set_xlabel('Componentes Principais')
    ax.set_ylabel('Proporção da Variância Explicada')
    ax.set_title('Variância Explicada pelos Componentes Principais')
    ax.legend(loc='best')
    ax.set_xticks(range(1, n_components + 1))
    st.pyplot(fig)

    st.info(f"Com **{n_components}** componentes, conseguimos explicar **{cumulative_variance[-1]:.2%}** da variância total dos dados.")
    
    st.markdown("### 💿 Dados Transformados pelo PCA")
    st.markdown("Abaixo está o nosso dataset transformado, agora com um número reduzido de dimensões. Estes serão os dados que usaremos para a clusterização.")
    
    df_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components)])
    st.dataframe(df_pca.head())

    if st.button("Salvar dados do PCA para próximas etapas"):
        st.session_state['pca_df'] = df_pca
        st.success("Dados transformados pelo PCA salvos na sessão! ✅")


def pagina_7_clusterizacao(df):
    st.subheader("🧩 7. Clusterização")
    st.markdown("""
    Agora que nossos dados estão preparados, vamos aplicar algoritmos de **clusterização** para encontrar grupos (clusters) de músicas com características semelhantes. O objetivo é descobrir "playlists" naturais escondidas nos dados.
    """)

    if 'pca_df' not in st.session_state or st.session_state['pca_df'] is None:
        st.warning("Por favor, execute a Redução de Dimensionalidade na página '6. Redução de Dimensionalidade' e salve os dados antes de continuar.")
        return
    
    X_data = st.session_state['pca_df']

    st.markdown("### 🤖 Escolha do Algoritmo de Clusterização")
    algo_choice = st.selectbox(
        "Selecione o algoritmo:",
        ["K-Means", "DBSCAN", "Clustering Aglomerativo"]
    )

    model = None
    labels = None

    if algo_choice == "K-Means":
        st.markdown("""
        **K-Means** é um dos algoritmos mais populares. Ele agrupa os dados tentando separar as amostras em *k* grupos de variância igual, minimizando um critério conhecido como inércia. Você precisa definir o número de clusters (k) antecipadamente.
        """)
        k = st.slider("Número de clusters (k):", min_value=2, max_value=20, value=8)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    elif algo_choice == "DBSCAN":
        st.markdown("""
        **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) agrupa pontos que estão densamente compactados, marcando como outliers os pontos que estão sozinhos em regiões de baixa densidade. É ótimo para encontrar clusters de formas arbitrárias e não exige que você defina o número de clusters.
        """)
        eps = st.slider("Epsilon (eps - raio da vizinhança):", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
        min_samples = st.slider("Número Mínimo de Amostras (min_samples):", min_value=1, max_value=50, value=10)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    elif algo_choice == "Clustering Aglomerativo":
        st.markdown("""
        O **Clustering Aglomerativo** realiza uma clusterização hierárquica. Ele começa tratando cada ponto como um cluster separado e, em seguida, mescla recursivamente os pares de clusters mais próximos até que um certo número de clusters seja alcançado.
        """)
        n_clusters_agg = st.slider("Número de clusters:", min_value=2, max_value=20, value=8)
        model = AgglomerativeClustering(n_clusters=n_clusters_agg)

    if st.button(f"Executar {algo_choice}"):
        with st.spinner("Clusterizando os dados... Isso pode levar um momento."):
            labels = model.fit_predict(X_data)
            st.session_state['cluster_labels'] = labels
            st.session_state['cluster_data'] = X_data
            st.success(f"Clusterização com {algo_choice} concluída! Os resultados foram salvos.")
            
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            st.write(f"Número de clusters encontrados: **{n_clusters_found}**")
            if -1 in labels:
                noise_points = np.sum(labels == -1)
                st.write(f"Número de pontos de ruído (outliers): **{noise_points}**")

def pagina_8_avaliacao_clusters(df):
    st.subheader("🏆 8. Avaliação dos Clusters")
    st.markdown("""
    Como saber se os clusters que encontramos são bons? Nesta etapa, vamos usar métricas quantitativas e visualizações para avaliar a qualidade dos nossos agrupamentos.
    """)

    if 'cluster_labels' not in st.session_state or st.session_state['cluster_labels'] is None or \
       'cluster_data' not in st.session_state or st.session_state['cluster_data'] is None:
        st.warning("Por favor, execute a Clusterização na página '7. Clusterização' antes de continuar.")
        return
        
    labels = st.session_state['cluster_labels']
    data = st.session_state['cluster_data']
    
    # Filtrar pontos de ruído (comuns no DBSCAN) para as métricas
    if -1 in labels:
        mask = labels != -1
        filtered_data = data[mask]
        filtered_labels = labels[mask]
    else:
        filtered_data = data
        filtered_labels = labels

    if len(set(filtered_labels)) < 2:
        st.error("A avaliação requer pelo menos 2 clusters (excluindo ruído). O algoritmo pode não ter encontrado grupos significativos com os parâmetros atuais.")
        return

    st.markdown("### 📈 Métricas de Avaliação")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Silhouette Score",
            value=f"{silhouette_score(filtered_data, filtered_labels):.3f}",
            help="Mede o quão semelhante um objeto é ao seu próprio cluster em comparação com outros clusters. Varia de -1 a 1. Valores mais altos são melhores."
        )
    with col2:
        st.metric(
            label="Davies-Bouldin Index",
            value=f"{davies_bouldin_score(filtered_data, filtered_labels):.3f}",
            help="Mede a semelhança média entre clusters. Valores mais baixos são melhores, indicando que os clusters estão bem separados."
        )

    st.markdown("---")
    st.markdown("### 🎨 Visualização dos Clusters")
    st.markdown("Aqui podemos ver os clusters plotados nos dois primeiros componentes principais. Cada cor representa um cluster diferente.")
    
    # Adicionar os labels ao dataframe de PCA para visualização
    df_plot = data.copy()
    df_plot['cluster'] = labels

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_plot,
        x='PC_1',
        y='PC_2',
        hue='cluster',
        palette=sns.color_palette("hsv", n_colors=len(set(labels))),
        legend='full',
        alpha=0.7,
        ax=ax
    )
    ax.set_title("Clusters Visualizados em 2D")
    st.pyplot(fig)

    st.markdown("### 🎶 Explorar Clusters")
    st.markdown("Selecione um cluster para ver algumas das músicas que pertencem a ele.")

    cluster_id = st.selectbox("Selecione um Cluster ID:", sorted(set(labels)))
    
    # Adicionar os labels ao dataframe original para análise
    df_original_with_clusters = df.iloc[data.index].copy()
    df_original_with_clusters['cluster'] = labels
    
    musicas_no_cluster = df_original_with_clusters[df_original_with_clusters['cluster'] == cluster_id]
    st.write(f"Mostrando as primeiras 15 músicas do Cluster {cluster_id}:")
    st.dataframe(musicas_no_cluster[['track_name', 'artists', 'track_genre', 'popularity']])


paginas = {
    "1. Visão Geral dos Dados": pagina_1_visao_geral,
    "2. Análise Univariada": pagina_2_analise_univariada,
    "3. Correlação entre Variáveis": pagina_3_correlacao,
    "4. Detecção de Outliers": pagina_4_outliers,
    "5. Pré-processamento": pagina_5_preprocessamento,
    "6. Redução de Dimensionalidade": pagina_6_reducao_dimensionalidade,
    "7. Clusterização": pagina_7_clusterizacao,
    "8. Avaliação dos Clusters": pagina_8_avaliacao_clusters
}

st.sidebar.title("📊 EDA TuneTAP")
escolha = st.sidebar.radio("Escolha uma etapa da análise:", list(paginas.keys()))

# Inicializar st.session_state se não existir
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = None
if 'pca_df' not in st.session_state:
    st.session_state['pca_df'] = None
if 'cluster_labels' not in st.session_state:
    st.session_state['cluster_labels'] = None
if 'cluster_data' not in st.session_state:
    st.session_state['cluster_data'] = None

paginas[escolha](df)
