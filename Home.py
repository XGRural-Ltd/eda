import streamlit as st
import joblib
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

st.set_page_config(layout = "wide")

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    path_dataset = path + '/dataset.csv'
    df = pd.read_csv(path_dataset, index_col=0)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'track_genre' not in df.columns:
        df['track_genre'] = 'unknown'
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def pagina_visao_geral(df):
    st.subheader("📊 Informações Gerais do Dataset")
    st.write("Nesta etapa vamos ficar mais familiarizados com os dados. Vamos explorar as colunas, tipos de dados, valores ausentes, estatísticas descritivas e visualizar alguns plots.")

    st.markdown("**Visualize o DataFrame com as colunas selecionadas:**")
    cols = st.multiselect("Colunas para exibir:", df.columns.tolist(), default=df.columns[:6].tolist())
    st.dataframe(df[cols].head(15), hide_index=True)

    st.markdown("**Estatísticas descritivas:**")
    desc = df.describe().rename(columns=cols_dict).T
    desc = desc.drop(columns=['count'])
    desc = desc.map(lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else x)
    st.table(desc)

    st.write(f"Existem {df[df.duplicated()].shape[0]} faixas duplicadas nessa base de dados.")
    st.write(f"Existem {df[df.isnull().any(axis=1)].shape[0]} faixas com dados faltantes.")
    df = df.dropna(axis=0)

    st.markdown("---")
    st.subheader("🧾 Dicionário de Dados: Descrição das Colunas")
    st.table(pd.DataFrame.from_dict(col_descriptions, orient='index', columns=['Descrição']))
    
    st.markdown("---")
    st.subheader("📈 Visualizações Gerais de Distribuição")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_col = st.selectbox("Selecione uma coluna numérica:", num_cols)
    selected_col_name = cols_dict.get(selected_col)
    plot_type = st.radio("Tipo de gráfico:", ["Histograma", "Boxplot", "Ambos"])
    if plot_type == "Ambos":
        # Show histogram
        fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
        sns.histplot(df[selected_col], kde=True, ax=ax_hist, color='skyblue')
        ax_hist.set_xlabel(selected_col_name)
        ax_hist.set_ylabel("Frequência")
        ax_hist.set_title(f"Histograma de {selected_col_name}")
        st.pyplot(fig_hist)
        # Show boxplot
        fig_box, ax_box = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=df[selected_col], ax=ax_box, color='salmon')
        ax_box.set_xlabel(selected_col_name)
        ax_box.set_ylabel("Valor")
        ax_box.set_title(f"Boxplot de {selected_col_name}")
        st.pyplot(fig_box)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        if plot_type == "Histograma":
            sns.histplot(df[selected_col], kde=True, ax=ax, color='skyblue')
            ax.set_xlabel(selected_col_name)
            ax.set_ylabel("Frequência")
            ax.set_title(f"Histograma de {selected_col_name}")
        else:
            sns.boxplot(x=df[selected_col], ax=ax, color='salmon')
            ax.set_xlabel(selected_col_name)
            ax.set_ylabel("Valor")
            ax.set_title(f"Boxplot de {selected_col_name}")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📈 Gráficos de Dispersão entre Variáveis Numéricas")
    x_axis = st.selectbox("Variável no eixo X:", num_features, index=0)
    y_axis = st.selectbox("Variável no eixo Y:", num_features, index=1)
    x_axis_name = cols_dict.get(x_axis)
    y_axis_name = cols_dict.get(y_axis)
    show_trend = st.checkbox("Mostrar linha de tendência (regressão linear)")
    fig, ax = plt.subplots()
    if x_axis == y_axis:
        sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.5, color='red', ax=ax)
        ax.plot(df[x_axis], df[y_axis], color='blue', linewidth=1, alpha=0.5)
    else:
        if show_trend:
            sns.regplot(data=df, x=x_axis, y=y_axis, ax=ax,
                        scatter_kws={'alpha': 0.5, 'color': 'red'},
                        line_kws={"color": "blue"})
        else:
            sns.scatterplot(data=df, x=x_axis, y=y_axis, alpha=0.5, color='red', ax=ax)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)
    ax.set_title(f"Dispersão entre {x_axis_name} e {y_axis_name}")
    st.pyplot(fig)

    # st.markdown("📌 Danceability vs. Energy \n"
    #             "- Já esperamnos uma correlação positiva entre 'danceability' e 'energy', pois músicas mais dançantes tendem a ter mais energia.  \n"
    #             "- A linha de tendência (regressão linear) ajuda a visualizar uma correlação moderadamente positiva. \n"
    #             "\n" 
    #             "📌 Acousticness vs. Energy \n"
    #             "- Correlação negativa forte esperada, pois músicas acústicas são menos energéticas \n"
    #             "- A linha de tendência decrescemte indica uma relação inversamente proporcional. \n"
    #             "\n"
    #             "📌 Loudness vs. Energy \n"
    #             "- Baixa dispersão e ascendência dos pontos mostram uma correlação fortemente positiva (músicas energéticas costumam ser mais altas) \n")

def pagina_analise_univariada(df):
    st.subheader("🔬 Análise Univariada Detalhada")
    st.markdown("Explore a distribuição de cada variável. Use os filtros para comparar diferentes gêneros e ajuste os gráficos para uma análise mais profunda.")

    st.markdown("### 🎭 Comparar Distribuições por Gênero")
    genres = sorted(df['track_genre'].unique().tolist())
    genres_cap = [g.capitalize() for g in genres]
    genre_map = dict(zip(genres_cap, genres))

    genres_to_compare_cap = st.multiselect(
        "Selecione um ou mais gêneros para comparar (opcional):",
        genres_cap
    )
    genres_to_compare = [genre_map[g] for g in genres_to_compare_cap]

    if genres_to_compare:
        df_filtered = df[df['track_genre'].isin(genres_to_compare)]
        hue_on = 'track_genre'
    else:
        df_filtered = df.copy()
        hue_on = None

    st.markdown("### ⚙️ Controles da Análise")
    selected_var = st.selectbox("Selecione uma variável numérica para análise:", num_features)
    if selected_var in col_descriptions:
        st.info(f"**Descrição:** {col_descriptions[selected_var]}")

    num_bins = st.slider("Número de Bins para o Histograma:", min_value=10, max_value=100, value=30)
    
    st.markdown(f"---\n### 🎯 Análise da variável: `{selected_var}`")
    st.markdown("**📊 Visualização da Distribuição:**")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        if hue_on:
            sns.histplot(data=df_filtered, x=selected_var, hue=hue_on, kde=True, ax=ax1, bins=num_bins, palette='viridis')
        else:
            sns.histplot(data=df_filtered, x=selected_var, kde=True, ax=ax1, bins=num_bins, color='skyblue')
        ax1.set_title(f"Distribuição de {selected_var}")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if hue_on:
            sns.boxplot(data=df_filtered, x=selected_var, y=hue_on, ax=ax2, palette='viridis', orient='h')
        else:
            sns.boxplot(data=df_filtered, x=selected_var, ax=ax2, color='skyblue', orient='h')
        ax2.set_title(f"Boxplot de {selected_var}")
        st.pyplot(fig2)

    st.markdown(f"**🔎 Estatísticas e Outliers para `{selected_var}`**")
    if genres_to_compare:
        st.write("Estatísticas descritivas por gênero selecionado:")
        st.write(df_filtered.groupby('track_genre')[selected_var].describe().T)
    else:
        st.write("Estatísticas descritivas gerais:")
        st.write(df_filtered[selected_var].describe())
    
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
        cols_to_show = ['track_name', 'artists', 'track_genre', selected_var]
        st.dataframe(outliers[cols_to_show].sort_values(by=selected_var, ascending=False))

def pagina_preprocessamento(df):
    st.subheader("⚙️ 5. Pré-processamento dos Dados")
    st.markdown("""
    O pré-processamento é uma etapa fundamental na preparação de dados para modelos de Machine Learning. Aqui, transformaremos nossas features para que os algoritmos possam interpretá-las da melhor forma possível.
        1.  **Tratamento de Valores Ausentes**: Lidaremos com quaisquer valores faltantes em nossas features numéricas.
        2.  **Feature Scaling**: Padronizar as escalas das nossas variáveis numéricas.
    """)

    df_preprocessed = df.copy()

    st.markdown("---")
    st.markdown("### 1 **Tratamento de Valores Ausentes**")
    st.markdown("""
    Antes de escalar os dados, precisamos lidar com valores ausentes (NaN). Uma estratégia comum e robusta é a **imputação pela mediana**, onde substituímos os valores ausentes pelo valor central da coluna. Isso é menos sensível a outliers do que usar a média.
    """)
    
    # Preencher NaNs com a mediana das colunas numéricas
    numeric_cols_with_na = df_preprocessed[num_features].columns[df_preprocessed[num_features].isnull().any()].tolist()
    if numeric_cols_with_na:
        st.write("Valores ausentes encontrados nas seguintes colunas e preenchidos com a mediana:")
        st.write(df_preprocessed[numeric_cols_with_na].isnull().sum().to_frame(name='NAs Preenchidos'))
        df_preprocessed.fillna(df_preprocessed.median(numeric_only=True), inplace=True)
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
        df_preprocessed[features_to_scale] = scaler.fit_transform(df_preprocessed[features_to_scale])

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
            sns.histplot(df_preprocessed['danceability'], kde=True, ax=ax, color='green')
            ax.set_title("Padronizado")
            st.pyplot(fig)
        st.success("Padronização concluída! ✅")
    else:
        st.warning("Por favor, selecione pelo menos uma feature para padronizar.")
        return
    
    st.markdown("---")
    st.markdown("### Resultado do pré-processamento")
    st.markdown("Abaixo está uma amostra do dataset após as transformações. Este é o conjunto de dados que serão usados para a clusterização.")
    
    final_features = df_preprocessed.select_dtypes(include=np.number).columns.tolist()
    df_preprocessed = df_preprocessed[final_features]

    st.dataframe(df_preprocessed.head())
    st.write(f"O dataset final possui **{df_preprocessed.shape[0]}** linhas e **{df_preprocessed.shape[1]}** features.")

    st.markdown("### 💾 Baixar Dados Processados")
    st.markdown("Clique no botão para baixar o DataFrame processado em um arquivo CSV para uso posterior.")

    csv = convert_df_to_csv(df_preprocessed)
    st.download_button(
       label="Baixar dados como CSV",
       data=csv,
       file_name='processed_spotify_data.csv',
       mime='text/csv',
    )
    if st.button("Salvar DataFrame em cache para próximas etapas"):
        st.session_state['df_preprocessed'] = df_preprocessed
        st.success("DataFrame processado salvo na sessão! ✅")

def pagina_correlacao(df):
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

    st.markdown("### ⚙️ Controles da matriz de correlação")
    corr_method = st.selectbox("Método de correlação:", ["pearson", "spearman", "kendall"])
    
    corr_threshold = st.slider("Ocultar no gráfico e na tabela correlações com valor absoluto abaixo de:", 0.0, 1.0, 0.0, 0.05)
    
    if not df_corr.empty and len(df_corr) > 1:
        corr_matrix = df_corr.corr(method=corr_method)
        
        mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))
        mask_threshold = np.abs(corr_matrix) < corr_threshold
        mask_heatmap = mask_upper | mask_threshold
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, mask=mask_heatmap, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 10}, vmin=-1, vmax=1, linewidths=0.5, linecolor='gray')
        ax.set_title(f"Matriz de correlação (Método: {corr_method.capitalize()}, Gênero: {corr_genre})", fontsize=16)
        st.pyplot(fig)

        st.markdown("### ✨ Pares com Maior Correlação")
        st.write(f"Abaixo estão os pares de variáveis com correlação absoluta acima de **{corr_threshold}** (excluindo duplicatas e auto-correlações).")
        
        mask_table = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_unstacked = corr_matrix.where(mask_table).stack()
        
        strong_pairs = corr_unstacked.sort_values(key=abs, ascending=False);
        strong_pairs = strong_pairs[abs(strong_pairs) > corr_threshold]
        if strong_pairs.empty:
            st.warning("Nenhum par encontrado acima do threshold. Tente um valor menor.")
        else:
            df_corr_pairs = strong_pairs.reset_index()
            df_corr_pairs.columns = ['Variável 1', 'Variável 2', 'Valor da Correlação']
            st.dataframe(df_corr_pairs.head(20))
    else:
        st.warning(f"Não há dados suficientes para o gênero '{corr_genre}' para calcular a correlação.")

def pagina_reducao_dimensionalidade(df):
    st.subheader("📉 6. Redução de Dimensionalidade")
    st.markdown("""
    Com um grande número de features, pode ser difícil visualizar e modelar os dados. A **Redução de Dimensionalidade** nos ajuda a "comprimir" as informações mais importantes em um número menor de componentes. Usaremos a **Análise de Componentes Principais (PCA)**, uma técnica popular que encontra novas eixos (componentes) que maximizam a variância nos dados.
    """)
    
    if 'df_preprocessed' not in st.session_state or st.session_state['df_preprocessed'] is None:
        st.warning("Por favor, execute o pré-processamento na página 'Pré-processamento' e clique em 'Salvar DataFrame' antes de continuar.")
        return
    df_preprocessed = st.session_state['df_preprocessed']
    st.markdown("O gráfico abaixo mostra quanta da variância original dos dados é 'capturada' por cada componente principal. O ideal é que os primeiros componentes capturem a maior parte da informação.")
    n_components = st.slider(
        "Número de componentes principais para gerar:",
        min_value=2, max_value=len(df_preprocessed.columns), value=int(np.mean([2, len(df_preprocessed.columns)])),
        help="Escolha quantos componentes (novas features) você deseja criar. Começar com 10 é, em geral, um bom ponto de partida."
    )
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df_preprocessed)
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
    
    st.markdown("### Dados Transformados pelo PCA")
    st.markdown("Abaixo está o nosso dataset transformado, agora com um número reduzido de dimensões. Estes serão os dados que usaremos para a clusterização.")
    
    df_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components)])
    st.dataframe(df_pca.head())

    if st.button("Salvar dados do PCA para próximas etapas"):
        st.session_state['df_pca'] = df_pca
        st.success("Dados transformados pelo PCA salvos na sessão! ✅")

def pagina_clusterizacao(df):
    st.subheader("🧩 7. Clusterização")
    st.markdown("""
    Agora que nossos dados estão preparados, vamos aplicar algoritmos de **clusterização** para encontrar grupos (clusters) de músicas com características semelhantes. O objetivo é descobrir "playlists" naturais escondidas nos dados.
    """)

    if 'df_pca' not in st.session_state or st.session_state['df_pca'] is None:
        st.warning("Por favor, execute a Redução de Dimensionalidade na página 'Redução de Dimensionalidade' e salve os dados antes de continuar.")
        return
    X_data = st.session_state['df_pca']

    st.markdown("### 🤖 Escolha do Algoritmo de Clusterização")
    algo_choice = st.selectbox(
        "Selecione o algoritmo:",
        ["K-Means", "DBSCAN", "Clustering Aglomerativo"]
    )
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

def pagina_avaliacao_clusters(df):
    st.subheader("🏆 8. Avaliação dos Clusters")
    st.markdown("""
    Como saber se os clusters que encontramos são bons? Nesta etapa, vamos usar métricas quantitativas e visualizações para avaliar a qualidade dos nossos agrupamentos.
    """)

    if 'cluster_labels' not in st.session_state or st.session_state['cluster_labels'] is None or len(st.session_state['cluster_labels']) == 0:
        st.warning("Por favor, execute a clusterização na página '7. Clusterização' antes de continuar.")
        return

    labels = st.session_state['cluster_labels']
    X_data = st.session_state['cluster_data']

    st.markdown("### 📊 Avaliação Quantitativa")
    st.markdown("""
    Vamos usar duas métricas populares para avaliar a qualidade dos clusters:
    - **Silhouette Score**: Mede quão semelhantes são os objetos dentro de um mesmo cluster em comparação com objetos de outros clusters. Varia de -1 a 1.
    - **Davies-Bouldin Score**: Mede a compactação e separação dos clusters. Valores mais baixos indicam melhores agrupamentos.
    """)

    if st.checkbox("Calcular métricas de avaliação"):
        if len(set(labels)) > 1:
            silhouette_avg = silhouette_score(X_data, labels)
            davies_bouldin = davies_bouldin_score(X_data, labels)
            st.success(f"Silhouette Score: {silhouette_avg:.3f}")
            st.success(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
        else:
            st.warning("Não é possível calcular as métricas. Tente com mais clusters.")

    st.markdown("### 📉 Visualização dos Clusters")
    st.markdown("""
    Uma imagem vale mais que mil palavras. Vamos visualizar os clusters em um gráfico 2D. Para isso, usaremos as duas primeiras componentes principais obtidas na redução de dimensionalidade.
    """)
    
    if 'df_pca' in st.session_state:
        df_pca = st.session_state['df_pca']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=df_pca, x='PC1', y='PC2', hue=labels, 
            palette='viridis', style=labels,
            markers={0: 'o', 1: 'X', 2: 's', 3: 'D', 4: '^', 5: 'v', 6: '<', 7: '>', -1: 'P'}, 
            s=100, alpha=0.7, ax=ax
        )
        ax.set_title("Visualização dos Clusters Encontrados")
        ax.legend(title="Clusters")
        st.pyplot(fig)
    else:
        st.warning("Dados do PCA não encontrados. Verifique a etapa de redução de dimensionalidade.")

def pagina_predicao(df):
    # Carregue modelo, encoder e dados
    rf = joblib.load('random_forest_model.pkl')
    le = joblib.load('label_encoder.pkl')
    selected_features = joblib.load('selected_features.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('seu_dataset.csv')

    # Interface
    generos = sorted(df['track_genre'].unique())
    generos_escolhidos = st.multiselect("Escolha os gêneros:", generos)
    tam_playlist = st.slider("Tamanho da playlist:", 5, 50, 10)

    if generos_escolhidos:
        # Filtrar músicas dos gêneros escolhidos
        df_filtrado = df[df['track_genre'].isin(generos_escolhidos)].copy()
        # (Opcional) Prever o gênero das músicas filtradas e pegar as mais "típicas"
        X_filtrado = scaler.transform(df_filtrado[selected_features])
        pred = rf.predict(X_filtrado)
        probas = rf.predict_proba(X_filtrado).max(axis=1)
        df_filtrado['probabilidade'] = probas
        # Ordena pelas mais "típicas" (maior probabilidade do modelo)
        df_final = df_filtrado.sort_values('probabilidade', ascending=False).head(tam_playlist)
        # Exibe a playlist
        st.write("Sua playlist personalizada:")
        st.dataframe(df_final[['track_name', 'artists', 'track_genre']])
    else:
        st.info("Selecione pelo menos um gênero.")

df = load_data()
num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cols_dict = {'track_id' : 'Track ID',
             'artists' : 'Artists',
             'album_name' : 'Album Name',
             'track_name' : 'Track Name',
             'popularity' : 'Popularity',
             'duration_ms' : 'Duration (ms)',
             'explicit' : 'Explicit',
             'danceability' : 'Danceability',
             'energy' : 'Energy',
             'key' : 'Key',
             'loudness' : 'Loudness',
             'mode' : 'Mode',
             'speechiness' : 'Speechiness',
             'acousticness' : 'Acousticness',
             'instrumentalness' : 'Instrumentalness',
             'liveness' : 'Liveness',
             'valence' : 'Valence',
             'tempo' : 'Tempo',
             'time_signature' : 'Time Signature',
             'track_genre' : 'Track Genre'}
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

PAGES = {
    "1. Visão Geral": pagina_visao_geral,
    "2. Análise Univariada": pagina_analise_univariada,
    "3. Pré-processamento": pagina_preprocessamento,
    "4. Correlação": pagina_correlacao,
    "5. Redução de Dimensionalidade": pagina_reducao_dimensionalidade,
    "6. Clusterização": pagina_clusterizacao,
    "7. Avaliação dos Clusters": pagina_avaliacao_clusters,
    "8. Predição": pagina_predicao,
}
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha a página:", list(PAGES.keys()))
PAGES[page](df)