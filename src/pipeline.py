import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pickle
import base64

def to_df(store_data):
    """Convert dcc.Store data (orient='split') to DataFrame."""
    if not store_data or not isinstance(store_data, dict):
        return None
    try:
        return pd.DataFrame(**store_data)
    except Exception:
        return None

def run_pca(processed_store, features=None, n_components=2, sample_size=10000):
    """
    Executa PCA nos dados processados.
    
    Returns:
        dict com:
        - pca_store: dict orient='split' com dados transformados (PC1, PC2, ...)
        - sampled_store: dict orient='split' com amostra dos dados transformados
        - explained_variance: list com % de variância por componente
        - cumulative_variance: list com % acumulada
        - pca_model: modelo PCA serializado (pickle base64)
        - pc_cols: list com nomes das colunas PC
    """
    df_proc = to_df(processed_store)
    if df_proc is None or df_proc.empty:
        raise ValueError("Dados processados não disponíveis")
    
    # se features não especificadas, usa todas numéricas
    if not features:
        features = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    
    # garante que features existem
    features = [f for f in features if f in df_proc.columns]
    if len(features) < 2:
        raise ValueError("Selecione ao menos 2 features numéricas para PCA")
    
    X = df_proc[features].values
    
    # executa PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # cria DataFrame com PCs
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols, index=df_proc.index)
    
    # amostra para clusterização (se dataset muito grande)
    if len(df_pca) > sample_size:
        df_sampled = df_pca.sample(n=sample_size, random_state=42)
    else:
        df_sampled = df_pca.copy()
    
    # calcula variância explicada
    explained_var = (pca.explained_variance_ratio_ * 100).tolist()
    cumulative_var = np.cumsum(explained_var).tolist()
    
    # serializa modelo PCA
    pca_model_b64 = base64.b64encode(pickle.dumps(pca)).decode('utf-8')
    
    return {
        'pca_store': df_pca.to_dict(orient='split'),
        'sampled_store': df_sampled.to_dict(orient='split'),
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'pca_model': pca_model_b64,
        'pc_cols': pc_cols,
    }

def run_clustering(pca_store, algo='kmeans', k=3, eps=0.5, min_samples=5, n_agg=3):
    """
    Executa clusterização nos dados PCA.
    
    Returns:
        dict com:
        - labels: list com labels de cluster
        - n_clusters: int número de clusters encontrados
        - model: modelo serializado (pickle base64)
    """
    df_pca = to_df(pca_store)
    if df_pca is None or df_pca.empty:
        raise ValueError("Dados PCA não disponíveis")
    
    X = df_pca.values
    
    if algo == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algo == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_agg)
    else:
        raise ValueError(f"Algoritmo {algo} não suportado")
    
    labels = model.fit_predict(X).tolist()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # serializa modelo
    model_b64 = base64.b64encode(pickle.dumps(model)).decode('utf-8')
    
    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'model': model_b64,
    }