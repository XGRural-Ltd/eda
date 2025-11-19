import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
import joblib
import io
import base64

dash.register_page(__name__, path='/clusterizacao', name='Clusterização', order=6)

# --- Layout ---
layout = dbc.Container([
    html.H3("🧩 Clusterização"),
    dcc.Markdown("Aplique algoritmos de clusterização para encontrar grupos (clusters) de músicas com características semelhantes."),
    html.Div(id='cluster-warning-div'),
    
    dbc.Row([
        dbc.Col([
            html.Label("Escolha o Algoritmo de Clusterização"),
            dcc.Dropdown(
                id='cluster-algo-dropdown',
                options=["K-Means", "DBSCAN", "Clustering Aglomerativo"],
                value="K-Means"
            )
        ], width=12)
    ], className="my-4"),

    # --- Container for all possible algorithm controls ---
    html.Div([
        # K-Means Controls (initially visible)
        html.Div([
            html.Label("Número de clusters (k):"),
            dcc.Slider(2, 20, 1, value=8, id='kmeans-k-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], id='kmeans-controls-div', style={'display': 'block'}),

        # DBSCAN Controls (initially hidden)
        html.Div([
            html.Label("Epsilon (eps - raio da vizinhança):"),
            dcc.Slider(0.1, 5.0, 0.1, value=1.5, id='dbscan-eps-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Número Mínimo de Amostras (min_samples):", className="mt-3"),
            dcc.Slider(1, 50, 1, value=10, id='dbscan-minsamples-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], id='dbscan-controls-div', style={'display': 'none'}),

        # Agglomerative Clustering Controls (initially hidden)
        html.Div([
            html.Label("Número de clusters:"),
            dcc.Slider(2, 20, 1, value=8, id='agg-n-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], id='agg-controls-div', style={'display': 'none'}),
    ]),


    dbc.Row([
        dbc.Col([
            html.Hr(),
            dbc.Button("Executar Clusterização", id='run-cluster-button', color="primary", n_clicks=0),
            html.Div(id='cluster-status-div', className="mt-3")
        ], width=12, className="mt-4 text-center")
    ]),
])

# --- Callbacks ---

# Callback to dynamically generate controls based on algorithm choice
@callback(
    Output('kmeans-controls-div', 'style'),
    Output('dbscan-controls-div', 'style'),
    Output('agg-controls-div', 'style'),
    Input('cluster-algo-dropdown', 'value')
)
def render_cluster_controls(algo_choice):
    kmeans_style = {'display': 'block'} if algo_choice == "K-Means" else {'display': 'none'}
    dbscan_style = {'display': 'block'} if algo_choice == "DBSCAN" else {'display': 'none'}
    agg_style = {'display': 'block'} if algo_choice == "Clustering Aglomerativo" else {'display': 'none'}
    return kmeans_style, dbscan_style, agg_style

# Callback to run the clustering algorithm
@callback(
    Output('cluster-labels-store', 'data'),
    Output('prediction-model-store', 'data'),
    Output('sampled-pca-df-store', 'data'),
    Output('cluster-status-div', 'children'),
    Output('cluster-warning-div', 'children'),
    Input('run-cluster-button', 'n_clicks'),
    State('pca-df-store', 'data'),
    State('cluster-algo-dropdown', 'value'),
    # K-Means states
    State('kmeans-k-slider', 'value'),
    # DBSCAN states
    State('dbscan-eps-slider', 'value'),
    State('dbscan-minsamples-slider', 'value'),
    # Agglomerative states
    State('agg-n-slider', 'value'),
    prevent_initial_call=True
)
def run_clustering(n_clicks, pca_data, algo, k, eps, min_samples, n_agg):
    if pca_data is None:
        alert = dbc.Alert("Execute a Redução de Dimensionalidade primeiro e salve os dados.", color="warning")
        return no_update, no_update, no_update, "", alert

    X_data = pd.DataFrame(**pca_data)
    model = None
    predictor_model = None
    
    # --- Sampling for expensive algorithms ---
    SAMPLE_SIZE = 10000
    data_to_cluster = X_data
    sampling_info = ""
    
    if algo in ["DBSCAN", "Clustering Aglomerativo"] and len(X_data) > SAMPLE_SIZE:
        data_to_cluster = X_data.sample(n=SAMPLE_SIZE, random_state=42)
        sampling_info = f" (usando uma amostra de {SAMPLE_SIZE} pontos)"

    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algo == "Clustering Aglomerativo":
        model = AgglomerativeClustering(n_clusters=n_agg)

    if model:
        labels = model.fit_predict(data_to_cluster)
        
        # --- Create a suitable prediction model ---
        if algo == "K-Means":
            predictor_model = model # K-Means can predict directly
        else:
            # For other types, train a KNN to learn the cluster assignments
            predictor_model = KNeighborsClassifier(n_neighbors=5)
            predictor_model.fit(data_to_cluster, labels)

        # Serialize the predictor model to a base64 string to store it
        mem_buffer = io.BytesIO()
        joblib.dump(predictor_model, mem_buffer)
        mem_buffer.seek(0)
        base64_model = base64.b64encode(mem_buffer.read()).decode('utf-8')
        
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = sum(labels == -1) if -1 in labels else 0

        status_msg = f"Clusterização com {algo} concluída{sampling_info}! Clusters: {n_clusters_found}. Ruído: {noise_points}."
        # Add duration to make the alert disappear after 7 seconds
        status_alert = dbc.Alert(status_msg, color="success", duration=7000)
        
        # Store the data that was actually clustered (sampled or full)
        clustered_data_dict = data_to_cluster.to_dict(orient='split')

        return labels.tolist(), base64_model, clustered_data_dict, status_alert, ""
    
    return no_update, no_update, no_update, no_update, "Erro: Algoritmo não selecionado."