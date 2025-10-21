import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import json

dash.register_page(__name__, path='/clusterizacao', name='6. Clusterização')

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
        return no_update, "", alert

    X_data = pd.DataFrame(**pca_data)
    model = None

    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algo == "Clustering Aglomerativo":
        model = AgglomerativeClustering(n_clusters=n_agg)

    if model:
        labels = model.fit_predict(X_data)
        
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        noise_points = sum(labels == -1) if -1 in labels else 0

        status_msg = f"Clusterização com {algo} concluída! Clusters encontrados: {n_clusters_found}. Pontos de ruído: {noise_points}."
        status_alert = dbc.Alert(status_msg, color="success")
        
        return labels.tolist(), status_alert, ""
    
    return no_update, "Erro: Algoritmo não selecionado.", ""