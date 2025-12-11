import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.constants import STORE_PROCESSED, STORE_PCA, STORE_SAMPLED_PCA, STORE_CLUSTER_LABELS, STORE_CLUSTER_MODEL
from src.utils import to_df
from src.pipeline import run_clustering as pipeline_run_clustering  # renamed to avoid shadowing

dash.register_page(__name__, path='/clusterizacao', name='Clusterização', order=6)

# === ADD STORES AT PAGE TOP (before layout definition) ===
layout = html.Div([
    html.H3("🧩 Clusterização"),
    dcc.Markdown("Aplique algoritmos de clusterização para encontrar grupos (clusters) de músicas com características semelhantes."),
    html.Div(id='cluster-warning-div'),
    
    dbc.Row([ 
        dbc.Col([
            html.Label("Escolha o Algoritmo de Clusterização"),
            dbc.Select(
                id='cluster-algo-dropdown',
                options=[
                    {'label': 'K-Means', 'value': 'kmeans'},
                    {'label': 'DBSCAN', 'value': 'dbscan'},
                    {'label': 'Agglomerative Clustering', 'value': 'agglomerative'},
                ],
                value='kmeans',
                className="mb-3"
            )
        ], width=12)
    ], className="my-4"),

    # algorithm-specific controls
    html.Div([
        html.Div([
            html.Label("Número de clusters (k):"),
            dcc.Slider(2, 20, 1, value=4, id='kmeans-k-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], id='kmeans-controls-div', style={'display': 'block'}),

        html.Div([
            html.Label("Epsilon (eps - raio da vizinhança):"),
            dcc.Slider(0.1, 5.0, 0.1, value=1.5, id='dbscan-eps-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            html.Label("Número Mínimo de Amostras (min_samples):", className="mt-3"),
            dcc.Slider(1, 50, 1, value=10, id='dbscan-minsamples-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], id='dbscan-controls-div', style={'display': 'none'}),

        html.Div([
            html.Label("Número de clusters:"),
            dcc.Slider(2, 20, 1, value=8, id='agg-n-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], id='agg-controls-div', style={'display': 'none'}),
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            dbc.Button("Executar Clusterização", id='run-cluster-button', color="primary", n_clicks=0),  # id matches callback now
            html.Div(id='cluster-status-div', className="mt-3")
        ], width=12, className="mt-4 text-center")
    ],),

    # novo: gráfico de clusters (PC_1 x PC_2)
    dbc.Row([
        dbc.Col(dbc.Spinner(dcc.Graph(id='cluster-plot', figure=px.scatter(title="Clusters (PC_1 x PC_2)"))), width=12)
    ]),

    # Avaliação (mesma página) - botão para disparar avaliação leve (com amostragem)
    dbc.Row([
        dbc.Col(dbc.Button("Avaliar Clusters", id='run-eval-button', color="info", n_clicks=0), width=12, className="mt-3")
    ]),
    dbc.Row([
        dbc.Col(dbc.Spinner(html.Div(id='cluster-eval-div-06')), width=6),  # RENOMEADO
        dbc.Col(dbc.Spinner(dcc.Graph(id='cluster-eval-plot-06')), width=6)
    ]),
])

# --- Callbacks ---

@callback(
    Output('kmeans-controls-div', 'style'),
    Output('dbscan-controls-div', 'style'),
    Output('agg-controls-div', 'style'),
    Input('cluster-algo-dropdown', 'value')
)
def render_cluster_controls(algo_choice):
    kmeans_style = {'display': 'block'} if algo_choice == "kmeans" else {'display': 'none'}
    dbscan_style = {'display': 'block'} if algo_choice == "dbscan" else {'display': 'none'}
    agg_style = {'display': 'block'} if algo_choice == "agglomerative" else {'display': 'none'}
    return kmeans_style, dbscan_style, agg_style

# Callback to run the clustering algorithm (renamed to avoid collision)
@callback(
    Output(STORE_CLUSTER_LABELS, 'data'),
    Output(STORE_CLUSTER_MODEL, 'data'),
    Output('cluster-status-div', 'children'),
    Output('cluster-warning-div', 'children'),
    Output('cluster-plot', 'figure'),
    Input('run-cluster-button', 'n_clicks'),
    State(STORE_PCA, 'data'),
    State(STORE_PROCESSED, 'data'),
    State('cluster-algo-dropdown', 'value'),
    State('kmeans-k-slider', 'value'),
    State('dbscan-eps-slider', 'value'),
    State('dbscan-minsamples-slider', 'value'),
    State('agg-n-slider', 'value'),
    prevent_initial_call=True
)
def run_clustering_cb(n_clicks, pca_store, processed_store, algo, k, eps, min_samples, agg_n):
    df = to_df(pca_store)
    if df is None or df.empty:
        empty_fig = go.Figure(layout={"title": "Nenhum dado PCA disponível", "template": "plotly_dark"})
        return no_update, no_update, dbc.Alert("Nenhum dado PCA disponível", color="warning"), no_update, empty_fig
    
    try:
        # For heavy algorithms, sample the PCA data to a max of 5000 rows to avoid long runtimes / OOM
        sampled = False
        df_for_pipeline = df
        if algo in ("dbscan", "agglomerative") and len(df) > 5000:
            sampled = True
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(len(df), size=5000, replace=False)
            df_for_pipeline = df.iloc[sample_idx].reset_index(drop=True)
            pca_input = df_for_pipeline.to_dict('split')
        else:
            pca_input = pca_store

        # call pipeline function with pca_input (may be sampled)
        out = pipeline_run_clustering(pca_input, algo=algo, k=k, eps=eps, min_samples=min_samples, n_agg=agg_n)
        labels = out.get('labels', [])
        model_obj = out.get('model', None)
        n_clusters = out.get('n_clusters', int(np.max(labels) + 1) if len(labels) else 0)
        print(f"[CLUSTER] Clustering done: {n_clusters} clusters (sampled={sampled})")
        
        # build plot using the same DataFrame that was clustered
        df_plot = (df_for_pipeline.copy() if sampled else df.copy())
        df_plot['cluster'] = labels
        
        fig = px.scatter(df_plot, x=df_plot.columns[0], y=df_plot.columns[1], color='cluster', 
                        title=f'Clusters ({algo.upper()}) - {n_clusters} clusters',
                        template='plotly_dark')
        
        status = dbc.Alert(f"✓ Clusterização concluída: {n_clusters} clusters encontrados (sampled={sampled})", color="success")
        warning = no_update
        
        return labels, model_obj, status, warning, fig
        
    except Exception as e:
        empty_fig = go.Figure(layout={"title": "Erro na clusterização", "template": "plotly_dark"})
        alert = dbc.Alert(f"Erro na clusterização: {type(e).__name__}: {e}", color="danger")
        print(f"[CLUSTER] Error: {e}")
        import traceback
        traceback.print_exc()
        return no_update, no_update, alert, no_update, empty_fig

# Restaurar callback de avaliação (usando dados consistentes)
@callback(
    Output('cluster-eval-div-06', 'children'),
    Output('cluster-eval-plot-06', 'figure'),
    Input('run-eval-button', 'n_clicks'),
    State(STORE_CLUSTER_LABELS, 'data'),
    State('pca-df-store', 'data'),  # Usar os mesmos dados da clusterização
    State('cluster-algo-dropdown', 'value'),
    prevent_initial_call=True
)
def evaluate_clusters(n_clicks, labels, pca_store, algo):
    if not n_clicks or not labels or not pca_store:
        empty_fig = go.Figure(layout={"title": "Avaliação não disponível", "template": "plotly_dark"})
        return no_update, empty_fig
    
    try:
        df = to_df(pca_store)
        if df is None or df.empty:
            return dbc.Alert("Dados PCA não disponíveis para avaliação", color="warning"), go.Figure()
        
        # use up to 5000 samples for evaluation to limit memory/time
        sample_size = min(5000, len(df))
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(df), size=sample_size, replace=False)
        df_sample = df.iloc[sample_indices]
        labels_arr = np.array(labels)[sample_indices]
        
        X = df_sample.values
        
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        sil_avg = silhouette_score(X, labels_arr)
        db_score = davies_bouldin_score(X, labels_arr)
        
        unique, counts = np.unique(labels_arr, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        # build a structured evaluation summary (not a single text block)
        eval_children = [
            html.H5("Métricas de Avaliação"),
            html.Ul([
                html.Li(f"Silhouette médio (global): {sil_avg:.3f}"),
                html.Li(f"Davies-Bouldin: {db_score:.3f}"),
                html.Li(f"Amostra usada: {sample_size} linhas"),
            ]),
            html.H6("Contagem por cluster"),
            html.Ul([html.Li(f"Cluster {int(k)}: {int(v)} pontos") for k, v in cluster_counts.items()])
        ]
        
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, labels_arr)
        
        fig = go.Figure()
        y_lower = 10
        for i in np.unique(labels_arr):
            ith_cluster_silhouette_values = silhouette_vals[labels_arr == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            fig.add_trace(go.Bar(
                x=ith_cluster_silhouette_values,
                y=np.arange(y_lower, y_upper),
                orientation='h',
                name=f'Cluster {int(i)}',
                showlegend=False
            ))
            fig.add_vline(x=sil_avg, line_dash="dash", line_color="red")
            y_lower = y_upper
        
        fig.update_layout(
            title="Gráfico de Silhueta",
            xaxis_title="Valor da Silhueta",
            yaxis_title="Cluster",
            template='plotly_dark'
        )
        
        return html.Div(eval_children), fig
        
    except Exception as e:
        empty_fig = go.Figure(layout={"title": "Erro na avaliação", "template": "plotly_dark"})
        return dbc.Alert(f"Erro na avaliação: {type(e).__name__}: {e}", color="danger"), empty_fig