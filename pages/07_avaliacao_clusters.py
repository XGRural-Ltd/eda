import dash
from dash import html, dcc, callback, Input, Output, no_update
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.metrics import silhouette_score, davies_bouldin_score

dash.register_page(__name__, path='/avaliacao', name='Avaliação dos Clusters', order=7)

# --- Layout ---
layout = dbc.Container([
    html.H3("🏆 Avaliação dos Clusters"),
    dcc.Markdown("Use métricas quantitativas e visualizações para avaliar a qualidade dos agrupamentos."),
    html.Div(id='eval-warning-div'),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Avaliação Quantitativa"),
                dbc.CardBody([
                    dbc.Checklist(
                        options=[{"label": "Calcular métricas de avaliação", "value": 1}],
                        value=[],
                        id="calculate-metrics-checklist",
                        switch=True,
                    ),
                    html.Div(id='cluster-metrics-div', className="mt-3")
                ])
            ])
        ], width=12)
    ], className="my-4"),

    dbc.Row([
        dbc.Col([
            html.H4("Visualização dos Clusters", className="text-center"),
            dbc.Spinner(dcc.Graph(id='cluster-scatter-plot', style={'height': '70vh'}))
        ])
    ])
])

# --- Callbacks ---
@callback(
    Output('cluster-metrics-div', 'children'),
    Input('calculate-metrics-checklist', 'value'),
    Input('cluster-labels-store', 'data'),
    Input('sampled-pca-df-store', 'data'),
)
def calculate_metrics(checklist_value, labels, pca_data):
    if not checklist_value or not labels or not pca_data:
        return ""

    X_data = pd.DataFrame(**pca_data)
    labels = pd.Series(labels) # Convert to series for easier filtering
    
    # Filter out noise points (label -1) from DBSCAN before calculating metrics
    mask = labels != -1
    if mask.sum() == 0: # All points are noise
        return dbc.Alert("Não foram encontrados clusters (apenas ruído). As métricas não podem ser calculadas.", color="info")

    X_filtered = X_data[mask]
    labels_filtered = labels[mask]
    
    if len(set(labels_filtered)) <= 1:
        return dbc.Alert("São necessários pelo menos 2 clusters (sem contar ruído) para calcular as métricas.", color="warning")

    try:
        silhouette = silhouette_score(X_filtered, labels_filtered)
        davies = davies_bouldin_score(X_filtered, labels_filtered)
        return [
            html.P(f"**Silhouette Score:** {silhouette:.3f} (Quanto mais perto de 1, melhor)"),
            html.P(f"**Davies-Bouldin Score:** {davies:.3f} (Quanto mais perto de 0, melhor)")
        ]
    except Exception as e:
        return dbc.Alert(f"Erro ao calcular métricas: {e}", color="danger")

@callback(
    Output('cluster-scatter-plot', 'figure'),
    Output('eval-warning-div', 'children'),
    Input('cluster-labels-store', 'data'),
    Input('sampled-pca-df-store', 'data')
)
def visualize_clusters(labels, pca_data):
    if labels is None or pca_data is None:
        alert = dbc.Alert("Execute a clusterização primeiro.", color="warning")
        return no_update, alert

    df_pca = pd.DataFrame(**pca_data)
    
    # Ensure labels are a string type for discrete coloring in plots
    df_pca['cluster'] = pd.Series(labels).astype(str)

    fig = px.scatter_3d(
        df_pca, x='PC1', y='PC2', z='PC3',
        color='cluster',
        title="Visualização 3D dos Clusters",
        labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'PC3': 'Componente Principal 3'}
    )
    return fig, ""