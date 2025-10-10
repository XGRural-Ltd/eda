import dash
from dash import html, dcc, callback, Input, Output, no_update
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from sklearn.metrics import silhouette_score, davies_bouldin_score

dash.register_page(__name__, path='/avaliacao', name='7. Avaliação dos Clusters')

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
            html.H4("Visualização dos Clusters (em 2D)", className="text-center"),
            dbc.Spinner(dcc.Graph(id='cluster-scatter-plot', style={'height': '70vh'}))
        ])
    ])
])

# --- Callbacks ---
@callback(
    Output('cluster-metrics-div', 'children'),
    Input('calculate-metrics-checklist', 'value'),
    Input('cluster-labels-store', 'data'),
    Input('pca-df-store', 'data'),
)
def calculate_metrics(checklist_value, labels, pca_data):
    if not checklist_value or not labels or not pca_data:
        return ""

    X_data = pd.DataFrame(**pca_data)
    
    if len(set(labels)) <= 1:
        return dbc.Alert("São necessários pelo menos 2 clusters para calcular as métricas.", color="warning")

    try:
        silhouette = silhouette_score(X_data, labels)
        davies = davies_bouldin_score(X_data, labels)
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
    Input('pca-df-store', 'data')
)
def visualize_clusters(labels, pca_data):
    if labels is None or pca_data is None:
        alert = dbc.Alert("Execute a clusterização primeiro.", color="warning")
        return no_update, alert

    df_pca = pd.DataFrame(**pca_data)
    
    # Ensure labels are a categorical type for better plotting
    df_pca['cluster'] = pd.Series(labels, dtype='category')
    
    fig = px.scatter(
        df_pca, x='PC_1', y='PC_2', color='cluster',
        title="Visualização dos Clusters Encontrados",
        labels={'PC_1': 'Componente Principal 1', 'PC_2': 'Componente Principal 2'}
    )
    return fig, ""