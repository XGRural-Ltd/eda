import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
import numpy as np
import json

dash.register_page(__name__, path='/reducao', name='5. Redução de Dimensionalidade')

# --- Layout ---
layout = dbc.Container([
    html.H3("📉 Redução de Dimensionalidade (PCA)"),
    dcc.Markdown("""
    A Análise de Componentes Principais (PCA) nos ajuda a 'comprimir' as informações mais importantes de muitas features em um número menor de componentes.
    """),
    html.Div(id='pca-warning-div'),
    
    dbc.Row([
        dbc.Col([
            html.Label("Número de componentes principais para gerar:"),
            dcc.Slider(2, 20, 1, value=10, id='pca-n-components-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], width=12)
    ], className="my-4"),

    dbc.Spinner(dcc.Graph(id='pca-variance-graph')),
    
    html.H5("Amostra dos Dados Transformados pelo PCA", className="mt-4"),
    dbc.Spinner(html.Div(id='pca-result-table')),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            dbc.Button("Salvar Dados do PCA para Próximas Etapas", id='save-pca-df-button', color="primary", n_clicks=0),
        ], width=12, className="mt-4 text-center")
    ], className="mb-4"),
    
    # Toast for save confirmation
    dbc.Toast(
        id="save-pca-toast",
        header="Sucesso!",
        icon="success",
        duration=4000,
        is_open=False,
        style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 9999},
    ),

    # Hidden storage for PCA results
    html.Div(id='pca-df-json-storage', style={'display': 'none'})
])

# --- Callbacks ---
@callback(
    Output('pca-variance-graph', 'figure'),
    Output('pca-result-table', 'children'),
    Output('pca-warning-div', 'children'),
    Output('pca-df-json-storage', 'children'),
    Input('processed-df-store', 'data'),
    Input('pca-n-components-slider', 'value')
)
def perform_pca(processed_data, n_components):
    if processed_data is None:
        alert = dbc.Alert("Execute o pré-processamento primeiro e salve os dados.", color="warning")
        return go.Figure(), "", alert, no_update

    df_preprocessed = pd.DataFrame(**processed_data)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(df_preprocessed)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Create variance plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, n_components + 1)), y=explained_variance, name='Variância Individual'))
    fig.add_trace(go.Scatter(x=list(range(1, n_components + 1)), y=cumulative_variance, name='Variância Cumulativa', mode='lines+markers'))
    fig.update_layout(title_text='Variância Explicada pelos Componentes Principais', xaxis_title='Componentes Principais', yaxis_title='Proporção da Variância')

    # Create result table
    df_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(n_components)])
    table = dbc.Table.from_dataframe(df_pca.head(), striped=True, bordered=True, hover=True, responsive=True)
    
    alert = dbc.Alert(f"Com {n_components} componentes, explicamos {cumulative_variance[-1]:.2%} da variância total.", color="info")
    
    return fig, table, alert, df_pca.to_json(orient='split')

@callback(
    Output('pca-df-store', 'data'),
    Output('save-pca-toast', 'is_open'),
    Output('save-pca-toast', 'children'),
    Input('save-pca-df-button', 'n_clicks'),
    State('pca-df-json-storage', 'children')
)
def save_pca_data(n_clicks, pca_json):
    if n_clicks > 0 and pca_json:
        data_to_store = json.loads(pca_json)
        return data_to_store, True, "Dados do PCA salvos com sucesso! ✅"
    return no_update, False, ""