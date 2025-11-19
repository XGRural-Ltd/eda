import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np
from io import StringIO

dash.register_page(__name__, path='/correlacao', name='Correlação', order=3)

# --- Layout ---
layout = dbc.Container([
    html.H3("↔️ Análise de Correlação"),
    dcc.Markdown("Investigue a relação entre as variáveis. Use o filtro de gênero e o slider de intensidade para focar nas correlações mais importantes."),

    dbc.Row([
        dbc.Col([
            html.Label("Filtrar por Gênero"),
            dcc.Dropdown(id='corr-genre-select', placeholder="Selecione um gênero...")
        ], width=4),
        dbc.Col([
            html.Label("Método de Correlação"),
            dcc.Dropdown(id='corr-method-select', options=['pearson', 'spearman', 'kendall'], value='pearson')
        ], width=4),
        dbc.Col([
            html.Label("Ocultar correlações com valor absoluto abaixo de:"),
            dcc.Slider(0.0, 1.0, 0.05, value=0.0, id='corr-threshold-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], width=4),
    ], className="my-3"),

    dbc.Row([
        dbc.Col(dbc.Spinner(dcc.Graph(id='corr-heatmap', style={'height': '80vh'})), width=8),
        dbc.Col(dbc.Spinner(html.Div(id='corr-strong-pairs-table')), width=4),
    ]),
], fluid=True)

# --- Callbacks ---
@callback(
    Output('corr-genre-select', 'options'),
    Input('main-df-store', 'data')
)
def populate_corr_dropdown(json_data):
    if json_data is None: return []
    df = pd.read_json(StringIO(json_data), orient='split')
    return ['Todos'] + sorted(df['track_genre'].unique().tolist())

@callback(
    Output('corr-heatmap', 'figure'),
    Output('corr-strong-pairs-table', 'children'),
    Input('main-df-store', 'data'),
    Input('corr-genre-select', 'value'),
    Input('corr-method-select', 'value'),
    Input('corr-threshold-slider', 'value')
)
def update_correlation_analysis(json_data, selected_genre, corr_method, corr_threshold):
    if not all([json_data, corr_method]) or selected_genre is None:
        return go.Figure(layout={'title': 'Selecione as opções para gerar o gráfico'}), "Selecione as opções"

    df = pd.read_json(StringIO(json_data), orient='split')
    num_features = df.select_dtypes(include=np.number).columns.tolist()

    if selected_genre == 'Todos':
        df_corr = df[num_features]
    else:
        df_corr = df[df['track_genre'] == selected_genre][num_features]

    if df_corr.empty or len(df_corr) < 2:
        return go.Figure(layout={'title': f'Não há dados suficientes para {selected_genre}'}), "Dados insuficientes"

    # Heatmap
    corr_matrix = df_corr.corr(method=corr_method)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_masked = corr_matrix.mask(mask) # Hide upper triangle
    
    fig = go.Figure(go.Heatmap(
        z=corr_matrix_masked,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix_masked.round(2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig.update_layout(title=f"Matriz de Correlação ({corr_method.capitalize()}, Gênero: {selected_genre})")

    # Strong pairs table
    corr_unstacked = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)).stack()
    strong_pairs = corr_unstacked[abs(corr_unstacked) >= corr_threshold].sort_values(key=abs, ascending=False)

    if strong_pairs.empty:
        table_content = dbc.Alert(f"Nenhum par encontrado com correlação absoluta > {corr_threshold}", color="warning")
    else:
        df_pairs = strong_pairs.reset_index()
        df_pairs.columns = ['Variável 1', 'Variável 2', 'Correlação']
        table_content = [
            html.H5(f"Pares com Maior Correlação (> {corr_threshold})"),
            dbc.Table.from_dataframe(df_pairs.head(20), striped=True, bordered=True, hover=True, size='sm')
        ]
        
    return fig, table_content