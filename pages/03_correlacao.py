import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.constants import STORE_MAIN
from src.utils import to_df

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
            # --- CHANGED: default alinhado ao notebook (0.4) ---
            dcc.Slider(0.0, 1.0, 0.05, value=0.4, id='corr-threshold-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
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
    Input(STORE_MAIN, 'data')
)
def populate_corr_dropdown(json_data):
    if json_data is None:
        return []
    df = to_df(json_data)
    # garante coluna de gênero como no pipeline2.0
    if 'track_genre' not in df.columns:
        df['track_genre'] = 'unknown'
    return ['Todos'] + sorted(df['track_genre'].unique().tolist())

@callback(
    Output('corr-heatmap', 'figure'),
    Output('corr-strong-pairs-table', 'children'),
    Input(STORE_MAIN, 'data'),
    Input('corr-genre-select', 'value'),
    Input('corr-method-select', 'value'),
    Input('corr-threshold-slider', 'value')
)
def update_correlation_analysis(main_store, selected_genre, corr_method, corr_threshold):
    df = to_df(main_store)

    if df is None or df.empty or corr_method is None or selected_genre is None:
        empty_fig = go.Figure(layout={"title": "Selecione método e gênero", "template": "plotly_dark"})
        return empty_fig, html.Div("Nenhum dado disponível.")

    # --- CHANGED: usa a lista explícita de num_features do pipeline2.0 ---
    num_features = [
        'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    # filtra apenas colunas que realmente existem no df
    num_features = [f for f in num_features if f in df.columns]

    # garante coluna de gênero
    if 'track_genre' not in df.columns:
        df['track_genre'] = 'unknown'

    if selected_genre == 'Todos':
        df_corr = df[num_features].copy()
    else:
        df_corr = df[df['track_genre'] == selected_genre][num_features].copy()

    if df_corr.empty or len(df_corr) < 2:
        return go.Figure(layout={'title': f'Não há dados suficientes para {selected_genre}'}), "Dados insuficientes"

    # --- CHANGED: corr_matrix conforme pipeline2.0 (método escolhível) ---
    corr_matrix = df_corr.corr(method=corr_method)

    # Heatmap: esconder triângulo superior (k=1) como no notebook
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_matrix_masked = corr_matrix.mask(mask)  # oculta triângulo superior e diagonal
    fig = go.Figure(go.Heatmap(
        z=corr_matrix_masked.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr_matrix_masked.round(2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig.update_layout(title=f"Matriz de Correlação ({corr_method.capitalize()}, Gênero: {selected_genre})")

    # --- CHANGED: unstack usando mask com k=1 e filtragem '>' como no notebook ---
    corr_unstacked = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)).stack()
    strong_pairs = corr_unstacked[abs(corr_unstacked) > corr_threshold].sort_values(key=abs, ascending=False)

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

def some_func_using_store(main_store):
    df = to_df(main_store)
    if df is None:
        # fallback consistent with original behavior
        return None