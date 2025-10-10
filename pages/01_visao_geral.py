import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from io import StringIO

dash.register_page(__name__, path='/', name='1. Visão Geral')

# --- Data Dictionaries (from your original script) ---
cols_dict = {
    'popularity': 'Popularity', 'duration_ms': 'Duration (ms)', 'explicit': 'Explicit', 'danceability': 'Danceability',
    'energy': 'Energy', 'key': 'Key', 'loudness': 'Loudness', 'mode': 'Mode', 'speechiness': 'Speechiness',
    'acousticness': 'Acousticness', 'instrumentalness': 'Instrumentalness', 'liveness': 'Liveness',
    'valence': 'Valence', 'tempo': 'Tempo', 'time_signature': 'Time Signature', 'track_genre': 'Track Genre'
}
col_descriptions = {
    "popularity": "Popularidade da faixa (0 a 100), baseada em número e recência de reproduções.",
    "duration_ms": "Duração da faixa em milissegundos.",
    "explicit": "Indica se a faixa possui conteúdo explícito (True = sim, False = não).",
    "danceability": "Quão dançante é a faixa, de 0.0 (menos) a 1.0 (mais dançante).",
    "energy": "Energia percebida da faixa, de 0.0 a 1.0.",
    "loudness": "Volume geral da faixa em decibéis (dB).",
    "valence": "Quão positiva é a música (0.0 = triste, 1.0 = alegre).",
    "tempo": "Tempo estimado da faixa (batidas por minuto)."
}
df_descriptions = pd.DataFrame.from_dict(col_descriptions, orient='index', columns=['Descrição'])

# --- Layout ---
layout = dbc.Container([
    html.H3("📊 Informações Gerais do Dataset"),
    dcc.Markdown("""
    Nesta etapa vamos ficar mais familiarizados com os dados. Vamos explorar as colunas, tipos de dados, 
    valores ausentes, estatísticas descritivas e visualizar alguns plots.
    """),

    dcc.Markdown("**Visualize o DataFrame com as colunas selecionadas:**"),
    dcc.Dropdown(id='geral-cols-multiselect', multi=True, placeholder="Selecione colunas para exibir..."),
    dbc.Spinner(html.Div(id='geral-head-table', className="mt-2")),

    dcc.Markdown("**Estatísticas descritivas:**", className="mt-4"),
    dbc.Spinner(html.Div(id='geral-describe-table')),
    html.Div(id='geral-info-text', className="mt-2"),
    
    html.Hr(),
    html.H3("🧾 Dicionário de Dados: Descrição das Colunas", className="mt-4"),
    dbc.Table.from_dataframe(df_descriptions.reset_index(), striped=True, bordered=True, hover=True, header=["Feature", "Descrição"]),
    
    html.Hr(),
    html.H3("📈 Visualizações Gerais de Distribuição", className="mt-4"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='geral-dist-col-select', placeholder="Selecione uma coluna numérica...")),
        dbc.Col(dcc.RadioItems(id='geral-dist-plot-type', options=['Histograma', 'Boxplot'], value='Histograma', inline=True)),
    ]),
    dbc.Spinner(dcc.Graph(id='geral-dist-graph', className="mt-2")),
    
    html.Hr(),
    html.H3("📈 Gráficos de Dispersão entre Variáveis Numéricas", className="mt-4"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='geral-scatter-x-axis', placeholder="Variável Eixo X...")),
        dbc.Col(dcc.Dropdown(id='geral-scatter-y-axis', placeholder="Variável Eixo Y...")),
        dbc.Col(dbc.Checklist(options=[{'label': 'Mostrar linha de tendência', 'value': 'trend'}], value=[], id='geral-scatter-trend-check'), align="center"),
    ]),
    dbc.Spinner(dcc.Graph(id='geral-scatter-graph', className="mt-2")),
], fluid=True)


# --- Callbacks ---

# Main callback to populate initial components based on the main dataframe
@callback(
    Output('geral-cols-multiselect', 'options'),
    Output('geral-cols-multiselect', 'value'),
    Output('geral-describe-table', 'children'),
    Output('geral-info-text', 'children'),
    Output('geral-dist-col-select', 'options'),
    Output('geral-scatter-x-axis', 'options'),
    Output('geral-scatter-y-axis', 'options'),
    Input('main-df-store', 'data')
)
def update_geral_page_components(json_data):
    if json_data is None: return [no_update] * 7
    df = pd.read_json(StringIO(json_data), orient='split')
    
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    desc = df.describe().T.drop(columns=['count']).reset_index()
    desc_table = dbc.Table.from_dataframe(desc, striped=True, bordered=True, hover=True, responsive=True)
    
    info_text = [
        html.P(f"Existem {df[df.duplicated()].shape[0]} faixas duplicadas nessa base de dados."),
        html.P(f"Existem {df[df.isnull().any(axis=1)].shape[0]} faixas com dados faltantes.")
    ]
    
    return all_cols, all_cols[:6], desc_table, info_text, num_cols, num_cols, num_cols

# Callback for the head table based on multiselect
@callback(
    Output('geral-head-table', 'children'),
    Input('main-df-store', 'data'),
    Input('geral-cols-multiselect', 'value')
)
def update_head_table(json_data, selected_cols):
    if not json_data or not selected_cols: return "Selecione colunas para exibir a tabela."
    df = pd.read_json(StringIO(json_data), orient='split')
    return dbc.Table.from_dataframe(df[selected_cols].head(15), striped=True, bordered=True, hover=True)

# Callback for distribution graph
@callback(
    Output('geral-dist-graph', 'figure'),
    Input('main-df-store', 'data'),
    Input('geral-dist-col-select', 'value'),
    Input('geral-dist-plot-type', 'value')
)
def update_dist_graph(json_data, selected_col, plot_type):
    if not all([json_data, selected_col, plot_type]): return go.Figure(layout={"title": "Selecione uma coluna para visualizar"})
    df = pd.read_json(StringIO(json_data), orient='split')
    col_name = cols_dict.get(selected_col, selected_col)
    if plot_type == 'Histograma':
        fig = px.histogram(df, x=selected_col, title=f"Histograma de {col_name}", labels={selected_col: col_name})
    else:
        fig = px.box(df, x=selected_col, title=f"Boxplot de {col_name}", labels={selected_col: col_name})
    return fig

# Callback for scatter plot
@callback(
    Output('geral-scatter-graph', 'figure'),
    Input('main-df-store', 'data'),
    Input('geral-scatter-x-axis', 'value'),
    Input('geral-scatter-y-axis', 'value'),
    Input('geral-scatter-trend-check', 'value')
)
def update_scatter_graph(json_data, x_axis, y_axis, trend_check):
    if not all([json_data, x_axis, y_axis]): return go.Figure(layout={"title": "Selecione as variáveis X e Y para visualizar"})
    df = pd.read_json(StringIO(json_data), orient='split')
    x_name = cols_dict.get(x_axis, x_axis)
    y_name = cols_dict.get(y_axis, y_axis)
    trendline = "ols" if 'trend' in trend_check else None
    fig = px.scatter(df, x=x_axis, y=y_axis, trendline=trendline,
                     title=f"Dispersão entre {x_name} e {y_name}",
                     labels={x_axis: x_name, y_axis: y_name})
    return fig