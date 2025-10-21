import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from io import StringIO

dash.register_page(__name__, path='/univariada', name='2. Análise Univariada')

# --- Data Dictionaries ---
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

# --- Layout ---
layout = dbc.Container([
    html.H3("🔬 Análise Univariada Detalhada"),
    dcc.Markdown("Explore a distribuição de cada variável. Use os filtros para comparar diferentes gêneros e ajuste os gráficos para uma análise mais profunda."),

    dbc.Row([
        dbc.Col([
            html.Label("Selecione um ou mais gêneros para comparar (opcional):"),
            dcc.Dropdown(id='univar-genre-multiselect', multi=True, placeholder="Comparar distribuições por gênero...")
        ], width=12)
    ], className="my-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Selecione uma variável numérica para análise:"),
            dcc.Dropdown(id='univar-col-select')
        ], width=6),
        dbc.Col([
            html.Label("Número de Bins para o Histograma:"),
            dcc.Slider(10, 100, 10, value=30, id='univar-bins-slider', marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], width=6)
    ]),
    
    html.Div(id='univar-description-info', className="mt-3"),

    html.Hr(),
    html.H4("Visualização da Distribuição", className="mt-4"),
    dbc.Row([
        dbc.Col(dbc.Spinner(dcc.Graph(id='univar-hist-chart')), width=6),
        dbc.Col(dbc.Spinner(dcc.Graph(id='univar-box-chart')), width=6)
    ], className="mb-4"),

    html.Hr(),
    html.H4("Estatísticas e Detecção de Outliers", className="mt-4"),
    dbc.Row([
        dbc.Col(dbc.Spinner(html.Div(id='univar-stats-table')), width=6),
        dbc.Col(dbc.Spinner(html.Div(id='univar-outliers-table')), width=6),
    ]),
], fluid=True)

# --- Callbacks ---
@callback(
    Output('univar-genre-multiselect', 'options'),
    Output('univar-col-select', 'options'),
    Input('main-df-store', 'data')
)
def populate_univar_dropdowns(json_data):
    if json_data is None: return [], []
    df = pd.read_json(StringIO(json_data), orient='split')
    genres = sorted(df['track_genre'].unique().tolist())
    num_cols = sorted(df.select_dtypes(include=np.number).columns.tolist())
    return genres, num_cols

@callback(
    Output('univar-hist-chart', 'figure'),
    Output('univar-box-chart', 'figure'),
    Output('univar-stats-table', 'children'),
    Output('univar-outliers-table', 'children'),
    Output('univar-description-info', 'children'),
    Input('main-df-store', 'data'),
    Input('univar-genre-multiselect', 'value'),
    Input('univar-col-select', 'value'),
    Input('univar-bins-slider', 'value')
)
def update_univar_analysis(json_data, selected_genres, selected_var, num_bins):
    if not json_data or not selected_var:
        empty_fig = go.Figure(layout={"title": "Selecione uma variável para análise"})
        # Return 2 empty figures and 3 empty strings
        return empty_fig, empty_fig, "", "", ""

    df = pd.read_json(StringIO(json_data), orient='split')
    
    # Filter by genre if any are selected
    if selected_genres:
        df_filtered = df[df['track_genre'].isin(selected_genres)]
        color_hue = 'track_genre'
    else:
        df_filtered = df.copy()
        color_hue = None
    
    # Graphs
    hist_fig = px.histogram(df_filtered, x=selected_var, color=color_hue, nbins=num_bins,
                            title=f"Distribuição de {selected_var}")
    box_fig = px.box(df_filtered, x=selected_var, y=color_hue,
                     title=f"Boxplot de {selected_var}", orientation='h')
    box_fig.update_layout(yaxis_title=None)

    # Stats Table
    if selected_genres:
        stats_df = df_filtered.groupby('track_genre')[selected_var].describe().T
        stats_table = [html.H5("Estatísticas por Gênero"), dbc.Table.from_dataframe(stats_df.reset_index(), striped=True, bordered=True)]
    else:
        stats_df = df_filtered[selected_var].describe().to_frame()
        stats_table = [html.H5("Estatísticas Gerais"), dbc.Table.from_dataframe(stats_df.reset_index(), striped=True, bordered=True)]
    
    # Outlier Detection
    q1 = df_filtered[selected_var].quantile(0.25)
    q3 = df_filtered[selected_var].quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df_filtered[(df_filtered[selected_var] < lower_bound) | (df_filtered[selected_var] > upper_bound)]
    
    if outliers.empty:
        outliers_table = dbc.Alert("✨ Nenhum outlier encontrado com base no critério de 1.5 * IQR.", color="success")
    else:
        cols_to_show = ['track_name', 'artists', 'track_genre', selected_var]
        outlier_df_show = outliers[cols_to_show].sort_values(by=selected_var, ascending=False).head(10)
        outliers_table = [
            html.H5(f"{len(outliers)} Outliers Encontrados (Top 10)"),
            dbc.Table.from_dataframe(outlier_df_show, striped=True, bordered=True, hover=True, size='sm')
        ]
        
    # Feature description
    desc_info = dbc.Alert(f"Descrição: {col_descriptions.get(selected_var, 'N/A')}", color="info") if selected_var else ""

    return hist_fig, box_fig, stats_table, outliers_table, desc_info