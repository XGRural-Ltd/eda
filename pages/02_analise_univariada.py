import dash
from dash import html, dcc, callback, Input, Output
from src.constants import STORE_MAIN
from src.utils import to_df
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/univariada', name='Análise Univariada', order = 2)

# --- Labels (iguais ao Home.py) ---
cols_dict = {
    'track_id':'Track ID','artists':'Artists','album_name':'Album Name','track_name':'Track Name',
    'popularity':'Popularity','duration_s':'Duration (s)','explicit':'Explicit','danceability':'Danceability',
    'energy':'Energy','key':'Key','loudness':'Loudness','mode':'Mode','speechiness':'Speechiness',
    'acousticness':'Acousticness','instrumentalness':'Instrumentalness','liveness':'Liveness',
    'valence':'Valence','tempo':'Tempo','time_signature':'Time Signature','track_genre':'Track Genre'
}

# --- Descrições (iguais ao Home.py) ---
col_descriptions = {
    "track_id": "O ID do Spotify para a faixa.",
    "artists": "Nomes dos artistas que performaram a faixa. Se houver mais de um, são separados por ponto e vírgula.",
    "album_name": "Nome do álbum no qual a faixa aparece.",
    "track_name": "Nome da faixa.",
    "popularity": "Popularidade da faixa (0 a 100), baseada em número e recência de reproduções.",
    "duration_ms": "Duração da faixa em milissegundos.",
    "explicit": "Indica se a faixa possui conteúdo explícito (True = sim, False = não).",
    "danceability": "Quão dançante é a faixa, de 0.0 (menos) a 1.0 (mais dançante).",
    "energy": "Energia percebida da faixa, de 0.0 a 1.0.",
    "key": "Tom da música (0 = Dó, 1 = Dó♯/Ré♭, ..., -1 = indetectável).",
    "loudness": "Volume geral da faixa em decibéis (dB).",
    "mode": "Modalidade: 1 = maior, 0 = menor.",
    "speechiness": "Detecta presença de fala. 1.0 = fala pura; 0.0 = música pura.",
    "acousticness": "Confiança de que a faixa é acústica (0.0 a 1.0).",
    "instrumentalness": "Probabilidade de não conter vocais. Próximo de 1.0 = instrumental.",
    "liveness": "Probabilidade de ter sido gravada ao vivo. Acima de 0.8 = performance ao vivo.",
    "valence": "Quão positiva é a música (0.0 = triste, 1.0 = alegre).",
    "tempo": "Tempo estimado da faixa (batidas por minuto).",
    "time_signature": "Compasso estimado (de 3 a 7).",
    "track_genre": "Gênero musical da faixa."
}

# --- Layout ---
layout = dbc.Container([
    html.H3("🔬 Análise Univariada Detalhada"),
    dcc.Markdown("Explore a distribuição de cada variável. Use os filtros para comparar diferentes gêneros e ajuste os gráficos."),

    dbc.Row([
        dbc.Col([
            html.Label("Comparar distribuições por gênero (opcional):"),
            dcc.Dropdown(id='univar-genre-multiselect', multi=True, placeholder="Comparar por gênero...")
        ], width=12)
    ], className="my-3"),

    dbc.Row([
        dbc.Col([
            html.Label("Variável numérica:"),
            dcc.Dropdown(id='univar-col-select', placeholder="Selecione a variável...")
        ], width=6),
        dbc.Col([
            html.Label("Bins do histograma:"),
            dcc.Slider(10, 100, 10, value=30, id='univar-bins-slider',
                       marks=None, tooltip={"placement": "bottom", "always_visible": True})
        ], width=6)
    ]),

    dbc.Alert(id='univar-description-info', color="info", is_open=False, className="mt-3"),

    html.Hr(),
    html.H4("Visualização da Distribuição"),
    dbc.Row([
        dbc.Col(dbc.Spinner(dcc.Graph(id='univar-hist-chart')), width=6),
        dbc.Col(dbc.Spinner(dcc.Graph(id='univar-box-chart')), width=6)
    ], className="mb-4"),

    html.Hr(),
    html.H4("Estatísticas e Detecção de Outliers"),
    dbc.Row([
        dbc.Col(dbc.Spinner(html.Div(id='univar-stats-table')), width=6),
        dbc.Col(dbc.Spinner(html.Div(id='univar-outliers-table')), width=6),
    ]),
], fluid=True)

# --- Callbacks ---
@callback(
    Output('univar-genre-multiselect', 'options'),
    Output('univar-col-select', 'options'),
    Input(STORE_MAIN, 'data')
)
def populate_univar_dropdowns(main_store):
    df = to_df(main_store)
    if df is None:
        return [], [], []

    # Gêneros com label capitalizado (comportamento próximo do Home.py)
    if 'track_genre' in df.columns:
        genres_raw = sorted(df['track_genre'].dropna().unique().tolist())
        genre_options = [{"label": g.capitalize(), "value": g} for g in genres_raw]
    else:
        genre_options = []

    # Variáveis numéricas com rótulos amigáveis
    num_cols = sorted(df.select_dtypes(include=np.number).columns.tolist())
    num_options = [{"label": cols_dict.get(c, c), "value": c} for c in num_cols]

    return genre_options, num_options

@callback(
    Output('univar-hist-chart', 'figure'),
    Output('univar-box-chart', 'figure'),
    Output('univar-stats-table', 'children'),
    Output('univar-outliers-table', 'children'),
    Output('univar-description-info', 'children'),
    Output('univar-description-info', 'is_open'),
    Input(STORE_MAIN, 'data'),
    Input('univar-genre-multiselect', 'value'),
    Input('univar-col-select', 'value'),
    Input('univar-bins-slider', 'value')
)
def update_univar_analysis(json_data, selected_genres, selected_var, num_bins):
    empty_fig = go.Figure(layout={"title": "Selecione uma variável para análise"})
    if not json_data or not selected_var:
        return empty_fig, empty_fig, "", "", "", False

    df = to_df(json_data)

    # Filtro por gênero (igual à ideia do Home.py)
    if selected_genres:
        df_filtered = df[df['track_genre'].isin(selected_genres)].copy()
        color_hue = 'track_genre'
    else:
        df_filtered = df.copy()
        color_hue = None

    df_filtered = df_filtered.dropna(subset=[selected_var])

    # Nomes amigáveis
    var_name = cols_dict.get(selected_var, selected_var)

    # Gráficos
    hist_fig = px.histogram(
        df_filtered, x=selected_var, color=color_hue, nbins=num_bins,
        title=f"Distribuição de {var_name}",
        labels={selected_var: var_name, "track_genre": cols_dict.get("track_genre")}
    )
    if color_hue:
        hist_fig.update_layout(legend_title_text="Gênero")

    box_fig = px.box(
        df_filtered, x=selected_var, y=color_hue,
        title=f"Boxplot de {var_name}",
        labels={selected_var: var_name, "track_genre": cols_dict.get("track_genre")},
        orientation='h'
    )
    box_fig.update_layout(yaxis_title=None)

    # Estatísticas (geral ou por gênero)
    if color_hue:
        stats_df = df_filtered.groupby('track_genre')[selected_var].describe().T
        stats_df = stats_df.round(3).reset_index().rename(columns={'index': 'Métrica'})
        stats_table = [
            html.H5("Estatísticas por Gênero"),
            dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, hover=True, responsive=True)
        ]
    else:
        stats_df = df_filtered[selected_var].describe().round(3).to_frame()
        stats_table = [
            html.H5("Estatísticas Gerais"),
            dbc.Table.from_dataframe(stats_df.reset_index().rename(columns={'index': 'Métrica'}), striped=True, bordered=True, hover=True, responsive=True)
        ]

    # Outliers (1.5 * IQR) — igual lógica do Home.py
    q1 = df_filtered[selected_var].quantile(0.25)
    q3 = df_filtered[selected_var].quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df_filtered[(df_filtered[selected_var] < lower_bound) | (df_filtered[selected_var] > upper_bound)]

    if outliers.empty:
        outliers_table = dbc.Alert("Não foram encontrados outliers com base no critério de 1.5 * IQR. ✨", color="success")
    else:
        cols_to_show = [c for c in ['track_name', 'artists', 'track_genre', selected_var] if c in outliers.columns]
        outliers_table = [
            html.H5(f"{len(outliers)} Outliers Encontrados (Top 10)"),
            dbc.Table.from_dataframe(
                outliers[cols_to_show].sort_values(by=selected_var, ascending=False).head(10),
                striped=True, bordered=True, hover=True, size='sm', responsive=True
            )
        ]

    # Descrição da feature (como no Home.py)
    desc_text = f"Descrição: {col_descriptions.get(selected_var, 'N/A')}"
    return hist_fig, box_fig, stats_table, outliers_table, desc_text, True