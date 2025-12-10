import dash
from dash import html, dcc, callback, Input, Output, State, no_update
from src.constants import STORE_MAIN
import pandas as pd
from src.utils import to_df
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import traceback  # ADICIONADO

dash.register_page(__name__, path='/', name='Visão Geral', order=1)

# --- Data Dictionaries (from your original script) ---
cols_dict = {
    'track_id':'Track ID','artists':'Artists','album_name':'Album Name','track_name':'Track Name',
    'popularity':'Popularity','duration_s':'Duration (s)','explicit':'Explicit','danceability':'Danceability',
    'energy':'Energy','key':'Key','loudness':'Loudness','mode':'Mode','speechiness':'Speechiness',
    'acousticness':'Acousticness','instrumentalness':'Instrumentalness','liveness':'Liveness',
    'valence':'Valence','tempo':'Tempo','time_signature':'Time Signature','track_genre':'Track Genre'
}

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
        dbc.Col(dcc.RadioItems(id='geral-dist-plot-type',
                               options=['Histograma', 'Boxplot', 'Ambos'],
                               value='Histograma',
                               inline=True)),
    ]),
    dbc.Spinner(dcc.Graph(id='geral-dist-graph-hist', className="mt-2")),
    dbc.Spinner(dcc.Graph(id='geral-dist-graph-box', className="mt-2")),
    
    html.Hr(),
    html.H3("📈 Gráficos de Dispersão entre Variáveis Numéricas", className="mt-4"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='geral-scatter-x-axis', placeholder="Variável Eixo X...")),
        dbc.Col(dcc.Dropdown(id='geral-scatter-y-axis', placeholder="Variável Eixo Y...")),
        dbc.Col(dbc.Checklist(options=[{'label': 'Mostrar linha de tendência', 'value': 'trend'}], value=[], id='geral-scatter-trend-check'), align="center"),
    ]),
    dbc.Spinner(dcc.Graph(id='geral-scatter-graph', className="mt-2")),
    html.Hr(),
    dcc.Markdown("""
    📌 Danceability vs. Energy
    - Já esperamos uma correlação positiva entre 'danceability' e 'energy', pois músicas mais dançantes tendem a ter mais energia.  
    - A linha de tendência (regressão linear) ajuda a visualizar uma correlação moderadamente positiva. 

    📌 Acousticness vs. Energy
    - Correlação negativa forte esperada, pois músicas acústicas são menos energéticas.
    - A linha de tendência decrescente indica uma relação inversamente proporcional.

    📌 Loudness vs. Energy 
    - Baixa dispersão e ascendência dos pontos mostram uma correlação fortemente positiva (músicas energéticas costumam ser mais altas).
    """),
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
    Input(STORE_MAIN, 'data'),
    prevent_initial_call=False
)
def update_geral_page_components(main_store):
    try:
        df = to_df(main_store)
        if df is None:
            return [], [], html.Div("Nenhum dado carregado."), "", [], [], []
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        # Estatísticas formatadas como no Home.py
        desc = df.describe().rename(columns=cols_dict).T
        desc = desc.drop(columns=['count'], errors='ignore')
        desc = desc.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float, np.floating)) else x)
        desc = desc.reset_index().rename(columns={'index': 'Feature'})
        desc_table = dbc.Table.from_dataframe(desc, striped=True, bordered=True, hover=True, responsive=True)
        
        info_text = [
            html.P(f"Existem {df[df.duplicated()].shape[0]} faixas duplicadas nessa base de dados."),
            html.P(f"Existem {df[df.isnull().any(axis=1)].shape[0]} faixas com dados faltantes.")
        ]
        return all_cols, all_cols[:6], desc_table, info_text, num_cols, num_cols, num_cols
    except Exception as e:
        print("ERROR in update_geral_page_components:", e)
        traceback.print_exc()
        # retornos fallback (mesma forma/quantidade de outputs)
        return [], [], html.Div("Erro interno: verifique o terminal."), "", [], [], []

# Callback for the head table based on multiselect
@callback(
    Output('geral-head-table', 'children'),
    Input(STORE_MAIN, 'data'),
    Input('geral-cols-multiselect', 'value')
)
def update_head_table(main_store, *args):
    # accept extra inputs/triggers safely; main_store is expected to be the first arg
    df = to_df(main_store)
    if df is None:
        return html.Div("Nenhum dado carregado."), ""

    return dbc.Table.from_dataframe(df.head(15), striped=True, bordered=True, hover=True)

# Distribuição: Histograma/Boxplot/Ambos (igual ao Home.py)
@callback(
    Output('geral-dist-graph-hist', 'figure'),
    Output('geral-dist-graph-box', 'figure'),
    Output('geral-dist-graph-hist', 'style'),
    Output('geral-dist-graph-box', 'style'),
    Input(STORE_MAIN, 'data'),
    Input('geral-dist-col-select', 'value'),
    Input('geral-dist-plot-type', 'value')
)
def update_dist_graphs(json_data, selected_col, plot_type):
    try:
        empty = go.Figure(layout={"title": "Selecione uma coluna para visualizar"})
        if not all([json_data, selected_col, plot_type]):
            return empty, empty, {'display': 'block'}, {'display': 'none'}
        df = to_df(json_data).dropna()
        col_name = cols_dict.get(selected_col, selected_col)

        fig_hist = px.histogram(df, x=selected_col, title=f"Histograma de {col_name}", labels={selected_col: col_name})
        fig_box = px.box(df, x=selected_col, title=f"Boxplot de {col_name}", labels={selected_col: col_name})

        show_hist = plot_type in ['Histograma', 'Ambos']
        show_box = plot_type in ['Boxplot', 'Ambos']
        style_hist = {'display': 'block'} if show_hist else {'display': 'none'}
        style_box = {'display': 'block'} if show_box else {'display': 'none'}
        return fig_hist, fig_box, style_hist, style_box
    except Exception as e:
        print("ERROR in update_dist_graphs:", e)
        traceback.print_exc()
        empty = go.Figure(layout={"title": "Erro interno"})
        return empty, empty, {'display': 'block'}, {'display': 'none'}

# Dispersão com lógica do Home.py
@callback(
    Output('geral-scatter-graph', 'figure'),
    Input(STORE_MAIN, 'data'),
    Input('geral-scatter-x-axis', 'value'),
    Input('geral-scatter-y-axis', 'value'),
    Input('geral-scatter-trend-check', 'value')
)
def update_scatter_graph(json_data, x_axis, y_axis, trend_check):
    if not all([json_data, x_axis, y_axis]):
        return go.Figure(layout={"title": "Selecione as variáveis X e Y para visualizar"})
    df = to_df(json_data)
    if df is None:
        return go.Figure(layout={"title": "Nenhum dado"})
    df = df.dropna()

    x_name = cols_dict.get(x_axis, x_axis)
    y_name = cols_dict.get(y_axis, y_axis)

    fig = px.scatter(df, x=x_axis, y=y_axis,
                     title=f"Dispersão entre {x_name} e {y_name}",
                     labels={x_axis: x_name, y_axis: y_name},
                     render_mode="webgl")
    fig.update_traces(marker=dict(color='red', opacity=0.5, size=6), showlegend=False)

    # Se x == y, adicionar uma linha ligando os pontos (equivalente ao plot do Home.py)
    if x_axis == y_axis:
        fig.add_trace(go.Scatter(
            x=df[x_axis], y=df[y_axis],
            mode='lines', line=dict(color='blue', width=1), opacity=0.5, name='Linha'
        ))
    else:
        # Linha de tendência (regressão linear simples) sem depender de statsmodels
        if trend_check and 'trend' in trend_check:
            xy = df[[x_axis, y_axis]].dropna()
            if len(xy) >= 2:
                m, b = np.polyfit(xy[x_axis].astype(float), xy[y_axis].astype(float), 1)
                x_vals = np.linspace(xy[x_axis].min(), xy[x_axis].max(), 100)
                y_pred = m * x_vals + b
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_pred, mode='lines',
                    line=dict(color='blue', width=2), name='Tendência'
                ))
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig