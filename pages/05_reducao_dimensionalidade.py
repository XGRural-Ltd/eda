import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, no_update, dash_table
import pandas as pd, numpy as np, plotly.express as px
from io import StringIO, BytesIO
import json
import joblib
import base64

dash.register_page(__name__, path='/reducao', name='Redução de Dimensionalidade', order=5)

DEFAULT_PCA_FEATURES = [
    'danceability', 'energy', 'loudness', 'acousticness', 'valence',
    'tempo', 'popularity', 'duration_s'
]

# --- Layout ---
layout = dbc.Container([
    html.H3("📉 Redução de Dimensionalidade (PCA)"),
    dcc.Markdown("""
    A Análise de Componentes Principais (PCA) nos ajuda a 'comprimir' as informações mais importantes de muitas features em um número menor de componentes.
    """),
    html.Div(id='pca-warning-div'),

    # Controles
    dbc.Row([
        dbc.Col([
            html.Label("Features numéricas:"),
            dcc.Dropdown(id='pca-features-dropdown', multi=True, placeholder="Selecione as features numéricas...")
        ], md=8),
        dbc.Col([
            html.Label("Número de componentes principais:"),
            dcc.Slider(2, 20, 1, value=10, id='pca-n-components-slider',
                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
            html.Div(className="mt-2"),
            dbc.Button("Gerar PCA", id='pca-run-btn', color="primary", className="mt-1"),
        ], md=4),
    ], className="my-3"),

    dbc.Spinner(dcc.Graph(id='pca-variance-graph')),

    html.H5("Amostra dos Dados Transformados pelo PCA", className="mt-4"),
    dbc.Spinner(html.Div(id='pca-result-table')),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            dbc.Button("Salvar Dados do PCA para Próximas Etapas", id='save-pca-df-button', color="primary", n_clicks=0),
        ], width=12, className="mt-4 text-center")
    ], className="mb-4"),

    # Toast para confirmação
    dbc.Toast(
        id="save-pca-toast",
        header="Sucesso!",
        icon="success",
        duration=4000,
        is_open=False,
        style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 9999},
    ),

    # Stores
    html.Div(id='pca-df-json-storage', style={'display': 'none'})
], fluid=True)

def _json_to_df(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            return pd.read_json(StringIO(obj), orient='split')
        except Exception:
            return None
    if isinstance(obj, dict) and {'data','columns'}.issubset(obj):
        return pd.DataFrame(obj['data'], columns=obj['columns'])
    return None

def _choose_df(proc_json, main_json):
    df_proc = _json_to_df(proc_json)
    df_main = _json_to_df(main_json)
    df = df_proc if df_proc is not None else df_main
    if df is None:
        return None
    # limpar Unnamed e garantir duration_s
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')].copy()
    if 'duration_ms' in df.columns:
        df['duration_s'] = (df['duration_ms'] / 1000).astype(float).round(3)
        df = df.drop(columns=['duration_ms'])
    return df

# Popular dropdown de features com as colunas numéricas
@callback(
    Output('pca-features-dropdown', 'options'),
    Output('pca-features-dropdown', 'value'),
    Input('processed-df-store', 'data'),
    Input('main-df-store', 'data')
)
def populate_pca_features(proc_json, main_json):
    df = _choose_df(proc_json, main_json)
    if df is None or df.empty:
        return [], []
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    options = [{"label": c, "value": c} for c in num_cols]
    defaults = [c for c in DEFAULT_PCA_FEATURES if c in num_cols]
    # se não achar defaults, escolha até 8 numéricas
    if not defaults:
        defaults = num_cols[:8]
    return options, defaults

# --- Callbacks ---
@callback(
    Output('pca-variance-graph', 'figure'),
    Output('pca-result-table', 'children'),
    Output('pca-warning-div', 'children'),
    Output('pca-df-store', 'data'),
    Output('pca-model-store', 'data'),
    Output('scaler-store', 'data'),
    Output('pca-features-store', 'data'),
    Input('pca-run-btn', 'n_clicks'),
    State('processed-df-store', 'data'),
    State('main-df-store', 'data'),
    State('pca-features-dropdown', 'value'),
    State('pca-n-components-slider', 'value'),
    prevent_initial_call=True
)
def perform_pca(n_clicks, processed_data, main_data, features, n_components):
    import plotly.graph_objects as go
    empty = go.Figure(layout={"title": "Selecione features e execute o PCA"})
    no_update_list = [no_update] * 4 # For models and features

    df = _choose_df(processed_data, main_data)
    if df is None or df.empty or not features:
        return empty, "", "Dados ou features inválidos.", *no_update_list

    feats = [c for c in (features or []) if c in df.columns]
    if not feats:
        return empty, "", "Nenhuma feature válida para PCA.", *no_update_list

    X = df[feats].select_dtypes(include=np.number).dropna()
    if X.empty:
        return empty, "", "As features selecionadas não são numéricas ou estão vazias.", *no_update_list

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(float))

    n_comp = int(n_components or 2)
    n_comp = max(2, min(n_comp, Xs.shape[1]))
    pca = PCA(n_components=n_comp)
    comps = pca.fit_transform(Xs)

    # --- Define variables for graph and table ---
    ratio = pca.explained_variance_ratio_
    cum_ratio = np.cumsum(ratio)
    components_labels = [f"PC{i+1}" for i in range(n_comp)]

    # --- Serialize and save models and features ---
    def serialize_model(model):
        mem_buffer = BytesIO()
        joblib.dump(model, mem_buffer)
        mem_buffer.seek(0)
        return base64.b64encode(mem_buffer.read()).decode('utf-8')

    b64_pca_model = serialize_model(pca)
    b64_scaler_model = serialize_model(scaler)
    
    # Gráfico: barras (variância) + linha (acumulada)
    var_fig = go.Figure()
    var_fig.add_bar(x=components_labels, y=ratio, name="Variância explicada")
    var_fig.add_scatter(x=components_labels, y=cum_ratio, mode="lines+markers", name="Variância acumulada")
    var_fig.update_layout(
        title="Variância explicada e acumulada por componente",
        yaxis=dict(title="Proporção", range=[0, 1]),
        xaxis_title="Componentes",
        legend=dict(orientation="h", y=-0.2)
    )

    # Tabela com coluna de variância acumulada
    res_table = pd.DataFrame({
        "Componente": components_labels,
        "Variância Explicada": np.round(ratio, 4),
        "Variância Acumulada": np.round(cum_ratio, 4)
    })
    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in res_table.columns],
        data=res_table.to_dict("records"),
        style_table={"overflowX": "auto"}
    )

    pca_df = pd.DataFrame(comps, index=X.index, columns=[f"PC{i+1}" for i in range(comps.shape[1])])
    pca_dict = pca_df.to_dict(orient='split')
    return var_fig, table, "", pca_dict, b64_pca_model, b64_scaler_model, feats

# Toast de salvar: usa o botão correto e só LÊ do Store
@callback(
    Output('save-pca-toast', 'is_open'),
    Output('save-pca-toast', 'children'),
    Input('save-pca-df-button', 'n_clicks'),
    State('pca-df-store', 'data'),
    State('pca-model-store', 'data'),
    prevent_initial_call=True
)
def save_pca(n, pca_json, pca_model_b64):
    if not n:
        return False, ""
    
    # Check if both data and the model were generated before confirming
    if pca_json and pca_model_b64:
        msg = "Dados e modelos do PCA foram salvos com sucesso!"
    else:
        msg = "Nenhum resultado de PCA para salvar. Execute o PCA primeiro."
        
    return True, msg