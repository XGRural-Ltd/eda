import dash
from dash import html, dcc, callback, Input, Output, State, no_update
from io import StringIO
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

dash.register_page(__name__, path='/preprocessamento', name='Pré-processamento', order=4)

# Defaults inspirados no notebook
DEFAULT_NUM_FEATURES = [
    'popularity', 'duration_s', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature'
]
DEFAULT_OUTLIER_FEATURES = ['danceability', 'energy', 'loudness', 'acousticness', 'valence']

layout = dbc.Container([
    html.H3("🧹 Pré-processamento "),
    dcc.Markdown(
        """
**Etapas executadas:**

1. Detecção e remoção de outliers com IsolationForest (em features selecionadas, padronizadas).

2. Imputação dos NaNs nas features numéricas pela mediana.

3. Padronização (StandardScaler) das mesmas features numéricas.

O dataset processado é salvo em cache (Stores) e substitui o dataset principal usado nas demais páginas.
"""
    ),

    # Seleções
    dbc.Row([
        dbc.Col([
            html.Label("Features para detecção de outliers (IsolationForest):"),
            dcc.Dropdown(id='pp-outlier-features', multi=True, placeholder="Selecione as features...",
                         value=None)
        ], md=6),
        dbc.Col([
            html.Label("Features numéricas para imputar/escale (igual notebook):"),
            dcc.Dropdown(id='pp-num-features', multi=True, placeholder="Selecione as features...",
                         value=None)
        ], md=6),
    ], className="mb-2"),

    dbc.Row([
        dbc.Col([
            html.Label("Contaminação (proporção estimada de outliers):"),
            dcc.Slider(0.0, 0.15, 0.01, value=0.05, id='pp-contamination',
                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
        ], md=6),
        dbc.Col([
            html.Div([
                dbc.Button("Detectar outliers", id='pp-detect-btn', color="warning", className="me-2"),
                dbc.Button("Aplicar pré-processamento", id='pp-apply-btn', color="primary"),
            ], className="mt-3")
        ], md=6),
    ], className="mb-3"),

    # Feedback
    dbc.Alert(id='pp-status', color="info", is_open=False, style={"whiteSpace": "pre-line"}),

    # Visualização PCA dos outliers (como no notebook)
    html.Hr(),
    html.H5("Visualização PCA (PC1 x PC2) com outliers"),
    dbc.Spinner(dcc.Graph(id='pp-outlier-pca-graph')),

    html.Hr(),
    html.H5("Prévia do DataFrame processado (head)"),
    dbc.Spinner(html.Div(id='pp-processed-preview')),

    # Stores auxiliares
    dcc.Store(id='pp-df-no-outliers'),
], fluid=True)


# Popular dropdowns com base no dataset
@callback(
    Output('pp-outlier-features', 'options'),
    Output('pp-outlier-features', 'value'),
    Output('pp-num-features', 'options'),
    Output('pp-num-features', 'value'),
    Input('main-df-store', 'data')
)
def _populate_dropdowns(json_data):
    if not json_data:
        return [], [], [], []
    df = pd.read_json(StringIO(json_data), orient='split')
    # Remover colunas Unnamed se existirem
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Sugestões padrão do notebook (apenas se existirem)
    outlier_defaults = [c for c in DEFAULT_OUTLIER_FEATURES if c in df.columns]
    num_defaults = [c for c in DEFAULT_NUM_FEATURES if c in df.columns]

    opts_num = [{'label': c, 'value': c} for c in num_cols]
    return opts_num, outlier_defaults, opts_num, num_defaults


# Helper: ler JSON do Store
def _json_to_df(obj):
    if not obj:
        return None
    try:
        return pd.read_json(StringIO(obj), orient='split')
    except Exception:
        return None


# Detecta outliers com tratamento de erros
@callback(
    Output('pp-outlier-pca-graph', 'figure'),
    Output('pp-status', 'children'),
    Output('pp-status', 'is_open'),
    Output('pp-df-no-outliers', 'data'),
    Input('pp-detect-btn', 'n_clicks'),
    State('main-df-store', 'data'),
    State('pp-outlier-features', 'value'),
    State('pp-contamination', 'value'),
    prevent_initial_call=True
)
def _detect_outliers(n_clicks, main_json, features_for_outliers, contamination):
    import plotly.graph_objects as go
    empty = go.Figure(layout={"title": "Selecione as features e clique em Detectar outliers"})

    try:
        df = _json_to_df(main_json)
        if df is None or df.empty:
            return empty, "Sem dados para analisar.", True, no_update

        # Garantir remoção de Unnamed e duration_ms
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')].copy()
        if 'duration_ms' in df.columns:
            df['duration_s'] = (df['duration_ms'] / 1000).astype(float).round(3)
            df = df.drop(columns=['duration_ms'])

        if not features_for_outliers:
            return empty, "Selecione as features para outliers.", True, no_update

        feats = [c for c in features_for_outliers if c in df.columns]
        if len(feats) < 2:
            return empty, "Escolha pelo menos duas features para detecção/visualização.", True, no_update

        # Padronização para detecção
        X = df[feats]
        X = X.select_dtypes(include=np.number).dropna()
        if X.empty:
            return empty, "As features selecionadas não são numéricas ou estão vazias.", True, no_update

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.astype(float))

        # IsolationForest
        iso = IsolationForest(contamination=float(contamination or 0.05), random_state=42)
        preds = iso.fit_predict(X_scaled)  # 1 = normal, -1 = outlier

        clean_index = X.index
        df_out = df.copy()
        df_out.loc[clean_index, 'anomaly'] = preds
        df_out['anomaly'] = df_out['anomaly'].fillna(1).astype(int)

        # PCA 2D para visualização
        pca = PCA(n_components=2, random_state=42)
        comps = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(comps, columns=['PC1', 'PC2'], index=clean_index)
        df_pca['anomaly'] = preds

        fig = px.scatter(
            df_pca, x='PC1', y='PC2', color='anomaly',
            color_discrete_map={1: 'blue', -1: 'red'},
            title="Outliers detectados (PCA 2D)", opacity=0.7
        )

        df_no_outliers = df_out[df_out['anomaly'] != -1].drop(columns=['anomaly'])
        msg = (
            f"Detectados {(df_out['anomaly'] == -1).sum()} outliers de {len(df_out)} linhas.\n"
            f"Após remoção: {len(df_no_outliers)} linhas."
        )
        return fig, msg, True, df_no_outliers.to_json(date_format='iso', orient='split')

    except Exception as e:
        # Mensagem amigável no toast/alert
        return empty, f"Erro ao detectar outliers: {e}", True, no_update


# Aplica a etapa de imputação (mediana) e padronização (StandardScaler) nas features numéricas
@callback(
    Output('processed-df-store', 'data'),
    Output('main-df-store', 'data', allow_duplicate=True),  # atualizar cache principal
    Output('pp-processed-preview', 'children'),
    Output('pp-status', 'children', allow_duplicate=True),
    Output('pp-status', 'is_open', allow_duplicate=True),
    Input('pp-apply-btn', 'n_clicks'),
    State('pp-df-no-outliers', 'data'),
    State('main-df-store', 'data'),
    State('pp-num-features', 'value'),
    prevent_initial_call=True
)
def _apply_preprocessing(n_clicks, no_outliers_json, main_json, num_features):
    if not main_json or not num_features:
        return no_update, no_update, no_update, "Selecione as features numéricas.", True

    # Usa DF sem outliers, se houver
    if no_outliers_json:
        df = pd.read_json(StringIO(no_outliers_json), orient='split')
    else:
        df = pd.read_json(StringIO(main_json), orient='split')

    # Limpar Unnamed e garantir duration_s
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')].copy()
    if 'duration_ms' in df.columns:
        df['duration_s'] = (df['duration_ms'] / 1000).astype(float).round(3)
        df = df.drop(columns=['duration_ms'])

    feats = [c for c in (num_features or []) if c in df.columns]

    if not feats:
        return no_update, no_update, no_update, "Nenhuma feature válida selecionada.", True

    # Imputação (mediana) e padronização
    df[feats] = df[feats].fillna(df[feats].median())
    scaler = StandardScaler()
    df[feats] = scaler.fit_transform(df[feats].astype(float))

    # Prévia
    preview_cols = [c for c in ['track_name', 'artists', 'track_genre'] if c in df.columns] + [c for c in feats if c not in ['track_name','artists','track_genre']][:8]
    preview = dbc.Table.from_dataframe(df[preview_cols].head(15) if preview_cols else df.head(15),
                                       striped=True, bordered=True, hover=True, responsive=True)

    msg = f"Pré-processamento concluído e salvo em cache: {len(df)} linhas, {len(df.columns)} colunas. Features padronizadas: {', '.join(feats)}."
    processed_json = df.to_json(date_format='iso', orient='split')
    return processed_json, processed_json, preview, msg, True