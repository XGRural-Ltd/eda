import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.constants import STORE_MAIN, STORE_PROCESSED, STORE_PCA, STORE_SAMPLED_PCA, STORE_PCA_MODEL
from src.utils import to_df
from src.pipeline import run_pca  # ENSURE THIS IMPORT

dash.register_page(__name__, path='/reducao', name='Redução de Dimensionalidade', order=5)

# Default numeric features to select for PCA (if they exist in dataset)
DEFAULT_PCA_FEATURES = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'popularity', 'duration_s']

# === ADD STORES AT PAGE TOP (before layout definition) ===
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

    # --- ELBOW: gráfico automático ligado ao pca-df-store ---
    html.Hr(),
    html.Div([
        html.H5("Elbow Method (Inércia vs k) "),
        dcc.Graph(id='elbow-plot', config={'displayModeBar': False}),
        html.Label("Valor máximo de k para o Elbow:"),
        dcc.Slider(2, 20, 1, value=6, id='elbow-kmax-slider', marks=None,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Div(id='elbow-status-div', className="mt-2"),

        # NearestNeighbors (k-distance) para ajudar a escolher eps do DBSCAN
        html.H6("k-distance (NearestNeighbors) — ajuda a escolher eps para DBSCAN", className="mt-3"),
        html.Label("k (número de vizinhos) para k-distance:"),
        dcc.Slider(1, 50, 1, value=4, id='nn-k-slider', marks=None,
                   tooltip={"placement": "bottom", "always_visible": True}),
        dcc.Graph(id='nn-plot', config={'displayModeBar': False}),
    ], className="my-3"),

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

    # Stores usados pelos callbacks (armazenam PCA, modelo e meta)
], fluid=True)

def _json_to_df(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        try:
            return to_df(obj)
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
    Input(STORE_PROCESSED, 'data'),
    Input(STORE_MAIN, 'data')
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
    Output('pca-df-store', 'data'),
    Output('sampled-pca-df-store', 'data'),
    Output(STORE_PCA_MODEL, 'data'),
    Input('pca-run-btn', 'n_clicks'),
    State('processed-df-store', 'data'),
    State('pca-features-dropdown', 'value'),
    State('pca-n-components-slider', 'value'),
    prevent_initial_call=True
)
def perform_pca(n_clicks, processed_store, features, n_components):
    print(f"[PCA] perform_pca called: n_clicks={n_clicks}, features={features}, n_components={n_components}")
    
    empty = go.Figure(layout={"title": "Selecione features e execute o PCA", "template": "plotly_dark"})
    
    if not n_clicks:
        print("[PCA] No clicks yet")
        return empty, no_update, no_update, no_update, no_update

    df_proc = to_df(processed_store)
    if df_proc is None or df_proc.empty:
        warn = dbc.Alert([
            html.H5("⚠️ Dados processados não encontrados", className="alert-heading"),
            html.P("Você precisa executar o pré-processamento primeiro:"),
            html.Ol([
                html.Li("Vá para a página 'Pré-processamento'"),
                html.Li("Execute o pré-processamento"),
                html.Li("Volte aqui e execute o PCA")
            ])
        ], color="warning")
        print("[PCA] No processed data")
        return empty, warn, no_update, no_update, no_update

    if not features or len(features) < 2:
        warn = dbc.Alert("Selecione ao menos 2 features para PCA.", color="warning")
        print("[PCA] Not enough features selected")
        return empty, warn, no_update, no_update, no_update

    try:
        print(f"[PCA] Running PCA with {len(features)} features, n_components={n_components}")
        out = run_pca(processed_store, features=features, n_components=n_components)
        print(f"[PCA] PCA output keys: {list(out.keys())}")
        
        # Check if PCA succeeded
        if 'pca_store' not in out or not out['pca_store']:
            raise ValueError("run_pca did not return valid pca_store")
            
        # Build variance explained figure (robust: converte para % se necessário e melhora visibilidade)
        explained_var = out.get('explained_variance', [])
        cumulative_var = out.get('cumulative_variance', [])

        def _to_percent(arr):
            arr = np.array(arr, dtype=float)
            if arr.size == 0:
                return arr
            if arr.max() <= 1.01:
                return arr * 100.0
            return arr

        print(f"[PCA] explained_var (raw): {explained_var[:8]}... total={len(explained_var)}")
        print(f"[PCA] cumulative_var (raw): {cumulative_var[:8]}... total={len(cumulative_var)}")

        if explained_var and cumulative_var:
            ev = _to_percent(explained_var)
            cv = _to_percent(cumulative_var)
            labels = [f'PC{i+1}' for i in range(len(ev))]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=labels,
                y=ev,
                name='Variância explicada (%)',
                marker_color='rgba(100,150,255,0.95)',
                hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=labels,
                y=cv,
                name='Variância acumulada (%)',
                mode='lines+markers',
                line=dict(color='orange', width=3),
                marker=dict(color='orange', size=7, line=dict(color='rgba(0,0,0,0.6)', width=0.5)),
                yaxis='y2',
                hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
            ))
            fig.update_layout(
                title=f'Variância Explicada - PCA com {len(ev)} componentes',
                template='plotly_dark',
                yaxis=dict(title='Variância explicada (%)', rangemode='tozero'),
                yaxis2=dict(title='Acumulada (%)', overlaying='y', side='right', rangemode='tozero'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            print(f"[PCA] Created variance figure with {len(ev)} components (converted to % if needed)")
        else:
            fig = go.Figure(layout={"title": "PCA concluído (sem dados de variância)", "template": "plotly_dark"})
            print("[PCA] No variance data available")

        # Build result table
        pca_df = pd.DataFrame(**out['pca_store'])
        n_samples = len(pca_df)
        n_comps = len([c for c in pca_df.columns if c.startswith('PC')])
        table = html.Div([
            html.P(f"✓ PCA concluído: {n_comps} componentes, {n_samples} amostras"),
            html.P(f"Variância explicada acumulada: {cumulative_var[-1]:.1f}%") if cumulative_var else None
        ])

        print(f"[PCA] Returning: fig, table, pca_store ({n_samples} rows), sampled_store, pca_model")
        return fig, table, out['pca_store'], out.get('sampled_store'), out.get('pca_model')

    except Exception as e:
        warn = dbc.Alert(f"Erro ao executar PCA: {type(e).__name__}: {e}", color="danger")
        print(f"[PCA] Error: {e}")
        import traceback
        traceback.print_exc()
        return empty, warn, no_update, no_update, no_update

@callback(
    Output('elbow-plot', 'figure'),
    Input(STORE_PCA, 'data'),
    Input('elbow-kmax-slider', 'value')
)
def update_elbow_on_pca(pca_dict, kmax):
    import plotly.express as _px
    empty = _px.line(title="Execute PCA para gerar o Elbow automaticamente", template="plotly_dark")
    if not pca_dict:
        return empty
    try:
        df_pca = pd.DataFrame(**pca_dict)
    except Exception:
        return empty

    # usa primeiras 2 colunas do PCA (se existirem) para o Elbow inertia (visual)
    if df_pca.shape[1] >= 2:
        X_elbow = df_pca.iloc[:, :2].values
    else:
        X_elbow = df_pca.iloc[:, :1].values

    SAMPLE_SIZE = 5000
    if X_elbow.shape[0] > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_elbow.shape[0], SAMPLE_SIZE, replace=False)
        X_sample = X_elbow[idx]
    else:
        X_sample = X_elbow

    try:
        kmax = int(kmax or 6)
    except Exception:
        kmax = 6
    k_range = list(range(1, max(1, kmax) + 1))

    inertias = []
    from sklearn.cluster import KMeans
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sample)
        inertias.append(km.inertia_)

    fig = _px.line(x=k_range, y=inertias, markers=True,
                   labels={'x': 'k', 'y': 'Inércia'},
                   title="Elbow Method (Inércia vs k)",
                   template="plotly_dark")
    fig.update_traces(marker=dict(size=6, color='rgba(255,255,255,0.95)', line=dict(width=0.3, color='rgba(0,0,0,0.35)')),
                      line=dict(color='rgba(200,200,200,0.9)'))
    fig.update_layout(xaxis=dict(dtick=1))
    return fig

# NEW callback: k-distance (NearestNeighbors) plot — usa o MESMO PCA (todas as componentes)
@callback(
    Output('nn-plot', 'figure'),
    Input(STORE_PCA, 'data'),
    Input('nn-k-slider', 'value')
)
def update_nn_plot(pca_dict, k_neighbors):
    import plotly.graph_objects as _go
    from sklearn.neighbors import NearestNeighbors

    empty_fig = _go.Figure(layout={"title": "k-distance (NearestNeighbors) — carregue PCA primeiro", "template": "plotly_dark"})
    if not pca_dict:
        return empty_fig
    try:
        df_pca = pd.DataFrame(**pca_dict)
    except Exception:
        return empty_fig

    # use todas as componentes do PCA para medir distâncias (mesmo espaço usado para cluster)
    X = df_pca.values
    SAMPLE_SIZE = 5000
    if X.shape[0] > SAMPLE_SIZE:
        rng = np.random.RandomState(42)
        idx = rng.choice(X.shape[0], SAMPLE_SIZE, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    # garantir k válido
    try:
        k = max(1, int(k_neighbors or 4))
    except Exception:
        k = 4
    if X_sample.shape[0] <= k:
        return _go.Figure(layout={"title": f"Sample size ({X_sample.shape[0]}) <= k ({k}), aumente o sample ou reduza k", "template": "plotly_dark"})

    try:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
        distances, _ = nbrs.kneighbors(X_sample)
        # k-distance: distância ao k-ésimo vizinho
        k_distances = distances[:, k-1]
        k_distances_sorted = np.sort(k_distances)
        # plot crescente (forma típica do k-distance plot)
        fig = _go.Figure()
        fig.add_trace(_go.Scatter(x=np.arange(len(k_distances_sorted)), y=k_distances_sorted,
                                  mode='lines+markers', marker=dict(size=4, color='cyan'),
                                  line=dict(color='white')))
        fig.update_layout(title=f'k-distance plot (k={k}) — procurar "cotovelo" para escolher eps',
                          xaxis_title='Pontos ordenados',
                          yaxis_title=f'Distância ao {k}º vizinho',
                          template='plotly_dark')
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _go.Figure(layout={"title": f"Erro ao calcular NearestNeighbors: {type(e).__name__}", "template": "plotly_dark"})

# Toast de salvar: usa o botão correto e só LÊ do Store
@callback(
    Output('save-pca-toast', 'is_open'),
    Output('save-pca-toast', 'children'),
    Input('save-pca-df-button', 'n_clicks'),
    State(STORE_PCA, 'data'),
    State(STORE_PCA_MODEL, 'data'),
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