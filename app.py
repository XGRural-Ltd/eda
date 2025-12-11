import os
import sys
import pkgutil
import importlib

from dash import Dash, dcc, html
import dash as dash
import dash_bootstrap_components as dbc
import plotly.io as pio
import pandas as pd

from src.constants import (
    STORE_MAIN,
    STORE_PROCESSED,
    STORE_PCA,
    STORE_SAMPLED_PCA,
    STORE_CLUSTER_LABELS,
    STORE_PREDICTION_MODEL,
    STORE_PCA_MODEL,
    STORE_SCALER,
    STORE_PCA_FEATURES,
    STORE_CLUSTER_MODEL,   # <--- added
)

# Restaurar tema escuro (visual anterior)
pio.templates.default = "plotly_dark"
BOOTSTRAP_THEME = dbc.themes.CYBORG  # tema escuro do Bootstrap

# 1) Criar app ANTES de importar/registrar páginas
app = Dash(__name__, use_pages=True, external_stylesheets=[BOOTSTRAP_THEME])
server = app.server

# 2) Importar todas as pages somente após a criação do app
pages_dir = os.path.join(os.path.dirname(__file__), "pages")
if os.path.isdir(pages_dir):
    sys.path.insert(0, os.path.dirname(__file__))
    for finder, name, ispkg in pkgutil.iter_modules([pages_dir]):
        importlib.import_module(f"pages.{name}")

# Pequena função para montar a sidebar a partir de dash.page_registry
def _build_sidebar():
    from dash import page_registry
    links = []
    for p in page_registry.values():
        path = p.get("path")
        if not path:
            continue
        links.append(dbc.NavLink(p.get("name", p["module"]), href=path, active="exact", className="mb-1"))
    nav = dbc.Nav(links, vertical=True, pills=True, className="flex-column")
    return dbc.Card(dbc.CardBody([html.H5("Navegação"), nav]), color="dark", inverse=True)

# Tenta carregar dataset inicial (opcional)
def _load_initial_dataset():
    candidates = [
        os.path.join(os.getcwd(), "data", "dataset.csv"),
        os.path.join(os.getcwd(), "data", "dataset.parquet"),
        os.path.join(os.getcwd(), "data", "spotify_dataset.csv"),
        os.path.join(os.getcwd(), "dataset.csv"),
        os.path.join(os.getcwd(), "spotify_dataset.csv"),
        os.path.join(os.getcwd(), "dataset.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".parquet"):
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p, index_col=0)
                print(f"Loaded dataset from {p} (shape: {df.shape})")
                return df.to_dict(orient="split")
            except Exception as e:
                print(f"Failed to read {p}: {e}")

    # tentativa via kagglehub (opcional)
    try:
        import kagglehub
        print("Nenhum arquivo local — tentando baixar via kagglehub...")
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        csv_path = os.path.join(path, "dataset.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            print(f"Loaded dataset via kagglehub (shape: {df.shape})")
            return df.to_dict(orient="split")
    except Exception as e:
        print("kagglehub download failed or not configured:", e)

    # fallback: placeholder vazio (ou gerar sintético se preferir)
    print("No initial dataset found in candidate locations. Using empty dataset placeholder.")
    return {"index": [], "columns": [], "data": []}

_INITIAL_MAIN_STORE = _load_initial_dataset()

# Layout: stores devem existir no root para serem compartilhados entre páginas
app.layout = dbc.Container([
    dcc.Store(id=STORE_MAIN, data=_INITIAL_MAIN_STORE, storage_type='memory'),
    dcc.Store(id=STORE_PROCESSED, storage_type='memory'),
    dcc.Store(id=STORE_PCA, storage_type='memory'),
    dcc.Store(id=STORE_SAMPLED_PCA, storage_type='memory'),
    dcc.Store(id=STORE_CLUSTER_LABELS, storage_type='memory'),
    dcc.Store(id=STORE_PREDICTION_MODEL, storage_type='memory'),
    dcc.Store(id=STORE_CLUSTER_MODEL, storage_type='memory'),  # <--- new store for cluster models
    dcc.Store(id=STORE_PCA_MODEL, storage_type='memory'),
    dcc.Store(id=STORE_SCALER, storage_type='memory'),
    dcc.Store(id=STORE_PCA_FEATURES, storage_type='memory'),

    dcc.Location(id='url', refresh=False),

    dbc.Row([
        dbc.Col(_build_sidebar(), width=2, style={"height": "100vh", "overflowY": "auto", "backgroundColor": "#222"}),
        dbc.Col(html.Div(id="page-content", children=[dash.page_container]), width=10, style={"height": "100vh", "overflowY": "auto"})
    ], className="g-0", style={"height": "100vh"}),
], fluid=True, style={"height": "100vh", "overflow": "hidden"})

if __name__ == "__main__":
    # production / stable dev run: desativa o reloader para evitar reinícios ao criar arquivos em tempo de execução
    app.run(debug=False, port=8050, use_reloader=False)