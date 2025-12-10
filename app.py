import os
import pandas as pd
import kagglehub
import dash
import threading
from dash import dcc, html, Dash, Input, Output, State, ALL
import plotly.io as pio
import dash_bootstrap_components as dbc
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
)

pio.templates.default = "plotly_dark"

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)

# --- Sidebar automático (gera links a partir de dash.page_registry) ---
def _build_sidebar():
    links = []
    for p in dash.page_registry.values():
        # ignora páginas marcadas como `path=None` (não registradas)
        if not p.get("path"):
            continue
        links.append(
            dbc.NavLink(p.get("name", p["module"]), href=p["path"], active="exact", className="mb-1")
        )
    nav = dbc.Nav(links, vertical=True, pills=True, className="flex-column")
    card = dbc.Card(dbc.CardBody([html.H5("Navegação"), nav]), className="h-100")
    return card

# --- TRY TO LOAD DATA ON STARTUP (serialized for dcc.Store initial value) ---
def _load_initial_dataset():
    """Return DataFrame serialized as orient='split' or None. Tries local CSVs then kagglehub."""
    candidates = [
        os.path.join(os.getcwd(), "data", "dataset.csv"),
        os.path.join(os.getcwd(), "dataset.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, index_col=0)
                print(f"Loaded dataset from {p} (shape: {df.shape})")
                return df.to_dict(orient="split")
            except Exception as e:
                print(f"Failed to read {p}: {e}")
    # fallback: kagglehub
    try:
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        csv_path = os.path.join(path, "dataset.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            print(f"Loaded dataset via kagglehub (shape: {df.shape})")
            return df.to_dict(orient="split")
    except Exception as e:
        print("kagglehub load failed:", e)
    print("No initial dataset available; main store will start empty.")
    return None

# call once at startup
_INITIAL_MAIN_STORE = _load_initial_dataset()

# --- CRITICAL: ALL stores MUST be in app.layout root (shared across all pages) ---
app.layout = dbc.Container([
    # === GLOBAL STORES (shared by all pages) - MUST be before page_container ---
    dcc.Store(id="main-df-store", data=_INITIAL_MAIN_STORE, storage_type='memory'),
    dcc.Store(id="processed-df-store", storage_type='memory'),
    dcc.Store(id="pca-df-store", storage_type='memory'),
    dcc.Store(id="sampled-pca-df-store", storage_type='memory'),
    dcc.Store(id="cluster-labels-store", storage_type='memory'),
    dcc.Store(id="prediction-model-store", storage_type='memory'),
    dcc.Store(id="pca-model-store", storage_type='memory'),
    dcc.Store(id="scaler-store", storage_type='memory'),
    dcc.Store(id="pca-features-store", storage_type='memory'),

    dcc.Location(id='url', refresh=False),
    
    dbc.Row([
        dbc.Col(_build_sidebar(), width=2, style={"height": "100vh", "overflow-y": "auto"}),
        dbc.Col(dash.page_container, width=10, style={"height": "100vh", "overflow-y": "auto"})
    ], className="g-0", style={"height": "100vh"}),
], fluid=True, style={"height": "100vh", "overflow": "hidden"})

# --- REMOVE THIS (causes Stores to be hidden) ---
# def print_callbacks_for_ids(target_ids):
#     ...

if __name__ == "__main__":
    # Production/dev on Windows: disable reloader and debug threads to avoid WinError 10038
    # Use threaded=False and use_reloader=False to prevent werkzeug reloader conflicts
    app.run(debug=False, port=8050, use_reloader=False, threaded=False)