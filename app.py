import os
import pandas as pd
import kagglehub
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, Input, Output, State
import threading
from io import StringIO
import json

# --- 1. LOAD DATA ONCE WHEN THE APP STARTS ---
def load_initial_data():
    """Loads the main dataframe from Kaggle ONCE."""
    try:
        print("Attempting to download dataset from Kaggle...")
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        path_dataset = os.path.join(path, 'dataset.csv')
        print(f"Dataset downloaded to: {path_dataset}")

        # Ler e tratar índice salvo/Unnamed
        tmp = pd.read_csv(path_dataset)
        if tmp.columns[0].startswith('Unnamed'):
            df = pd.read_csv(path_dataset, index_col=0)
        else:
            df = tmp
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

        # duration_ms -> duration_s
        if 'duration_ms' in df.columns:
            df['duration_s'] = (df['duration_ms'] / 1000).astype(float).round(3)
            df = df.drop(columns=['duration_ms'])

        # Remover colunas duplicadas por segurança
        df = df.loc[:, ~df.columns.duplicated()]

        print("Dataset loaded and processed successfully.")
        return df.to_json(date_format='iso', orient='split')
    except Exception as e:
        print(f"CRITICAL ERROR loading data from Kaggle: {e}")
        return None

# Hold dataset JSON; populated once
main_df_json = None

# --- App Initialization ---
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server

# one-time init
_init_lock = threading.Lock()
_initialized = False

@server.before_request
def _ensure_init():
    global _initialized, main_df_json
    if not _initialized:
        with _init_lock:
            if not _initialized:
                main_df_json = load_initial_data()
                _initialized = True

# Sidebar (links para as páginas registradas)
sidebar = dbc.Nav(
    [
        dbc.NavLink(
            html.Div(page["name"], className="ms-2"),
            href=page["path"],
            active="exact"
        )
        for page in sorted(dash.page_registry.values(), key=lambda p: p.get("name", ""))
    ],
    vertical=True,
    pills=True,
    className="bg-light p-2"
)

# --- Layout (ensure session storage to persist across pages/tabs) ---
app.layout = dbc.Container([
    dcc.Store(id='main-df-store', storage_type='memory'),
    dcc.Store(id='processed-df-store', storage_type='memory'),
    dcc.Store(id='pca-df-store', storage_type='memory'),
    dcc.Store(id='cluster-labels-store', storage_type='memory'),
    dcc.Store(id='prediction-model-store'),
    dcc.Store(id='pca-model-store'),
    dcc.Store(id='scaler-store'),
    dcc.Store(id='pca-features-store'),
    dcc.Store(id='sampled-pca-df-store'),
    dcc.Location(id='url', refresh=False),

    dbc.Row([
        html.H1("Spotify Tracks EDA", className="my-4 text-center"),
        html.Hr(),
    ]),
    dbc.Row([
        dbc.Col([html.H4("Navegação"), sidebar], xs=4, sm=3, md=2),
        dbc.Col(dash.page_container, xs=8, sm=9, md=10),
    ]),
], fluid=True)

# Do not overwrite cached main df once set (allow duplicate with initial_duplicate)
@app.callback(
    Output('main-df-store', 'data', allow_duplicate=True),
    Input('url', 'pathname'),
    State('main-df-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def store_initial_data(_pathname, current):
    return current if current is not None else main_df_json

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)