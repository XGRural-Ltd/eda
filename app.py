import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Dash, Input, Output
import pandas as pd
import kagglehub
import os
from io import StringIO

# --- 1. LOAD DATA ONCE WHEN THE APP STARTS ---
def load_initial_data():
    """Loads the main dataframe from Kaggle ONCE."""
    try:
        print("Attempting to download dataset from Kaggle...")
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        path_dataset = os.path.join(path, 'dataset.csv')
        print(f"Dataset downloaded to: {path_dataset}")
        
        df = pd.read_csv(path_dataset)
        
        # Preprocessing steps
        df = df.loc[:, ~df.columns.duplicated()]
        if 'track_genre' not in df.columns: df['track_genre'] = 'unknown'
        if 'duration_ms' in df.columns: df['duration_s'] = df['duration_ms'] / 1000
        
        print("Dataset loaded and processed successfully.")
        return df.to_json(date_format='iso', orient='split')
        
    except Exception as e:
        print(f"CRITICAL ERROR loading data from Kaggle: {e}")
        return None

# Execute the function to load data into a global variable
main_df_json = load_initial_data()


# --- App Initialization ---
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)
server = app.server


# --- 2. Use a simple callback to put the pre-loaded data into the store ---
@app.callback(
    Output('main-df-store', 'data'),
    Input('url', 'pathname') # Still triggered on page load
)
def store_initial_data(_):
    """
    This callback's only job is to place the pre-loaded data 
    into the dcc.Store. It doesn't re-download anything.
    """
    return main_df_json

# --- App Layout ---
sidebar = dbc.Nav(
    [
        dbc.NavLink(
            html.Div(page["name"], className="ms-2"), href=page["path"], active="exact"
        )
        for page in dash.page_registry.values()
    ],
    vertical=True,
    pills=True,
    className="bg-light",
)

app.layout = dbc.Container([
    # Create hidden Store components to hold dataframes between pages
    dcc.Store(id='main-df-store', storage_type='memory'),
    dcc.Store(id='processed-df-store', storage_type='memory'),
    dcc.Store(id='pca-df-store', storage_type='memory'),
    dcc.Store(id='cluster-labels-store', storage_type='memory'),
    dcc.Location(id='url', refresh=False),

    dbc.Row([
        html.H1("Spotify Tracks EDA with Dash", className="my-4 text-center"),
        html.Hr(),
    ]),
    dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Navegação", className="mb-3"),
                    sidebar
                ], xs=4, sm=3, md=2, lg=2, xl=2, xxl=2),
            dbc.Col(
                [
                   dash.page_container
                ], xs=8, sm=9, md=10, lg=10, xl=10, xxl=10)
        ]
    )
], fluid=True)


if __name__ == '__main__':
    app.run(debug=True)