import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from io import StringIO

dash.register_page(__name__, path='/preprocessamento', name='4. Pré-processamento')

# --- Layout ---
layout = dbc.Container([
    html.H3("⚙️ Pré-processamento dos Dados"),
    dcc.Markdown("""
    Aqui, transformaremos as features para que os algoritmos de Machine Learning possam interpretá-las da melhor forma. Usaremos o **StandardScaler** para padronizar as features numéricas, dando a elas média 0 e desvio padrão 1.
    """),
    
    dbc.Row([
        dbc.Col([
            html.Label("Selecione as features numéricas para padronizar:"),
            dcc.Dropdown(id='preprocess-features-dropdown', multi=True)
        ], width=12)
    ], className="my-4"),
    
    dbc.Row([
        dbc.Col([
            html.H5("Antes da Padronização (Danceability)"),
            dbc.Spinner(dcc.Graph(id='preprocess-before-graph'))
        ], width=6),
        dbc.Col([
            html.H5("Depois da Padronização (Danceability)"),
            dbc.Spinner(dcc.Graph(id='preprocess-after-graph'))
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            dbc.Button("Salvar DataFrame Processado para Próximas Etapas", id='save-processed-df-button', color="primary", n_clicks=0),
            dbc.Alert(id='save-status-alert', is_open=False, duration=4000, color="success")
        ], width=12, className="mt-4 text-center")
    ]),

    # Hidden div to store the processed dataframe as JSON before saving to dcc.Store
    html.Div(id='processed-df-json-storage', style={'display': 'none'})
])

# --- Callbacks ---

# Callback to populate the feature dropdown
@callback(
    Output('preprocess-features-dropdown', 'options'),
    Output('preprocess-features-dropdown', 'value'),
    Input('main-df-store', 'data')
)
def populate_preprocess_dropdown(json_data):
    if json_data is None:
        return [], []
    df = pd.read_json(StringIO(json_data), orient='split')
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    return num_features, num_features # Options and default value

# Callback to perform preprocessing and update graphs
@callback(
    Output('preprocess-before-graph', 'figure'),
    Output('preprocess-after-graph', 'figure'),
    Output('processed-df-json-storage', 'children'), # Store result here
    Input('main-df-store', 'data'),
    Input('preprocess-features-dropdown', 'value')
)
def run_preprocessing(json_data, features_to_scale):
    if not json_data or not features_to_scale:
        return no_update, no_update, no_update

    df = pd.read_json(StringIO(json_data), orient='split').dropna()
    df_preprocessed = df.copy()

    # Create 'before' graph for a sample feature
    fig_before = px.histogram(df, x='danceability', title="Original 'danceability'")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df_preprocessed[features_to_scale] = scaler.fit_transform(df_preprocessed[features_to_scale])
    
    # Create 'after' graph for the same sample feature
    fig_after = px.histogram(df_preprocessed, x='danceability', title="Padronizado 'danceability'")
    
    # Select only numeric columns for the final processed dataframe
    final_df = df_preprocessed[df_preprocessed.select_dtypes(include=np.number).columns.tolist()]
    
    return fig_before, fig_after, final_df.to_json(orient='split')

# Callback to save the processed data to the global store on button click
@callback(
    Output('processed-df-store', 'data'),
    Output('save-status-alert', 'is_open'),
    Output('save-status-alert', 'children'),
    Input('save-processed-df-button', 'n_clicks'),
    State('processed-df-json-storage', 'children')
)
def save_processed_data(n_clicks, processed_json):
    if n_clicks > 0 and processed_json:
        # The data is already in JSON format, just pass it through
        data_to_store = json.loads(processed_json)
        return data_to_store, True, "DataFrame processado salvo com sucesso! ✅"
    return no_update, False, ""