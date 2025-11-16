import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from io import StringIO

dash.register_page(__name__, path='/predicao', name='8. Atribuição de Cluster')

# This list should match the core numerical features used for PCA/Clustering
features_for_input = [
    'popularity', 'duration_s', 'danceability', 'energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

def create_input_form():
    """Helper function to create the new song input form."""
    form_elements = []
    for feature in features_for_input:
        # Simple heuristic for sliders vs. number inputs
        if feature in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'mode', 'explicit']:
            step = 0.01
            max_val = 1.0
        else:
            step = 1
            max_val = 250 if feature == 'tempo' else 500000

        form_elements.append(
            dbc.Row([
                dbc.Col(html.Label(feature.replace('_', ' ').capitalize()), width=4),
                dbc.Col(dcc.Input(id=f'pred-input-{feature}', type='number', placeholder=f'Enter {feature}...', step=step, style={'width': '100%'} ), width=8)
            ], className="mb-2")
        )
    return form_elements

# --- Layout ---
layout = dbc.Container([
    html.H3("🔮 Atribuição de Cluster (Predição)"),
    dcc.Markdown("""
    Utilize os resultados da sua análise de cluster para **atribuir uma música a um dos clusters existentes**.
    """),
    html.Div(id='pred-warning-div'),

    dbc.Tabs([
        dbc.Tab(label="Atribuir Nova Música", children=[
            dbc.Card(dbc.CardBody([
                html.H5("Insira as Características da Nova Música"),
                html.P("Preencha os campos abaixo e clique em 'Prever Cluster'."),
                *create_input_form(),
                dbc.Button("Prever Cluster para Nova Música", id='predict-new-button', color="primary", n_clicks=0, className="mt-3 w-100"),
            ]), className="mt-3")
        ]),
    ]),
    
    dbc.Row([
        dbc.Col(
            dbc.Spinner(html.Div(id='prediction-result-div', className="mt-4"))
        )
    ])
])

# --- Callback ---
@callback(
    Output('prediction-result-div', 'children'),
    Output('pred-warning-div', 'children'),
    Input('predict-new-button', 'n_clicks'),
    [State(f'pred-input-{f}', 'value') for f in features_for_input],
    State('processed-df-store', 'data'),
    State('cluster-labels-store', 'data'),
    State('main-df-store', 'data')
)
def assign_to_cluster(n_clicks, *args):
    if n_clicks == 0:
        return "", ""

    # Unpack the arguments manually
    # The number of features is now 11
    feature_values = args[:11]
    # The last 3 args are the data stores
    processed_data = args[11]
    labels = args[12]
    main_df_data = args[13]

    if not all([processed_data, labels, main_df_data]):
        alert = dbc.Alert("Dados necessários (processados, clusters) não encontrados. Execute os passos anteriores.", color="warning")
        return "", alert

    # --- 0. Validate user inputs ---
    if any(v is None for v in feature_values):
        alert = dbc.Alert("Por favor, preencha todos os campos de características da música.", color="danger")
        return "", alert

    # --- 1. Recreate the new song DataFrame from user inputs ---
    # Create a dictionary mapping feature names to their input values
    input_data = {feature: value for feature, value in zip(features_for_input, feature_values)}
    new_song_df = pd.DataFrame([input_data])

    # --- 2. Align columns with the original processed data ---
    df_processed = pd.read_json(StringIO(processed_data), orient='split')
    
    # --- 3. Apply the same preprocessing steps ---
    # Re-fit the scaler ONLY on the numeric features from the processed data
    scaler = StandardScaler().fit(df_processed[features_for_input])
    
    # Transform the new song data (it already has the correct columns from the input form)
    new_song_scaled = scaler.transform(new_song_df)

    # --- 4. Predict the cluster by finding the closest centroid ---
    # Add original labels to the processed data to calculate centroids
    df_processed['cluster'] = labels
    
    # Calculate the mean of each feature for each cluster to find the centroids
    # IMPORTANT: Calculate centroids only on the numeric features
    centroids = df_processed.groupby('cluster')[features_for_input].mean()
    
    # Calculate the distance from the new song to each centroid
    # new_song_scaled is a 2D array with one row, so we take the first row [0]
    distances = np.linalg.norm(centroids.values - new_song_scaled[0], axis=1)
    
    # The predicted cluster is the index of the minimum distance
    predicted_cluster = centroids.index[np.argmin(distances)]

    result_card = dbc.Card(
        dbc.CardBody([
            html.H4("Resultado da Atribuição", className="card-title"),
            html.H1(f"Cluster {predicted_cluster}", className="text-center text-success display-3"),
            html.P("Esta nova música seria atribuída ao cluster acima com base em suas características.", className="card-text"),
        ]),
    )
    return result_card, ""