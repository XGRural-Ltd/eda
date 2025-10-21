import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import pandas as pd
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

dash.register_page(__name__, path='/predicao', name='8. Atribuição de Cluster')

# These are the features the original model expected. We'll use them for the input form.
# Source: selected_features.pkl
features_for_input = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
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
    Esta página utiliza os resultados da sua análise de cluster para **atribuir uma música a um dos clusters existentes**. 
    Isso substitui o antigo modelo de predição de gênero para se alinhar melhor com o fluxo de trabalho de EDA.
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
    State('processed-df-store', 'data'),
    State('pca-df-store', 'data'),
    State('cluster-labels-store', 'data'),
    [State(f'pred-input-{feature}', 'value') for feature in features_for_input],
    prevent_initial_call=True
)
def assign_to_cluster(n_clicks_new, processed_data, pca_data, labels, new_song_features):
    # Check if prerequisite data exists
    if not all([processed_data, pca_data, labels]):
        alert = dbc.Alert("⚠️ Por favor, execute as etapas de Pré-processamento, Redução e Clusterização primeiro.", color="warning")
        return "", alert

    # Check if all fields for the new song are filled
    if any(v is None for v in new_song_features):
        return dbc.Alert("Por favor, preencha todos os campos da nova música.", color="danger"), ""

    try:
        # --- Re-create and re-fit the models based on the stored data ---
        df_processed = pd.DataFrame(**processed_data)
        df_pca = pd.DataFrame(**pca_data)
        
        # 1. Re-fit the Scaler and PCA model on the processed data
        # This ensures we use the exact same transformations
        scaler = StandardScaler().fit(df_processed)
        n_components = df_pca.shape[1]
        pca = PCA(n_components=n_components).fit(df_processed)
        
        # 2. Re-fit the Clustering model
        k = len(set(labels)) - (1 if -1 in labels else 0)
        if k < 2:
            return dbc.Alert("Erro: O número de clusters encontrados é menor que 2.", color="danger"), ""
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_pca)
        
        # --- Process and predict the new song ---
        # Create a dataframe for the new song
        new_song_df = pd.DataFrame([new_song_features], columns=features_for_input)
        
        # The processed dataframe might have more columns than the input form
        # We need to ensure the new song dataframe has the same columns in the same order
        new_song_aligned = pd.DataFrame(columns=df_processed.columns)
        new_song_aligned = pd.concat([new_song_aligned, new_song_df], ignore_index=True).fillna(0)
        
        # Apply transformations IN ORDER
        new_song_scaled = scaler.transform(new_song_aligned)
        new_song_pca = pca.transform(new_song_scaled)
        predicted_cluster = kmeans.predict(new_song_pca)
        
        result_card = dbc.Card(
            dbc.CardBody([
                html.H4("Resultado da Atribuição", className="card-title"),
                html.H1(f"Cluster {predicted_cluster[0]}", className="text-center text-success display-3"),
                html.P("Esta nova música seria atribuída ao cluster acima com base em suas características.", className="card-text"),
            ]),
        )
        return result_card, ""

    except Exception as e:
        return dbc.Alert(f"Ocorreu um erro durante a predição: {e}", color="danger"), ""