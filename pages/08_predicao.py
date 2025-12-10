import dash
from dash import html, dcc, callback, Input, Output, State, no_update, ALL
from src.constants import STORE_MAIN, STORE_PREDICTION_MODEL
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/predicao', name='Atribuição de Cluster', order=8)

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
                html.P("Preencha os campos abaixo. As features são baseadas na sua seleção para o PCA."),
                # This container will be filled dynamically
                dbc.Spinner(html.Div(id='prediction-form-container')),
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

# --- Callbacks ---

# Callback 1: Dynamically generate the input form
@callback(
    Output('prediction-form-container', 'children'),
    Input(STORE_MAIN, 'data')
)
def generate_prediction_form(features):
    if not features:
        return dbc.Alert("Execute o PCA na página 'Redução de Dimensionalidade' primeiro para definir as features.", color="info")
    
    form_elements = []
    for feature in features:
        form_elements.append(
            dbc.Row([
                dbc.Col(html.Label(feature.replace('_', ' ').capitalize()), width=4),
                dbc.Col(dcc.Input(
                    # Use pattern-matching IDs
                    id={'type': 'pred-input', 'feature': feature},
                    type='number',
                    placeholder=f'Enter {feature}...',
                    step=0.01 if feature != 'tempo' and feature != 'popularity' else 1,
                    style={'width': '100%'}
                ), width=8)
            ], className="mb-2")
        )
    return form_elements

# Callback 2: Perform the prediction
@callback(
    Output('prediction-result-div', 'children'),
    Output('pred-warning-div', 'children'),
    Input('predict-new-button', 'n_clicks'),
    # Use pattern-matching to get values and IDs from the dynamic form
    State({'type': 'pred-input', 'feature': ALL}, 'value'),
    State({'type': 'pred-input', 'feature': ALL}, 'id'),
    State(STORE_MAIN, 'data'),
    prevent_initial_call=True
)
def assign_to_cluster(n_clicks, feature_values, feature_ids, b64_models):
    if not b64_models:
        alert = dbc.Alert("Modelos necessários (Scaler, PCA, Cluster) não encontrados. Execute os passos anteriores.", color="warning")
        return "", alert

    if any(v is None for v in feature_values):
        alert = dbc.Alert("Por favor, preencha todos os campos de características da música.", color="danger")
        return "", alert

    # --- 1. Recreate the new song DataFrame from user inputs ---
    feature_names = [item['feature'] for item in feature_ids]
    input_data = {name: value for name, value in zip(feature_names, feature_values)}
    new_song_df = pd.DataFrame([input_data])

    # --- 2. Deserialize all models ---
    def deserialize_model(b64_model):
        decoded_model = base64.b64decode(b64_model)
        mem_buffer = BytesIO(decoded_model)
        return joblib.load(mem_buffer)

    models = [deserialize_model(b64_model) for b64_model in b64_models]

    # --- 3. Apply the FULL transformation pipeline ---
    new_song_scaled = models[0].transform(new_song_df)
    new_song_pca = models[1].transform(new_song_scaled)
    
    # --- 4. Predict ---
    predicted_cluster_array = models[2].predict(new_song_pca)
    predicted_cluster = predicted_cluster_array[0]

    result_card = dbc.Card(
        dbc.CardBody([
            html.H4("Resultado da Atribuição", className="card-title"),
            html.H1(f"Cluster {predicted_cluster}", className="text-center text-success display-3"),
            html.P("Esta nova música seria atribuída ao cluster acima com base em suas características.", className="card-text"),
        ]),
    )
    return result_card, ""