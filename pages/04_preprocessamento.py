import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.constants import STORE_MAIN, STORE_PROCESSED, STORE_PCA, STORE_SAMPLED_PCA
from src.utils import to_df
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

dash.register_page(__name__, path='/preprocessamento', name='Pré-processamento', order=4)

layout = dbc.Container([
    html.H2("Pré-processamento dos Dados", className="mb-4"),
    
    dbc.Card([
        dbc.CardBody([
            html.H5("Executar Pré-processamento", className="card-title"),
            html.P("Aplica: remoção de duplicatas, detecção e remoção de outliers, e pipeline numérico (imputação mediana → RobustScaler → MinMaxScaler)."),
            dbc.Button("Executar Pré-processamento", id="preprocess-run-btn", color="primary", className="me-2"),
            html.Div(id="preprocess-status", className="mt-3")
        ])
    ], className="mb-4"),
    
    dbc.Card([ 
        dbc.CardBody([ 
            html.H5("O que é feito nesta etapa?", className="card-title"),
            html.Ul([
                html.Li("Seleciona colunas numéricas essenciais e a coluna 'track_genre'."),
                html.Li("Remoção de duplicatas."),
                html.Li("Imputação por mediana apenas nas features numéricas necessárias (evita NaNs antes do IsolationForest)."),
                html.Li("Detecção de outliers com IsolationForest sobre as features selecionadas."),
                html.Li("Visualização (PCA 2D) dos outliers usando as mesmas features imputadas."),
                html.Li("Remoção de outliers antes do pré-processamento final."),
                html.Li("Pré-processamento final: RobustScaler → MinMaxScaler aplicado somente às colunas numéricas selecionadas.")
            ])
        ])
    ], className="mb-4"),
    
    html.Div(id="outliers-plot-div"),
    html.Div(id="preprocess-summary-div")
], fluid=True)


@callback(
    Output(STORE_PROCESSED, 'data'),
    Output('preprocess-status', 'children'),
    Output('outliers-plot-div', 'children'),
    Output('preprocess-summary-div', 'children'),
    Input('preprocess-run-btn', 'n_clicks'),
    State(STORE_MAIN, 'data'),
    prevent_initial_call=True
)
def run_preprocessing(n_clicks, main_store):
    if not n_clicks:
        return no_update, no_update, no_update, no_update

    df = to_df(main_store)
    if df is None or df.empty:
        alert = dbc.Alert("Nenhum dado disponível para pré-processar", color="danger")
        return no_update, alert, no_update, no_update

    try:
        # --- alinhar com pipeline2.0: lista de num_features esperada ---
        expected_num_features = [
            'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'time_signature'
        ]
        # garantir track_genre
        if 'track_genre' not in df.columns:
            df['track_genre'] = 'unknown'
        else:
            df['track_genre'] = df['track_genre'].fillna('unknown')

        # manter somente as colunas relevantes (interseção)
        num_features = [f for f in expected_num_features if f in df.columns]
        if not num_features:
            raise ValueError("Nenhuma das num_features esperadas está presente no dataset.")
        df = df[num_features + ['track_genre']].copy()

        # Etapa 1: remover duplicatas
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)

        # Etapa 2: imputação mediana apenas nas numéricas antes do IsolationForest (mesma lógica do notebook)
        imputer = SimpleImputer(strategy='median')
        X = df[num_features].select_dtypes(include=np.number).copy()
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

        # Etapa 3: Detecção de outliers com IsolationForest (sem scaling, conforme pipeline2.0)
        contamination = 0.05
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X_imputed)  # 1 = inlier, -1 = outlier
        df.loc[X_imputed.index, 'anomaly'] = preds
        outliers_count = int((preds == -1).sum())

        # Etapa 4: PCA 2D para visualização (usar X_imputed para consistência com notebook)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_imputed)
        df_pca = pd.DataFrame(X_pca, columns=['PC_1', 'PC_2'], index=X_imputed.index)
        df_pca['anomaly'] = preds

        # plot com contraste no tema escuro
        fig = px.scatter(
            df_pca, x='PC_1', y='PC_2', color='anomaly',
            color_discrete_map={1: '#00d1ff', -1: '#ffcc00'},
            title=f'Outliers detectados via PCA (IsolationForest) - {outliers_count} outliers',
            labels={'anomaly': 'Tipo'},
            template='plotly_dark'
        )
        fig.update_traces(marker=dict(size=7, opacity=0.95, line=dict(width=0.6, color='rgba(255,255,255,0.6)')))
        fig.update_layout(legend_title_text='Anomaly (1=inlier, -1=outlier)')

        outliers_plot = dbc.Card([
            dbc.CardBody([
                html.H5("Visualização de Outliers", className="card-title"),
                dcc.Graph(figure=fig)
            ])
        ], className="mt-3")

        # Etapa 5: remover outliers e preparar df_no_outliers
        df_no_outliers = df.loc[X_imputed.index][df_pca['anomaly'] != -1].copy()

        # Etapa 6: Aplicar pipeline de pré-processamento (SimpleImputer(median)->RobustScaler->MinMaxScaler)
        preproc_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),
            RobustScaler(),
            MinMaxScaler()
        )

        df_no_outliers[num_features] = preproc_pipeline.fit_transform(df_no_outliers[num_features])

        # NOTE: agrupamento de gêneros removido daqui — será feito apenas na etapa de classificação

        # Serializar para Store (orient='split' compatível com to_df)
        processed_store = df_no_outliers.to_dict(orient='split')

        # Resumo informativo (alinhado com pipeline2.0, sem agrupamento de gêneros nesta etapa)
        summary = dbc.Card([
            dbc.CardBody([
                html.H5("✓ Pré-processamento concluído", className="text-success"),
                html.Hr(),
                html.P(f"Linhas processadas (após remoção de outliers): {len(df_no_outliers):,}"),
                html.P(f"Duplicatas removidas: {duplicates_removed:,}"),
                html.P(f"Outliers detectados e removidos: {outliers_count:,} ({contamination*100:.1f}% contaminação)"),
                html.P(f"Colunas numéricas processadas: {len(num_features)} (imputação mediana, RobustScaler, MinMaxScaler)"),
                html.Hr(),
                html.H6("Detalhes do pipeline:"),
                html.Ul([
                    html.Li("SimpleImputer(strategy='median'): Substitui valores ausentes por mediana (aplicado às numéricas)."),
                    html.Li("IsolationForest: detecta outliers com contaminação estimada; usado antes do escalonamento final."),
                    html.Li("RobustScaler: centraliza pela mediana e escala pelo IQR."),
                    html.Li("MinMaxScaler: normaliza para [0,1]."),
                    html.Li("Observação: agrupamento/filtragem de gêneros será aplicado apenas na etapa de classificação, para manter os dados brutos nesta fase.")
                ])
            ])
        ], className="mt-3")

        status = dbc.Alert("Pré-processamento executado com sucesso!", color="success")

        print(f"[PREPROCESS] processed {len(df_no_outliers)} rows, removed {outliers_count} outliers, duplicates removed {duplicates_removed}")
        return processed_store, status, outliers_plot, summary

    except Exception as e:
        alert = dbc.Alert(f"Erro no pré-processamento: {type(e).__name__}: {e}", color="danger")
        print(f"[PREPROCESS] Error: {e}")
        import traceback
        traceback.print_exc()
        return no_update, alert, no_update, no_update