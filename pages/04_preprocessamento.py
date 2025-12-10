import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.constants import STORE_MAIN, STORE_PROCESSED
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
            html.P("Aplica: remoção de duplicatas, tratamento de missing values, detecção e remoção de outliers, escalonamento robusto e normalização Min-Max."),
            dbc.Button("Executar Pré-processamento", id="preprocess-run-btn", color="primary", className="me-2"),
            html.Div(id="preprocess-status", className="mt-3")
        ])
    ], className="mb-4"),
    
    dbc.Card([
        dbc.CardBody([
            html.H5("O que é feito nesta etapa?", className="card-title"),
            html.Ul([
                html.Li("Remoção de duplicatas: Removes identical rows to avoid bias in models."),
                html.Li("Tratamento de valores ausentes: For numeric columns, replaces missing values with the median (robust to outliers). For categorical columns, uses the mode."),
                html.Li("Detecção de outliers: Uses IsolationForest on selected numeric features to identify anomalies, followed by 2D PCA visualization."),
                html.Li("Remoção de outliers: Removes points identified as anomalies to improve data quality."),
                html.Li("Escalonamento robusto (RobustScaler): Standardizes data using median and IQR, being less sensitive to outliers than StandardScaler."),
                html.Li("Normalização Min-Max: Scales values to the [0, 1] range, useful for algorithms that assume data in this range."),
                html.Li("Este pipeline garante consistência e robustez no pré-processamento, seguindo boas práticas de machine learning.")
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
        # Etapa 1: Remoção de duplicatas
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        # Identificar colunas numéricas e categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Etapa 2: Tratamento de valores ausentes
        # Numéricas: mediana (robusta)
        # Categóricas: moda
        for col in numeric_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isna().any():
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        # Etapa 3: Detecção de outliers com IsolationForest (igual ao pipeline2.0.ipynb)
        features_for_outliers = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'instrumentalness']
        feats = [c for c in features_for_outliers if c in df.columns]
        if not feats:
            raise ValueError("Nenhuma das features para detecção de outliers está presente no dataframe.")
        
        # Seleciona apenas numéricas e imputa mediana para evitar NaNs
        X = df[feats].select_dtypes(include=np.number).copy()
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
        # Escalonamento robusto
        scaler_out = RobustScaler()
        X_scaled = scaler_out.fit_transform(X_imputed.astype(float))
        
        # IsolationForest
        contamination = 0.05
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X_scaled)  # 1 = inlier, -1 = outlier
        
        # Mapear para dataframe
        df.loc[X_imputed.index, 'anomaly'] = preds
        
        outliers_count = (preds == -1).sum()
        
        # Etapa 4: Plot PCA 2D dos outliers (igual ao notebook)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=['PC_1', 'PC_2'], index=X_imputed.index)
        df_pca['anomaly'] = preds
        
        fig = px.scatter(
            df_pca, x='PC_1', y='PC_2', color='anomaly',
            color_discrete_map={1: 'blue', -1: 'red'},
            title=f'Outliers detectados via PCA (IsolationForest) - {outliers_count} outliers',
            labels={'anomaly': 'Tipo'}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        
        outliers_plot = dbc.Card([
            dbc.CardBody([
                html.H5("Visualização de Outliers", className="card-title"),
                dcc.Graph(figure=fig)
            ])
        ], className="mt-3")
        
        # Etapa 5: Remoção de outliers
        df_no_outliers = df.loc[X_imputed.index][df_pca['anomaly'] != -1].copy()
        
        # Etapa 6: Aplicar pipeline de pré-processamento (igual ao pipeline2.0.ipynb)
        preproc_pipeline = make_pipeline(
            SimpleImputer(strategy='median'),  # Imputação mediana
            RobustScaler(),                    # Escalonamento robusto
            MinMaxScaler()                     # Normalização Min-Max
        )
        
        # Aplicar apenas em colunas numéricas
        df_no_outliers[numeric_cols] = preproc_pipeline.fit_transform(df_no_outliers[numeric_cols])
        
        # Salvar pipeline para uso futuro (se necessário)
        # scaler_pipeline = preproc_pipeline  # Pode ser serializado se precisar
        
        # Serializar para Store
        processed_store = df_no_outliers.to_dict(orient='split')
        
        # Etapa 7: Criar resumo detalhado
        summary = dbc.Card([
            dbc.CardBody([
                html.H5("✓ Pré-processamento concluído", className="text-success"),
                html.Hr(),
                html.P(f"Linhas processadas: {len(df_no_outliers):,}"),
                html.P(f"Duplicatas removidas: {duplicates_removed:,}"),
                html.P(f"Outliers detectados e removidos: {outliers_count:,} ({contamination*100:.1f}% contaminação)"),
                html.P(f"Colunas numéricas processadas: {len(numeric_cols)} (imputação mediana, RobustScaler, MinMaxScaler)"),
                html.P(f"Colunas categóricas processadas: {len(categorical_cols)} (imputação moda)"),
                html.P(f"Valores ausentes tratados: {df.isna().sum().sum()}"),
                html.Hr(),
                html.H6("Detalhes do pipeline:"),
                html.Ul([
                    html.Li("SimpleImputer(strategy='median'): Substitui valores ausentes por mediana."),
                    html.Li("RobustScaler: Centraliza com mediana e escala com IQR (menos sensível a outliers)."),
                    html.Li("MinMaxScaler: Escala para [0, 1]."),
                    html.Li("IsolationForest: Detecta outliers com contaminação estimada.")
                ])
            ])
        ], className="mt-3")
        
        status = dbc.Alert("Pré-processamento executado com sucesso!", color="success")
        
        print(f"[PREPROCESS] Successfully processed {len(df_no_outliers)} rows, {len(df_no_outliers.columns)} columns, removed {outliers_count} outliers")
        return processed_store, status, outliers_plot, summary
        
    except Exception as e:
        alert = dbc.Alert(f"Erro no pré-processamento: {type(e).__name__}: {e}", color="danger")
        print(f"[PREPROCESS] Error: {e}")
        import traceback
        traceback.print_exc()
        return no_update, alert, no_update, no_update