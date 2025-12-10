import dash
from dash import html, dcc, callback, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.constants import STORE_MAIN, STORE_PROCESSED, STORE_PCA, STORE_SAMPLED_PCA, STORE_CLUSTER_LABELS
from src.utils import to_df
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
import warnings
import multiprocessing
import tempfile
import json
import os
import traceback

# new imports for SHAP fallback and classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

dash.register_page(__name__, path='/avaliacao', name='Avaliação dos Clusters', order=7)

layout = dbc.Container([
    html.H3("🧾 Avaliação e Comparação de Algoritmos de Clusterização"),
    dcc.Markdown(
        "Comparação entre vários algoritmos de clusterização. "
        "Plota os 4 melhores algoritmos (por Silhouette) com visualização dos clusters."
    ),
    dbc.Row([
        dbc.Col([
            dbc.Button("Executar Comparação", id='run-compare-button', color="primary", n_clicks=0),
            html.Div(id='compare-status', className="mt-2")
        ], width=12)
    ], className="my-3"),

    dbc.Row([dbc.Col(dcc.Loading(dash_table.DataTable(id='compare-results-table',
                                                style_table={'overflowX': 'auto'},
                                                page_size=10)), width=12)], className="mb-2"),

    dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id='top4-plot')), width=12)], className="mb-3"),

    # NEW: SHAP option + outputs (sampled, fast)
    dbc.Row([
        dbc.Col([
            dbc.Checklist(options=[{"label": "Gerar SHAP (amostrado, rápido)", "value": "shap"}],
                          value=[], id='include-shap', inline=True),
        ], width=6)
    ], className="mb-2"),

    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id='shap-plot')), width=8),
        dbc.Col(html.Div(id='shap-info'), width=4)
    ], className="mb-4"),

], fluid=True)


@callback(
    Output('compare-status', 'children'),
    Output('compare-results-table', 'data'),
    Output('compare-results-table', 'columns'),
    Output('top4-plot', 'figure'),
    Output('shap-plot', 'figure'),
    Output('shap-info', 'children'),
    Input('run-compare-button', 'n_clicks'),
    State('include-shap', 'value'),
    State(STORE_PCA, 'data'),
    State(STORE_PROCESSED, 'data'),
    State(STORE_MAIN, 'data'),
    prevent_initial_call=True
)
def run_comparison(n_clicks, include_shap, pca_data, processed_data, main_data):
    warnings.filterwarnings("ignore")
    empty_fig = go.Figure()
    empty_fig.update_layout(title="Nenhum resultado", template="plotly_dark")
    empty_cols = []
    empty_data = []

    if not (pca_data or processed_data):
        return dbc.Alert("PCA ou dados processados necessários para executar a comparação.", color="warning"), empty_data, empty_cols, empty_fig, empty_fig, ""

    # Prefer PCA-transformed data
    try:
        dfX = pd.DataFrame(**pca_data) if pca_data else pd.DataFrame(**processed_data).select_dtypes(include=np.number)
    except Exception as e:
        return dbc.Alert(f"Erro ao processar dados: {str(e)}", color="danger"), empty_data, empty_cols, empty_fig, empty_fig, ""

    if dfX.empty or dfX.shape[0] < 10:
        return dbc.Alert("Dados insuficientes para comparação.", color="warning"), empty_data, empty_cols, empty_fig, empty_fig, ""

    X_full = dfX.values
    n_total = X_full.shape[0]

    # Fixed sampling limits (from pipeline2.0.ipynb)
    limits = {
        "KMeans": 5000,
        "GMM": 5000,
        "MeanShift": 5000,
        "DBSCAN": 5000,
        "Agglomerative": 3000,
        "Spectral": 1500
    }

    algorithms = [
        ("KMeans", lambda: KMeans(n_clusters=3, random_state=42, n_init=10)),
        ("MeanShift", lambda: MeanShift()),
        ("DBSCAN", lambda: DBSCAN(eps=0.5, min_samples=10)),
        ("Agglomerative", lambda: AgglomerativeClustering(n_clusters=3)),
        ("Spectral", lambda: SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)),
        ("GMM", lambda: GaussianMixture(n_components=3, random_state=42))
    ]

    results = []
    rng = np.random.RandomState(42)

    for name, ctor in algorithms:
        lim = limits.get(name, 5000)
        indices = rng.choice(n_total, min(lim, n_total), replace=False)
        X = X_full[indices]

        try:
            model = ctor()
            labels = model.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Calculate silhouette only if valid clusters
            if n_clusters > 1:
                mask = labels != -1 if -1 in labels else slice(None)
                if isinstance(mask, slice) or mask.sum() > 1:
                    sil = float(silhouette_score(X[mask] if isinstance(mask, np.ndarray) else X, 
                                                 labels[mask] if isinstance(mask, np.ndarray) else labels))
                else:
                    sil = np.nan
            else:
                sil = np.nan

            results.append({
                "Algoritmo": name,
                "Clusters": int(n_clusters) if n_clusters > 0 else 0,
                "Silhouette": round(sil, 4) if not np.isnan(sil) else "-"
            })
        except Exception:
            results.append({
                "Algoritmo": name,
                "Clusters": "-",
                "Silhouette": "-"
            })

    res_df = pd.DataFrame(results)
    
    # Sort by silhouette (handle mixed types)
    res_df['sil_sort'] = pd.to_numeric(res_df['Silhouette'], errors='coerce')
    res_df = res_df.sort_values(by='sil_sort', ascending=False, na_position='last').drop('sil_sort', axis=1).reset_index(drop=True)

    cols = [{"name": c, "id": c} for c in res_df.columns]
    data = res_df.to_dict("records")

    # Choose top 4 by silhouette
    top4_names = res_df[res_df['Silhouette'] != '-'].head(4)["Algoritmo"].tolist()
    
    if len(top4_names) == 0:
        return (dbc.Alert("Nenhum algoritmo produziu clusters válidos.", color="warning"), 
                data, cols, empty_fig, empty_fig, "")

    # Create subplots with scatter plots (PC_1 vs PC_2) with silhouette scores
    fig_top4 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"{algo} (Silhouette: {res_df[res_df['Algoritmo']==algo]['Silhouette'].values[0]})" 
                        for algo in top4_names],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )

    # Use Plotly's built-in color sequence for discrete colors
    color_palette = px.colors.qualitative.Plotly

    for idx, algo in enumerate(top4_names):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        lim = limits.get(algo, 5000)
        indices = rng.choice(n_total, min(lim, n_total), replace=False)
        X_plot = X_full[indices]

        try:
            model = [c for n, c in algorithms if n == algo][0]()
            labels = model.fit_predict(X_plot)
            
            # Extract PC_1 and PC_2 (first two columns)
            x_vals = X_plot[:, 0]
            y_vals = X_plot[:, 1] if X_plot.shape[1] >= 2 else np.zeros_like(x_vals)
            
            # Create scatter plot with color by cluster
            unique_labels = sorted(set(labels))
            for label_idx, label_val in enumerate(unique_labels):
                mask = labels == label_val
                color = color_palette[label_idx % len(color_palette)]
                
                fig_top4.add_trace(
                    go.Scatter(
                        x=x_vals[mask],
                        y=y_vals[mask],
                        mode='markers',
                        marker=dict(size=6, color=color, opacity=0.7),
                        name=f'Cluster {label_val}' if label_val != -1 else 'Ruído',
                        showlegend=(idx == 0),  # Legend only for first subplot
                        hovertemplate='<b>PC_1:</b> %{x:.3f}<br><b>PC_2:</b> %{y:.3f}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
        except Exception:
            pass

        # Update axes labels
        x_label = 'PC_1' if 'PC_1' in dfX.columns else 'Dim 1'
        y_label = 'PC_2' if 'PC_2' in dfX.columns else 'Dim 2'
        fig_top4.update_xaxes(title_text=x_label, row=row, col=col)
        fig_top4.update_yaxes(title_text=y_label, row=row, col=col)

    fig_top4.update_layout(
        height=900,
        title_text="<b>Top 4 Algoritmos de Clusterização (por Silhouette)</b><br><sub>Visualização com PC_1 vs PC_2</sub>",
        template="plotly_dark",
        hovermode='closest',
        showlegend=True
    )

    # --- NEW: SHAP (sampled, fast) ---
    shap_fig = empty_fig
    shap_info = ""
    if include_shap and 'shap' in include_shap:
        # need processed_data and main_data to align features and labels
        try:
            processed_df = pd.DataFrame(**processed_data) if processed_data else None
            main_df = pd.DataFrame(**main_data) if main_data else None

            if processed_df is None or main_df is None:
                raise ValueError("dados processados ou main ausentes para SHAP")

            # choose numeric features as default
            features_for_shap = processed_df.select_dtypes(include=np.number).columns.tolist()
            if not features_for_shap:
                raise ValueError("nenhuma feature numérica encontrada para SHAP")

            # prepare target: group rare genres (top 10) as in pipeline2.0
            if 'track_genre' not in main_df.columns:
                raise ValueError("coluna 'track_genre' não encontrada em main data")
            df_main_aligned = main_df.loc[processed_df.index]
            top_n = 10
            top_genres = df_main_aligned['track_genre'].value_counts().nlargest(top_n).index
            df_main_aligned['track_genre_grouped'] = df_main_aligned['track_genre'].where(df_main_aligned['track_genre'].isin(top_genres), 'outros')

            mask = df_main_aligned['track_genre_grouped'] != 'outros'
            X_shap_full = processed_df.loc[mask, features_for_shap].copy()
            y_shap_full = df_main_aligned.loc[mask, 'track_genre_grouped'].copy()

            if X_shap_full.shape[0] < 20 or len(y_shap_full.unique()) < 2:
                raise ValueError("dados insuficientes para SHAP sampling")

            # Label encode
            le_shap = LabelEncoder()
            y_shap_enc = le_shap.fit_transform(y_shap_full)
            y_shap_series = pd.Series(y_shap_enc, index=X_shap_full.index)

            # sampling for speed
            n_explain = min(500, len(X_shap_full))
            bg_n = min(50, len(X_shap_full))
            X_background = X_shap_full.sample(n=bg_n, random_state=42)
            X_explain = X_shap_full.sample(n=n_explain, random_state=42)
            y_explain = y_shap_series.loc[X_explain.index].values

            # train lightweight RF
            model_shap = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=-1)
            model_shap.fit(X_shap_full, y_shap_series.values)

            # try SHAP TreeExplainer, fallback to permutation importance
            try:
                import shap
                # Prefer new unified API quando disponível
                explainer = None
                try:
                    if hasattr(shap, "Explainer"):
                        # usa background pequeno (já definido) para economizar memória
                        explainer = shap.Explainer(model_shap, X_background, algorithm="tree")
                except Exception:
                    explainer = None

                # fallback para TreeExplainer se Explainer não funcionar
                if explainer is None:
                    explainer = shap.TreeExplainer(model_shap, data=X_background, feature_perturbation="interventional")

                # extrai os valores SHAP (pode retornar Explanation ou lista/array)
                if hasattr(explainer, "__call__"):
                    out = explainer(X_explain)
                    # Explanation: .values ou .values[..., j]; compatibiliza formatos
                    shap_values = out.values if hasattr(out, "values") else out
                else:
                    shap_values = explainer.shap_values(X_explain)

                # compute mean abs importance robustly (same helper as notebook)
                def mean_abs_shap(shv):
                    a = shv
                    if isinstance(a, list):
                        arrs = []
                        for sv in a:
                            sv = np.asarray(sv)
                            if sv.ndim == 3:
                                sv = np.mean(np.abs(sv), axis=2)
                            arrs.append(np.abs(sv))
                        stacked = np.stack(arrs, axis=0)
                        return stacked.mean(axis=(0,1))
                    else:
                        sv = np.asarray(a)
                        if sv.ndim == 3:
                            if sv.shape[2] == X_explain.shape[1]:
                                sv = np.mean(np.abs(sv), axis=2)
                            else:
                                sv = np.abs(sv)
                            if sv.ndim == 3:
                                return sv.mean(axis=(0,1))
                        if sv.ndim == 2:
                            return np.mean(np.abs(sv), axis=0)
                        return np.ravel(np.abs(sv))

                mean_abs = mean_abs_shap(shap_values)
                shap_importances = pd.Series(mean_abs, index=features_for_shap).sort_values(ascending=False)
                top10 = shap_importances.head(10).index.tolist()

                # save globally for classification pipeline (mimic notebook behavior)
                globals()['selected_features'] = top10

                # build a horizontal bar plot (plotly)
                top_plot = shap_importances.head(12)
                shap_fig = go.Figure(go.Bar(
                    x=top_plot.values[::-1],
                    y=top_plot.index[::-1],
                    orientation='h',
                    marker=dict(color='rgba(46,137,205,0.8)')
                ))
                shap_fig.update_layout(title="SHAP mean(|value|) - top features (sampled)", template="plotly_dark", height=450)
                shap_info = dbc.Card(dbc.CardBody([
                    html.P(f"Top 10 features salvas para classificação:"),
                    html.Ul([html.Li(f) for f in top10])
                ]))
            except Exception as e_shap:
                # fallback permutation importance (faster / safer)
                perm = permutation_importance(model_shap, X_shap_full, y_shap_series.values, n_repeats=8, random_state=42, n_jobs=-1, scoring='f1_macro')
                shap_importances = pd.Series(perm.importances_mean, index=features_for_shap).sort_values(ascending=False)
                top10 = shap_importances.head(10).index.tolist()
                globals()['selected_features'] = top10

                top_plot = shap_importances.head(12)
                shap_fig = go.Figure(go.Bar(
                    x=top_plot.values[::-1],
                    y=top_plot.index[::-1],
                    orientation='h',
                    marker=dict(color='rgba(204,121,167,0.8)')
                ))
                shap_fig.update_layout(title="Permutation importance (fallback) - top features", template="plotly_dark", height=450)
                shap_info = dbc.Card(dbc.CardBody([
                    html.P("SHAP TreeExplainer não disponível/erro — usado permutation importance como fallback."),
                    html.P(f"Top 10 features salvas para classificação:"),
                    html.Ul([html.Li(f) for f in top10])
                ]))

        except Exception as e_outer:
            shap_fig = empty_fig
            shap_info = dbc.Alert(f"SHAP não gerado: {type(e_outer).__name__} - {str(e_outer)}", color="warning")

    status = dbc.Alert(f"✓ Comparação finalizada ({len(algorithms)} algoritmos testados).", 
                       color="success", duration=4000)
    
    return status, data, cols, fig_top4, shap_fig, shap_info


def _shap_worker(processed_dict, main_dict, out_path):
    """Executado em processo separado: treina RF leve e tenta SHAP; grava JSON com importâncias/top10."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.inspection import permutation_importance

        # reconstruir dataframes
        processed_df = pd.DataFrame(**processed_dict)
        main_df = pd.DataFrame(**main_dict)

        # features numéricas
        features_for_shap = processed_df.select_dtypes(include=np.number).columns.tolist()
        if not features_for_shap:
            raise ValueError("nenhuma feature numérica para SHAP")

        # agrupa gêneros (top 10) igual ao notebook
        if 'track_genre' not in main_df.columns:
            raise ValueError("coluna 'track_genre' ausente em main")
        aligned = main_df.loc[processed_df.index]
        top_n = 10
        top_genres = aligned['track_genre'].value_counts().nlargest(top_n).index
        aligned['track_genre_grouped'] = aligned['track_genre'].where(aligned['track_genre'].isin(top_genres), 'outros')
        mask = aligned['track_genre_grouped'] != 'outros'
        X_shap_full = processed_df.loc[mask, features_for_shap].copy()
        y_shap_full = aligned.loc[mask, 'track_genre_grouped'].copy()

        if X_shap_full.shape[0] < 20 or len(y_shap_full.unique()) < 2:
            raise ValueError("dados insuficientes para SHAP/permutation")

        le = LabelEncoder()
        y_enc = le.fit_transform(y_shap_full)
        y_series = pd.Series(y_enc, index=X_shap_full.index)

        # treina RF leve com n_jobs=1 (evitar fork/multithread no worker)
        model_shap = RandomForestClassifier(n_estimators=40, max_depth=8, random_state=42, n_jobs=1)
        model_shap.fit(X_shap_full, y_series.values)

        # primeiro tenta SHAP.Explainer / TreeExplainer, mas não é crítico — se falhar, cai em permutation_importance
        importances = None
        try:
            import shap
            # usa background pequeno
            bg = X_shap_full.sample(n=min(30, len(X_shap_full)), random_state=42)
            # preferir shap.Explainer quando disponível
            expl = None
            if hasattr(shap, "Explainer"):
                try:
                    expl = shap.Explainer(model_shap, bg, algorithm="tree")
                except Exception:
                    expl = None
            if expl is None:
                expl = shap.TreeExplainer(model_shap, data=bg, feature_perturbation="interventional")
            X_explain = X_shap_full.sample(n=min(200, len(X_shap_full)), random_state=42)
            out = expl(X_explain)
            # compatibiliza formatos: Explanation.values ou list
            vals = out.values if hasattr(out, "values") else out
            # compute mean abs per feature robustly (copiado do notebook)
            if isinstance(vals, list):
                arrs = []
                for sv in vals:
                    sv = np.asarray(sv)
                    if sv.ndim == 3:
                        sv = np.mean(np.abs(sv), axis=2)
                    arrs.append(np.abs(sv))
                stacked = np.stack(arrs, axis=0)
                mean_abs = stacked.mean(axis=(0,1))
            else:
                sv = np.asarray(vals)
                if sv.ndim == 3:
                    if sv.shape[2] == X_explain.shape[1]:
                        sv = np.mean(np.abs(sv), axis=2)
                    else:
                        sv = np.abs(sv)
                if sv.ndim == 2:
                    mean_abs = np.mean(np.abs(sv), axis=0)
                else:
                    mean_abs = np.ravel(np.abs(sv))
            importances = mean_abs.tolist()
        except Exception:
            # fallback rápido: permutation importance (mais estável no server)
            perm = permutation_importance(model_shap, X_shap_full, y_series.values, n_repeats=6, random_state=42, n_jobs=1, scoring='f1_macro')
            importances = perm.importances_mean.tolist()

        # grava resultados
        feat_list = features_for_shap
        top_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:10]
        top_features = [feat_list[i] for i in top_idx]
        res = {"features": feat_list, "importances": importances, "top10": top_features}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
    except Exception as e:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"error": repr(e), "trace": traceback.format_exc()}, f)