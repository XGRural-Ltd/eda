import base64
import io
import os
import time
import joblib
import tempfile

import dash
from dash import html, dcc, callback, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from src.constants import STORE_MAIN, STORE_PROCESSED, STORE_PREDICTION_MODEL
from src.utils import to_df

# register page early (keeps app stable even if heavy libs missing at import time)
dash.register_page(__name__, path='/predicao', name='Atribuição de Cluster', order=8)

layout = dbc.Container([
    html.H3("🔮 Construir Playlist via Classificação"),
    dcc.Markdown(
        "Treine um classificador (RandomForest) usando a lógica do pipeline (agrupamento de gêneros por keyword + top-N) "
        "e gere uma playlist com as faixas mais prováveis para um gênero alvo."
    ),

    dbc.Row([
        dbc.Col(dbc.Button("Treinar / Atualizar Modelo", id="train-model-btn", color="primary", n_clicks=0), width=4),
        dbc.Col(html.Div(id="train-status-div"), width=8)
    ], className="my-2"),

    dbc.Row([
        dbc.Col([
            html.Label("Gênero alvo (após agrupamento)"),
            dcc.Dropdown(id='pred-genres-dropdown', options=[], value=None, clearable=False)
        ], width=6),
        dbc.Col([
            html.Label("Tamanho da playlist"),
            dcc.Slider(id='playlist-length', min=1, max=50, step=1, value=10,
                       marks={1: '1', 10: '10', 25: '25', 50: '50'})
        ], width=6)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Button("Gerar Playlist", id="generate-playlist-btn", color="success", n_clicks=0), width=4),
        dbc.Col(html.Div(id="generate-status-div"), width=8)
    ], className="mb-2"),

    html.Hr(),
    html.Div(id='playlist-output-div'),

    # store for model (base64-serialized joblib)
    dcc.Store(id=STORE_PREDICTION_MODEL)
], fluid=True)


# lightweight helper (pure python) — safe at import-time
def _map_and_group_genres(series_genre, top_n=30):
    import re
    keyword_map = {
        'pop': 'Pop', 'indie': 'Indie/Alt', 'rock': 'Rock', 'metal': 'Metal/Hard',
        'hip': 'Hip-Hop/Rap', 'r-n-b': 'R&B/Soul', 'soul': 'R&B/Soul',
        'electro': 'Electronic', 'edm': 'Electronic', 'house': 'Electronic/Dance',
        'techno': 'Electronic/Dance', 'trance': 'Electronic/Dance', 'dance': 'Electronic/Dance',
        'dub': 'Electronic', 'drum': 'Electronic', 'jazz': 'Jazz/Classical',
        'classical': 'Jazz/Classical', 'acoustic': 'Folk/Acoustic', 'folk': 'Folk/Acoustic',
        'country': 'Country', 'latin': 'Latin', 'samba': 'Latin', 'reggae': 'Reggae/Dancehall',
        'reggaeton': 'Reggaeton/Latin', 'blues': 'Blues', 'punk': 'Punk', 'emo': 'Rock',
        'ambient': 'Ambient/Chill', 'chill': 'Ambient/Chill', 'kids': 'Kids/Family',
        'soundtrack': 'Soundtrack', 'opera': 'Classical/Opera', 'world': 'World', 'brazil': 'Latin'
    }

    def _map_genre_by_keyword(g):
        g_low = (g or '').lower()
        for kw, cat in keyword_map.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', g_low):
                return cat
        return None

    series = series_genre.fillna('').astype(str)
    genres = sorted(series[series != ''].unique())
    manual_map = {g: (_map_genre_by_keyword(g) or 'Other') for g in genres}
    freq = series.value_counts()
    top_genres = set(freq.nlargest(top_n).index)
    condensed_map = {g: (manual_map.get(g, g) if g in top_genres else 'Other') for g in genres}
    return series_genre.map(condensed_map)


# Unified callback: treinar modelo ou gerar playlist dependendo do botão acionado.
@callback(
    Output(STORE_PREDICTION_MODEL, 'data'),
    Output('train-status-div', 'children'),
    Output('pred-genres-dropdown', 'options'),
    Output('playlist-output-div', 'children'),
    Output('generate-status-div', 'children'),
    Input('train-model-btn', 'n_clicks'),
    Input('generate-playlist-btn', 'n_clicks'),
    State(STORE_PREDICTION_MODEL, 'data'),
    State(STORE_PROCESSED, 'data'),
    State(STORE_MAIN, 'data'),
    State('pred-genres-dropdown', 'value'),
    State('playlist-length', 'value'),
    prevent_initial_call=True
)
def handle_train_or_generate(n_train, n_generate, model_path, processed_store, main_store, target_genre, playlist_length):
    try:
        triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    except Exception:
        triggered = None

    default_store = no_update
    default_train_status = no_update
    default_options = no_update
    default_playlist = no_update
    default_generate_status = no_update

    # salvar modelos fora da árvore do projeto para não acionar o reloader
    models_dir = os.path.join(tempfile.gettempdir(), "eda_models")
    os.makedirs(models_dir, exist_ok=True)

    # TRAINING branch
    if triggered == "train-model-btn":
        try:
            # lazy imports
            import base64, io
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            from sklearn.impute import SimpleImputer

            if not processed_store:
                return default_store, dbc.Alert("Execute o pré-processamento primeiro (página Pré-processamento).", color="warning"), [], default_playlist, default_generate_status

            processed_df = pd.DataFrame(**processed_store) if isinstance(processed_store, dict) else to_df(processed_store)
            main_df = to_df(main_store) if main_store else None

            if 'track_genre' not in processed_df.columns and main_df is not None:
                try:
                    processed_df['track_genre'] = main_df.loc[processed_df.index, 'track_genre']
                except Exception:
                    processed_df['track_genre'] = main_df['track_genre'].values[:len(processed_df)]

            processed_df['genre_grouped_manual'] = _map_and_group_genres(processed_df.get('track_genre', pd.Series(['unknown'] * len(processed_df))), top_n=30)

            numerical_features = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numerical_features:
                return default_store, dbc.Alert("Nenhuma coluna numérica encontrada em dados processados.", color="danger"), [], default_playlist, default_generate_status

            before_n = len(processed_df)
            proc = processed_df.dropna(subset=numerical_features + ['genre_grouped_manual']).copy()
            after_n = len(proc)
            if after_n == 0:
                return default_store, dbc.Alert("Após filtragem não restaram linhas para treinar.", color="danger"), [], default_playlist, default_generate_status

            X = proc[numerical_features].copy()
            y = proc['genre_grouped_manual'].copy()

            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            clf.fit(X_imputed, y_enc)

            # salvar modelo em disco (joblib) e guardar caminho no Store
            ts = int(time.time())
            model_filename = f"pred_model_{ts}.joblib"
            model_filepath = os.path.join(models_dir, model_filename)
            joblib.dump({'model': clf, 'le': le, 'features': numerical_features}, model_filepath, compress=3)

            # remover modelo anterior salvo (se existir) para não acumular
            try:
                if model_path and isinstance(model_path, str) and os.path.exists(model_path) and model_path.startswith(models_dir):
                    if model_path != model_filepath:
                        os.remove(model_path)
            except Exception:
                pass

            options = [{'label': g, 'value': g} for g in le.classes_.tolist()]
            status = dbc.Alert(f"Modelo treinado com sucesso. Treinou com {after_n} amostras (removidas {before_n-after_n}). Modelo salvo em {model_filepath}", color="success")
            return model_filepath, status, options, default_playlist, default_generate_status

        except Exception as e:
            import traceback
            traceback.print_exc()
            return default_store, dbc.Alert(f"Erro ao treinar modelo: {type(e).__name__}: {e}", color="danger"), [], default_playlist, default_generate_status

    # GENERATE branch
    if triggered == "generate-playlist-btn":
        try:
            import heapq
            from sklearn.impute import SimpleImputer

            if not model_path or not isinstance(model_path, str) or not os.path.exists(model_path):
                return default_store, default_train_status, default_options, default_playlist, dbc.Alert("Modelo não encontrado. Treine o modelo primeiro.", color="warning")

            obj = joblib.load(model_path)
            clf = obj['model']
            le = obj['le']
            features = obj['features']

            if target_genre is None:
                return default_store, default_train_status, default_options, default_playlist, dbc.Alert("Selecione um gênero alvo.", color="warning")

            if target_genre not in le.classes_:
                return default_store, default_train_status, default_options, default_playlist, dbc.Alert("Gênero selecionado não está no modelo treinado. Re-treine o modelo.", color="danger")

            processed_df = pd.DataFrame(**processed_store) if isinstance(processed_store, dict) else to_df(processed_store)
            main_df = to_df(main_store) if main_store else None

            missing = [f for f in features if f not in processed_df.columns]
            if missing:
                return default_store, default_train_status, default_options, default_playlist, dbc.Alert(f"Colunas de feature ausentes nos dados processados: {missing}", color="danger")

            medians = processed_df[features].median()

            n_rows = len(processed_df)
            if n_rows == 0:
                return default_store, default_train_status, default_options, default_playlist, dbc.Alert("Dados processados vazios.", color="warning")

            class_idx = list(le.classes_).index(target_genre)
            top_n = int(playlist_length or 10)
            chunk_size = 1000  # conservador para evitar OOM

            best = []

            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                chunk_df = processed_df.iloc[start:end][features]
                X_chunk = chunk_df.fillna(medians).values
                probs_chunk = clf.predict_proba(X_chunk)[:, class_idx]

                for i, p in enumerate(probs_chunk):
                    pos = start + i
                    if len(best) < top_n:
                        heapq.heappush(best, (p, pos))
                    else:
                        if p > best[0][0]:
                            heapq.heapreplace(best, (p, pos))

            if not best:
                return default_store, default_train_status, default_options, default_playlist, dbc.Alert("Nenhuma predição disponível.", color="warning")

            best_sorted = sorted(best, key=lambda x: -x[0])
            chosen_pos = [pos for prob, pos in best_sorted]
            chosen_prob = [prob for prob, pos in best_sorted]

            if main_store:
                main_df = to_df(main_store)
                try:
                    meta = main_df.loc[processed_df.index].iloc[chosen_pos]
                except Exception:
                    meta = main_df.iloc[chosen_pos]
                display_df = meta.copy()
            else:
                display_df = processed_df.iloc[chosen_pos].copy()

            display_df = display_df.reset_index(drop=True)
            display_df['score'] = np.round(chosen_prob, 4)

            show_cols = []
            for c in ['track_name', 'artists', 'album_name', 'track_genre']:
                if c in display_df.columns:
                    show_cols.append(c)
            if not show_cols:
                show_cols = processed_df.columns[:3].tolist()

            show_cols = show_cols + ['score']

            table = dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in show_cols],
                data=display_df[show_cols].to_dict('records'),
                page_size=playlist_length,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#111111', 'color': 'white'},
                style_cell={'backgroundColor': '#111111', 'color': 'white'}
            )

            card = dbc.Card([
                dbc.CardHeader(html.H5(f"Playlist sugerida — {target_genre} (top {playlist_length})")),
                dbc.CardBody([table])
            ], className="mt-3")

            return default_store, default_train_status, default_options, card, dbc.Alert("Playlist gerada com sucesso.", color="success")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return default_store, default_train_status, default_options, default_playlist, dbc.Alert(f"Erro ao gerar playlist: {type(e).__name__}: {e}", color="danger")

    # if nothing matched
    return no_update, no_update, no_update, no_update, no_update