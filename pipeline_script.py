import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re
import pandas as pd
import kagglehub

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split

import joblib

cols_dict = {
    'track_id': 'Track ID', 'artists': 'Artists', 'album_name': 'Album Name',
    'track_name': 'Track Name', 'popularity': 'Popularity', 'duration_ms': 'Duration (ms)',
    'explicit': 'Explicit', 'danceability': 'Danceability', 'energy': 'Energy',
    'key': 'Key', 'loudness': 'Loudness', 'mode': 'Mode', 'speechiness': 'Speechiness',
    'acousticness': 'Acousticness', 'instrumentalness': 'Instrumentalness',
    'liveness': 'Liveness', 'valence': 'Valence', 'tempo': 'Tempo',
    'time_signature': 'Time Signature', 'track_genre': 'Track Genre'
}


# --- STEP 1: Use the same data source as Home.py ---
try:
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    df = pd.read_csv(path + '/dataset.csv', index_col=0)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'track_genre' not in df.columns:
        df['track_genre'] = 'unknown'
    print("Dataset loaded successfully from URL.")
except Exception as e:
    print(f"Failed to load data from URL: {e}")
    exit()

# --- STEP 2: Define the FULL list of features for the model ---
numerical_features = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature'
]

df = df[numerical_features + ['track_genre']].copy()
data_pipeline = make_pipeline(SimpleImputer(strategy='median'), RobustScaler(), MinMaxScaler())
df_processed = df.copy()
df_processed[numerical_features] = data_pipeline.fit_transform(df_processed[numerical_features])

# --- STEP 3: Prepare the data ---
genres = sorted(df_processed['track_genre'].unique())

# 1) Mapeamento por palavras-chave (rápido, interpretável)
keyword_map = {
    'pop': 'Pop',
    'indie': 'Indie/Alt',
    'rock': 'Rock',
    'metal': 'Metal/Hard',
    'hip': 'Hip-Hop/Rap',
    'r-n-b': 'R&B/Soul',
    'soul': 'R&B/Soul',
    'electro': 'Electronic',
    'edm': 'Electronic',
    'house': 'Electronic/Dance',
    'techno': 'Electronic/Dance',
    'trance': 'Electronic/Dance',
    'dance': 'Electronic/Dance',
    'dub': 'Electronic',
    'drum': 'Electronic',
    'jazz': 'Jazz/Classical',
    'classical': 'Jazz/Classical',
    'acoustic': 'Folk/Acoustic',
    'folk': 'Folk/Acoustic',
    'country': 'Country',
    'latin': 'Latin',
    'samba': 'Latin',
    'reggae': 'Reggae/Dancehall',
    'reggaeton': 'Reggaeton/Latin',
    'blues': 'Blues',
    'punk': 'Punk',
    'emo': 'Rock',
    'ambient': 'Ambient/Chill',
    'chill': 'Ambient/Chill',
    'kids': 'Kids/Family',
    'soundtrack': 'Soundtrack',
    'opera': 'Classical/Opera',
    'world': 'World',
    'brazil': 'Latin',
    # adicione aqui conforme necessário
}

def map_genre_by_keyword(g):
    g_low = (g or '').lower()
    # verifica keywords mais longas primeiro
    for kw, cat in keyword_map.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', g_low):
            return cat
    return None

# Aplica o mapeamento por palavra-chave
manual_map = {g: (map_genre_by_keyword(g) or 'Other') for g in genres}

# Exibe contagem por categoria resultante
mapped_series = df_processed['track_genre'].map(manual_map)
# print("Contagem por categoria (após mapeamento por keyword):")
# print(mapped_series.value_counts())

# 2) Manter os gêneros mais frequentes separados e agrupar os raros
freq = df_processed['track_genre'].value_counts()
top_n = 30   # ajusta conforme desejado
top_genres = set(freq.nlargest(top_n).index)
condensed_map = {}
for g in genres:
    if g in top_genres:
        condensed_map[g] = manual_map.get(g, g)  # se já mapeado, usa categoria; senão mantém
    else:
        condensed_map[g] = 'Other'

# Aplicar condensed_map no DataFrame
df_processed['genre_grouped_manual'] = df_processed['track_genre'].map(condensed_map)
df_processed.dropna(subset=numerical_features + ['genre_grouped_manual'], inplace=True)

# --- STEP 4: Slice the data into Train and Test Split to Evaluate the Model ---
# # The actual features used for training (all except the target)
target_col = 'genre_grouped_manual'
le = LabelEncoder()
X = df_processed[numerical_features].copy()
y_enc = le.fit_transform(df_processed['genre_grouped_manual'])
y = y_enc

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete.")

# Evaluate the model
y_pred = model.predict(X_test)
print(f"\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- STEP 5: Save the CORRECT model, scaler, and feature list ---
pkl_files_path = './pkl_files/'
joblib.dump(model, pkl_files_path + 'random_forest_model.pkl')
# joblib.dump(scaler, pkl_files_path + 'scaler.pkl')
joblib.dump(numerical_features, pkl_files_path + 'selected_features.pkl')

print(f"\n✅ Random Forest model and numerical feature list saved successfully as .pkl files. at {pkl_files_path}!")