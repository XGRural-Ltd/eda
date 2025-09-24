import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Carregamento dos Dados ---
path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
df = pd.read_csv(path + '/dataset.csv', index_col=0)
if 'track_genre' not in df.columns:
    df['track_genre'] = 'unknown'

num_features = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature'
]
df = df[num_features + ['track_genre']].copy()

# --- Detecção e remoção de outliers ---
features_for_outliers = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'instrumentalness']
scaler_out = StandardScaler()
X_out = scaler_out.fit_transform(df[features_for_outliers])
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(X_out)
df_no_outliers = df[df['anomaly'] != -1].copy()

# --- Agrupar gêneros pouco frequentes ---
top_n = 10
top_genres = df_no_outliers['track_genre'].value_counts().nlargest(top_n).index
df_no_outliers['track_genre_grouped'] = df_no_outliers['track_genre'].where(df_no_outliers['track_genre'].isin(top_genres), 'outros')

# Remover a classe 'outros' ANTES de criar df_processed
df_no_outliers = df_no_outliers[df_no_outliers['track_genre_grouped'] != 'outros'].copy()

# --- Imputação e padronização ---
df_no_outliers[num_features] = df_no_outliers[num_features].fillna(df_no_outliers[num_features].median())
scaler = StandardScaler()
scaler_norm = MinMaxScaler()
df_processed = df_no_outliers.copy()
df_processed[num_features] = scaler.fit_transform(df_processed[num_features])
df_processed[num_features] = scaler_norm.fit_transform(df_processed[num_features])

# --- Seleção de features importantes ---
rf_temp = RandomForestClassifier(n_estimators=30, random_state=42)
rf_temp.fit(df_processed[num_features], df_processed['track_genre_grouped'])
importances = pd.Series(rf_temp.feature_importances_, index=num_features).sort_values(ascending=False)
selected_features = importances.head(10).index.tolist()

# --- Balanceamento com SMOTE ---
X = df_processed[selected_features]
y = df_processed['track_genre_grouped']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y_encoded)

max_samples = 5000
if X_bal.shape[0] > max_samples:
    X_bal, y_bal = resample(X_bal, y_bal, n_samples=max_samples, random_state=42, stratify=y_bal)

# --- Split treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal)

# --- Treinamento do melhor modelo (Random Forest) ---
rf = RandomForestClassifier(n_estimators=70, random_state=42)
rf.fit(X_train, y_train)

# --- Exportação dos objetos necessários ---
joblib.dump(rf, 'random_forest_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(scaler_norm, 'scaler_norm.pkl')
joblib.dump(selected_features, 'selected_features.pkl')