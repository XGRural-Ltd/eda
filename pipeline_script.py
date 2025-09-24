import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np

# --- STEP 1: Use the EXACT SAME data source as Home.py ---
# This ensures data consistency.
try:
    path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
    path_dataset = path + '/dataset.csv'
    df = pd.read_csv(path_dataset, index_col=0)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'track_genre' not in df.columns:
        df['track_genre'] = 'unknown'
    print("Dataset loaded successfully from URL.")
except Exception as e:
    print(f"Failed to load data from URL: {e}")
    exit() # Exit if we can't get the data

# --- STEP 2: Define the FULL list of features for the model ---
# This list must match what your app will provide.
# We are including 'liveness' and 'time_signature' here.
selected_features = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
]

# The feature to be predicted
target_col = 'popularity'

<<<<<<< HEAD
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
=======
# The actual features used for training (all except the target)
training_features = [col for col in selected_features if col != target_col]

# --- STEP 3: Prepare the data ---
# Drop rows with missing values specifically in the columns we will use
df_model = df[selected_features].dropna()

X = df_model[training_features]
y = df_model[target_col]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: Scale, Train, and Evaluate the Model ---
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=2)
print("\nTraining model...")
model.fit(X_train_scaled, y_train)
print("Training complete.")

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print(f"\nModel Evaluation:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# --- STEP 5: Save the CORRECT model, scaler, and feature list ---
joblib.dump(model, 'random_forest_model.pkl')
>>>>>>> origin/c_dev
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(training_features, 'selected_features.pkl') # Save the correct, complete list

print("\n✅ Model, scaler, and feature list saved successfully!")