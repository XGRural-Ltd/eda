import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import numpy as np

# --- STEP 1: Use the same data source as Home.py ---
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
    exit()

# --- STEP 2: Define the FULL list of features for the model ---
selected_features = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
    'key', 'loudness', 'mode', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'
]

# The feature to be predicted
target_col = 'popularity'

# The actual features used for training (all except the target)
training_features = [col for col in selected_features if col != target_col]

# --- STEP 3: Prepare the data ---
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
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(training_features, 'selected_features.pkl')

print("\n✅ Model, scaler, and feature list saved successfully!")