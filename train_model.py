import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')

print("1. LOADING DATA...")
try:
    # Ensure you use the correct file path
    df = pd.read_csv('phivolcs_earthquake_data.csv')
except FileNotFoundError:
    print("Error: 'phivolcs_earthquake_data.csv' not found.")
    exit()

# --- PREPROCESSING ---
print("2. CLEANING DATA...")
numeric_cols = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)

# --- TARGET CREATION ---
# We still use Depth here to TEACH the model what a "bad" area looks like.
# But we won't use it as an input for prediction later.
df['high_risk_area'] = (
    (df['Depth_In_Km'] <= 15) &  # Shallow quakes (historically dangerous)
    (df['Magnitude'] >= 4.0)     # Significant activity
).astype(int)

# --- DBSCAN CLUSTERING ---
print("3. PERFORMING SPATIAL CLUSTERING...")
# We cluster ONLY on location (Lat/Lon)
spatial_coords = df[['Latitude', 'Longitude']].values

# Calculate optimal epsilon (standard DBSCAN approach)
min_pts = 5
nearest_neighbors = NearestNeighbors(n_neighbors=min_pts)
neighbors = nearest_neighbors.fit(spatial_coords)
distances, indices = neighbors.kneighbors(spatial_coords)
distances = np.sort(distances[:, min_pts-1], axis=0)
optimal_eps = distances[int(0.85 * len(distances))] 

# Apply DBSCAN
dbscan = DBSCAN(eps=optimal_eps, min_samples=min_pts)
df['seismic_zone'] = dbscan.fit_predict(spatial_coords)

# Calculate historical stats for each zone
# This allows the model to know "Zone 5 usually has deep quakes" without asking the user for depth
zone_stats = df.groupby('seismic_zone').agg({
    'Depth_In_Km': ['mean', 'count'], # Historical average depth
    'Magnitude': 'mean'
}).fillna(0)

zone_stats.columns = ['_'.join(col).strip() for col in zone_stats.columns]

# Create a risk score for the ZONE (based on history)
# This is a lookup value, not a user input
zone_stats['zone_risk_score'] = (
    (1 / (zone_stats['Depth_In_Km_mean'] + 1)) + 
    (zone_stats['Depth_In_Km_count'] / zone_stats['Depth_In_Km_count'].max())
) / 2

df = df.merge(zone_stats[['zone_risk_score']], left_on='seismic_zone', right_index=True, how='left')

# --- FEATURE ENGINEERING (LOCATION ONLY) ---
print("4. ENGINEERING FEATURES (COORDINATES ONLY)...")

# Generate features derived ONLY from Latitude and Longitude
df['Latitude_abs'] = np.abs(df['Latitude'])
df['Longitude_abs'] = np.abs(df['Longitude'])
df['distance_from_center'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)
df['lat_long_interact'] = df['Latitude'] * df['Longitude']

# --- CRITICAL CHANGE HERE ---
# Removed 'Depth_In_Km', 'depth_log', etc. from inputs.
feature_columns = [
    'Latitude', 
    'Longitude', 
    'seismic_zone', 
    'zone_risk_score',       # Historical risk of this location
    'Latitude_abs', 
    'Longitude_abs', 
    'distance_from_center', 
    'lat_long_interact'
]

X = df[feature_columns]
y = df['high_risk_area']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- XGBOOST TRAINING ---
print("5. TRAINING MODEL...")

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,             # Increased slightly to learn complex spatial borders
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# --- THRESHOLD OPTIMIZATION ---
print("6. OPTIMIZING...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_threshold = thresholds_pr[np.argmax(f1_scores)]
print(f"Optimal Threshold: {best_threshold:.4f}")

# --- EVALUATION ---
print("\n" + "="*30)
print("RESULTS (Location Only)")
print("="*30)
y_pred_final = (y_pred_proba >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_final))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# --- SAVING ---
print("\nSaving files...")
joblib.dump(model, 'risk_area_identifier.pkl')
joblib.dump(scaler, 'scaler_risk_identifier.pkl')
joblib.dump(dbscan, 'dbscan_zone_identifier.pkl')
joblib.dump(best_threshold, 'threshold_risk_identifier.pkl')
joblib.dump(feature_columns, 'feature_cols_risk_identifier.pkl')
joblib.dump(zone_stats, 'zone_stats_lookup.pkl') # Save this for the API to look up zone scores

print("Done. Model now accepts ONLY Latitude/Longitude.")