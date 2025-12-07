import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')

print("1. LOADING DATA...")
try:
    df = pd.read_csv('phivolcs_earthquake_data.csv')
except FileNotFoundError:
    print("Error: 'phivolcs_earthquake_data.csv' not found. Please check the file name.")
    exit()

# --- PREPROCESSING ---
print("2. CLEANING DATA...")
numeric_cols = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols)

# Create a target based on seismic activity patterns (not magnitude directly)
# This represents "high-risk area" based on historical patterns
df['high_risk_area'] = (
    (df['Depth_In_Km'] <= 15) &  # Shallow quakes are often more dangerous
    (df['Magnitude'] >= 4.0)     # Historical significant activity
).astype(int)

# --- DBSCAN CLUSTERING (For Risk Area Identification) ---
print("3. PERFORMING DBSCAN CLUSTERING...")

# Prepare spatial coordinates for clustering (only lat/lon for location-based clustering)
spatial_coords = df[['Latitude', 'Longitude']].values

# Find optimal epsilon using k-distance plot
def find_optimal_eps(X, min_pts):
    nearest_neighbors = NearestNeighbors(n_neighbors=min_pts)
    neighbors = nearest_neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)
    distances = np.sort(distances[:, min_pts-1], axis=0)
    return distances

# Calculate optimal epsilon
min_pts = 5
distances = find_optimal_eps(spatial_coords, min_pts)
optimal_eps = distances[int(0.85 * len(distances))]  # Use 85th percentile

# Apply DBSCAN - This identifies geographical zones
dbscan = DBSCAN(eps=optimal_eps, min_samples=min_pts)
df['seismic_zone'] = dbscan.fit_predict(spatial_coords)

# CRITICAL: Calculate zone risk WITHOUT using the target variable directly
# Instead, use depth patterns and cluster density as risk indicators
zone_stats = df.groupby('seismic_zone').agg({
    'Depth_In_Km': ['mean', 'std', 'count'],
    'Magnitude': 'mean',  # But don't use this for target calculation
    'Latitude': 'mean',
    'Longitude': 'mean'
}).fillna(0)

# Flatten column names
zone_stats.columns = ['_'.join(col).strip() for col in zone_stats.columns]

# Calculate risk score based on depth patterns and density (not target variable)
# Shallow depth + high density = higher risk
zone_stats['depth_risk'] = 1 / (zone_stats['Depth_In_Km_mean'] + 1)  # Inverse relationship
zone_stats['density_risk'] = zone_stats['Depth_In_Km_count'] / zone_stats['Depth_In_Km_count'].max()  # Normalize
zone_stats['zone_risk_score'] = (zone_stats['depth_risk'] + zone_stats['density_risk']) / 2

# Merge back to main dataframe
df = df.merge(zone_stats[['zone_risk_score']], left_on='seismic_zone', right_index=True, how='left')

# --- FEATURE ENGINEERING (Without Magnitude as Input) ---
print("4. ENGINEERING FEATURES FOR RISK IDENTIFICATION...")

# Create additional features
df['Latitude_abs'] = np.abs(df['Latitude'])
df['Longitude_abs'] = np.abs(df['Longitude'])
df['distance_from_center'] = np.sqrt(df['Latitude']**2 + df['Longitude']**2)
df['depth_log'] = np.log1p(df['Depth_In_Km'])
df['depth_normalized'] = df['Depth_In_Km'] / df['Depth_In_Km'].max()
df['lat_long_interact'] = df['Latitude'] * df['Longitude']
df['lat_depth'] = df['Latitude'] * df['Depth_In_Km']
df['long_depth'] = df['Longitude'] * df['Depth_In_Km']

# Features for XGBoost - NO MAGNITUDE (since we don't know it for future predictions)
feature_columns = [
    'Latitude', 'Longitude', 'Depth_In_Km',
    'seismic_zone', 'zone_risk_score',
    'Latitude_abs', 'Longitude_abs', 'distance_from_center',
    'depth_log', 'depth_normalized', 'lat_long_interact', 
    'lat_depth', 'long_depth'
]

X = df[feature_columns]
y = df['high_risk_area']  # Target based on historical patterns

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Target distribution - Train: {pd.Series(y_train).value_counts()}")
print(f"Target distribution - Test: {pd.Series(y_test).value_counts()}")

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- XGBOOST FOR RISK PREDICTION ---
print("5. TRAINING XGBOOST MODEL (RISK IDENTIFICATION)...")

# Calculate class weights
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# XGBoost model - predicting risk areas based on location and depth
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.2,
    reg_lambda=0.5,
    eval_metric='logloss',
    n_jobs=-1,
    random_state=42
)

# Train the model
model.fit(
    X_train_scaled, y_train,
    verbose=False
)

# --- THRESHOLD OPTIMIZATION ---
print("6. OPTIMIZING THRESHOLD...")
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics for threshold selection
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
weight_precision = 0.3
weight_recall = 0.7
weighted_score = weight_precision * precisions + weight_recall * recalls
best_weighted_idx = np.argmax(weighted_score)
best_threshold = thresholds_pr[best_weighted_idx]

print(f"   - Optimal Threshold: {best_threshold:.4f}")

y_pred_final = (y_pred_proba >= best_threshold).astype(int)

# --- EVALUATION ---
print("\n" + "="*60)
print("RISK AREA IDENTIFICATION RESULTS")
print("="*60)
print(classification_report(y_test, y_pred_final))
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean CV F1 score: {cv_scores.mean():.4f}")
print(f"Std CV F1 score: {cv_scores.std():.4f}")

# Check for overfitting
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")
print(f"Gap (indicates overfitting): {train_score - test_score:.4f}")

# --- SAVING ARTIFACTS ---
print("\nSaving model files...")
joblib.dump(model, 'risk_area_identifier.pkl')
joblib.dump(scaler, 'scaler_risk_identifier.pkl')
joblib.dump(dbscan, 'dbscan_zone_identifier.pkl')
joblib.dump(best_threshold, 'threshold_risk_identifier.pkl')
joblib.dump(feature_columns, 'feature_cols_risk_identifier.pkl')

# Create zone risk lookup
zone_risk_lookup = df[['seismic_zone', 'zone_risk_score']].drop_duplicates().set_index('seismic_zone')
joblib.dump(zone_risk_lookup, 'zone_risk_lookup.pkl')

print("All files saved successfully.")
print("\nMODEL READY: Risk identification using only coordinates and depth!")