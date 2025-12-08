# Seismic Risk Prediction Application

This application uses a machine learning model to predict seismic risk based on geographical coordinates (latitude and longitude). The model was trained on historical Philippine earthquake data using XGBoost and DBSCAN clustering.

## Features

- Predict seismic risk based on latitude, longitude
- Visualize locations on a map
- Detailed risk assessment with probability scores
- Historical pattern analysis

## Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: Latitude, Longitude, Depth, Seismic Zone, and derived features
- **Training Data**: Historical Philippine earthquake data
- **Performance**: AUC-ROC Score ~0.75
- **Risk Classification**: High Risk areas are defined as locations with shallow earthquakes (≤15km depth) and magnitude ≥4.0

## Files Included

- `streamlit_app.py` - The main Streamlit application
- `train_model.py` - Script to train the model (already executed)
- `requirements.txt` - Python dependencies
- Model files (generated during training):
  - `risk_area_identifier.pkl` - Trained XGBoost model
  - `scaler_risk_identifier.pkl` - Feature scaler
  - `dbscan_zone_identifier.pkl` - DBSCAN clustering model
  - `threshold_risk_identifier.pkl` - Classification threshold
  - `feature_cols_risk_identifier.pkl` - Feature column names
  - `zone_risk_lookup.pkl` - Zone risk lookup table



## Usage

1. Enter latitude and longitude coordinates 
2. Click "Calculate Risk" to get the risk assessment
3. View detailed analysis and probability scores

## Important Note

This application uses a machine learning model trained on historical data. This risk assessment is for informational purposes only and must not be interpreted as a definitive forecast.
## Model Architecture

The model combines:
1. DBSCAN clustering to identify seismic zones
2. Feature engineering with latitude, longitude, and depth-based features
3. XGBoost classifier to predict high-risk areas
4. Threshold optimization for balanced precision and recall
