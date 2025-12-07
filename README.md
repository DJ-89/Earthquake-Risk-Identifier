# Seismic Risk Prediction Application

This application uses a machine learning model to predict seismic risk based on geographical coordinates (latitude and longitude). The model was trained on historical Philippine earthquake data using XGBoost and DBSCAN clustering.

## Features

- Predict seismic risk based on latitude, longitude, and depth inputs
- Visualize locations on a map
- Detailed risk assessment with probability scores
- Historical pattern analysis

## Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: Latitude, Longitude, Depth, Seismic Zone, and derived features
- **Training Data**: Historical Philippine earthquake data
- **Performance**: AUC-ROC Score ~0.88
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

## Deployment Instructions

### Local Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

3. Access the application at `http://localhost:8501`

### Cloud Deployment (Heroku, Streamlit Cloud, etc.)

1. Make sure all model files are in the root directory
2. Use the `requirements.txt` file for dependency installation
3. Set the entry point to run `streamlit_app.py`

## Usage

1. Enter latitude and longitude coordinates (default is Manila coordinates)
2. Adjust the depth of the hypothetical seismic event
3. Click "Calculate Risk" to get the risk assessment
4. View detailed analysis and probability scores

## Important Note

This application uses a machine learning model trained on historical data. Predictions should not be used as the sole basis for critical decisions and should not replace professional geological assessment.

## Model Architecture

The model combines:
1. DBSCAN clustering to identify seismic zones
2. Feature engineering with latitude, longitude, and depth-based features
3. XGBoost classifier to predict high-risk areas
4. Threshold optimization for balanced precision and recall