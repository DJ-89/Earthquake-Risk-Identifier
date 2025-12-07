import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Seismic Risk Prediction",
    page_icon=" earthquak",
    layout="wide"
)

# Load the pre-trained model and other artifacts
@st.cache_resource
def load_model():
    model = joblib.load('risk_area_identifier.pkl')
    scaler = joblib.load('scaler_risk_identifier.pkl')
    dbscan = joblib.load('dbscan_zone_identifier.pkl')
    threshold = joblib.load('threshold_risk_identifier.pkl')
    feature_columns = joblib.load('feature_cols_risk_identifier.pkl')
    zone_risk_lookup = joblib.load('zone_risk_lookup.pkl')
    return model, scaler, dbscan, threshold, feature_columns, zone_risk_lookup

model, scaler, dbscan, threshold, feature_columns, zone_risk_lookup = load_model()

# Function to predict risk based on latitude and longitude
def predict_risk(lat, lon, depth=10.0):
    """
    Predict seismic risk based on latitude, longitude, and depth
    Default depth is set to 10km as a reasonable average for shallow quakes
    """
    # Create a dataframe with the input values
    input_df = pd.DataFrame({
        'Latitude': [lat],
        'Longitude': [lon],
        'Depth_In_Km': [depth]
    })
    
    # Predict the seismic zone using DBSCAN
    # For prediction, we'll use the zone that is closest to the input coordinates
    # Since we can't directly predict DBSCAN cluster for a new point, 
    # we'll create a simplified approach using the zone risk lookup
    # First, let's find the closest zone to the input coordinates
    
    # For this implementation, we'll calculate the zone based on distance to known zones
    # This is a simplified approach since DBSCAN can't predict new points directly
    # We'll use the pre-calculated zone information
    
    # Create additional features as used in training
    input_df['Latitude_abs'] = np.abs(input_df['Latitude'])
    input_df['Longitude_abs'] = np.abs(input_df['Longitude'])
    input_df['distance_from_center'] = np.sqrt(input_df['Latitude']**2 + input_df['Longitude']**2)
    input_df['depth_log'] = np.log1p(input_df['Depth_In_Km'])
    input_df['depth_normalized'] = input_df['Depth_In_Km'] / 100.0  # Using 100km as max depth reference
    input_df['lat_long_interact'] = input_df['Latitude'] * input_df['Longitude']
    input_df['lat_depth'] = input_df['Latitude'] * input_df['Depth_In_Km']
    input_df['long_depth'] = input_df['Longitude'] * input_df['Depth_In_Km']
    
    # Determine seismic zone - for simplicity, we'll assign a default zone risk
    # In a real implementation, you'd need to use the DBSCAN model to find the closest zone
    # For now, we'll use a simplified approach to find the closest zone based on coordinates
    
    # For this example, let's assign a default zone risk since DBSCAN can't predict new clusters directly
    # We'll use the average zone risk or find the closest zone
    if not zone_risk_lookup.empty:
        # Simplified approach: find the zone with the closest coordinates
        zone_risk_lookup_reset = zone_risk_lookup.reset_index()
        if 'Latitude_mean' in zone_risk_lookup_reset.columns and 'Longitude_mean' in zone_risk_lookup_reset.columns:
            # Calculate distances to all known zones and assign the closest one
            distances = np.sqrt(
                (zone_risk_lookup_reset['Latitude_mean'] - lat)**2 + 
                (zone_risk_lookup_reset['Longitude_mean'] - lon)**2
            )
            closest_zone_idx = distances.idxmin()
            closest_zone = zone_risk_lookup_reset.loc[closest_zone_idx, 'seismic_zone']
            zone_risk = zone_risk_lookup.loc[closest_zone, 'zone_risk_score']
        else:
            # If zone coordinates are not available, use the average risk
            zone_risk = zone_risk_lookup['zone_risk_score'].mean()
    else:
        zone_risk = 0.5  # Default risk value
    
    # Assign the zone risk to the input
    input_df['zone_risk_score'] = zone_risk
    # For the seismic zone, we'll use the zone of the closest match or default to 0
    input_df['seismic_zone'] = 0 if zone_risk_lookup.empty else closest_zone if 'closest_zone' in locals() else 0
    
    # Reorder to match training features
    X_input = input_df[feature_columns]
    
    # Scale the input
    X_scaled = scaler.transform(X_input)
    
    # Make prediction
    risk_prob = model.predict_proba(X_scaled)[0, 1]
    risk_prediction = risk_prob >= threshold
    
    return risk_prediction, risk_prob, zone_risk

# Streamlit UI
st.title(" earthquak Seismic Risk Prediction System")
st.markdown("""
This application uses a machine learning model to predict the likelihood of high seismic risk 
based on geographical coordinates (latitude and longitude). The model was trained on historical 
Philippine earthquake data.
""")

# Create input columns
col1, col2, col3 = st.columns(3)

with col1:
    latitude = st.number_input(
        "Latitude", 
        value=12.8797,  # Default to Manila coordinates
        min_value=-90.0, 
        max_value=90.0, 
        format="%.4f",
        help="Enter latitude in decimal degrees (e.g., 12.8797 for Manila)"
    )

with col2:
    longitude = st.number_input(
        "Longitude", 
        value=121.7740,  # Default to Manila coordinates
        min_value=-180.0, 
        max_value=180.0, 
        format="%.4f",
        help="Enter longitude in decimal degrees (e.g., 121.7740 for Manila)"
    )

with col3:
    depth = st.slider(
        "Depth (km)", 
        min_value=1.0, 
        max_value=100.0, 
        value=10.0,
        step=0.5,
        help="Depth of the hypothetical seismic event in kilometers"
    )

# Prediction button
if st.button("  Calculate Risk ", type="primary"):
    with st.spinner("Analyzing seismic risk..."):
        risk_prediction, risk_probability, zone_risk = predict_risk(latitude, longitude, depth)
        
        # Display results
        st.divider()
        st.subheader("Risk Assessment Results")
        
        # Create result columns
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric(
                label="Risk Level", 
                value="HIGH RISK" if risk_prediction else "LOW RISK",
                delta_color="inverse"
            )
        
        with res_col2:
            st.metric(
                label="Risk Probability", 
                value=f"{risk_probability:.2%}",
                delta=f"{zone_risk:.2%} zone risk"
            )
        
        with res_col3:
            st.metric(
                label="Alert Status", 
                value="⚠️ ALERT" if risk_prediction else "✅ SAFE",
                delta="Based on historical patterns" if risk_prediction else "Historical stability"
            )
        
        # Detailed results
        st.subheader("Detailed Analysis")
        
        if risk_prediction:
            st.error(f"""
            ⚠️ **HIGH RISK DETECTED**  
            The location ({latitude}, {longitude}) at depth {depth}km has been classified as a high-risk area based on historical seismic patterns.  
            Probability of significant seismic activity: **{risk_probability:.2%}**  
            Zone risk score: **{zone_risk:.2%}**
            """)
        else:
            st.success(f"""
            ✅ **LOW RISK CONFIRMED**  
            The location ({latitude}, {longitude}) at depth {depth}km has been classified as a low-risk area based on historical seismic patterns.  
            Probability of significant seismic activity: **{risk_probability:.2%}**  
            Zone risk score: **{zone_risk:.2%}**
            """)
        
        # Additional information
        st.info("""
        **About this prediction:**
        - This model uses historical earthquake data to identify areas with high seismic activity patterns
        - Risk is determined based on shallow depth events (≤15km) with magnitude ≥4.0
        - The prediction considers geographical clustering and depth patterns
        - This is for informational purposes only and should not replace professional geological assessment
        """)

# Add a map visualization section
st.divider()
st.subheader("Location Visualization")

# Create a simple map using streamlit's built-in map function
if st.button("Show Location on Map"):
    location_data = pd.DataFrame({
        'lat': [latitude],
        'lon': [longitude]
    })
    
    st.map(location_data, zoom=7)

# Model information
with st.expander("Model Information"):
    st.write("""
    **Model Details:**
    - Algorithm: XGBoost Classifier
    - Features: Latitude, Longitude, Depth, Seismic Zone, and derived features
    - Training Data: Historical Philippine earthquake data
    - Performance: AUC-ROC Score ~0.88 (as per training results)
    
    **Risk Classification:**
    - High Risk: Areas with shallow earthquakes (≤15km depth) and magnitude ≥4.0
    - Low Risk: Areas with deeper or less significant historical activity
    """)
    
    st.write(f"**Feature Columns Used:** {', '.join(feature_columns)}")

# Footer
st.divider()
st.caption("Note: This application uses a machine learning model trained on historical data. Predictions should not be used as the sole basis for critical decisions.")