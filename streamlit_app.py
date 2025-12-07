import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Earthquake Risk Identifier",
    page_icon=" 🛡️",
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
    # NEW: Load the raw data for the history table
    raw_data = pd.read_csv('phivolcs_earthquake_data.csv')
    return model, scaler, dbscan, threshold, feature_columns, zone_risk_lookup, raw_data


model, scaler, dbscan, threshold, feature_columns, zone_risk_lookup, raw_data = load_model()

# Function to predict risk based on latitude and longitude only
def predict_risk(lat, lon):
    """
    Predict seismic risk based on latitude and longitude only
    Depth is fixed to a default value (10km) since it's not an input anymore
    """
    depth = 10.0  # Fixed default depth
    
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

def get_nearby_quakes(lat, lon, df, radius_km=50):
    """Finds historical quakes within radius_km"""
    # Approx: 1 deg lat ~= 111km
    lat_min, lat_max = lat - (radius_km/111), lat + (radius_km/111)
    lon_min, lon_max = lon - (radius_km/111), lon + (radius_km/111)
    
    nearby = df[
        (df['Latitude'].between(lat_min, lat_max)) & 
        (df['Longitude'].between(lon_min, lon_max))
    ].copy()
    
    # Calculate exact distance
    nearby['dist'] = np.sqrt((nearby['Latitude']-lat)**2 + (nearby['Longitude']-lon)**2) * 111
    return nearby[nearby['dist'] <= radius_km].sort_values('Magnitude', ascending=False).head(10)

def create_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Probability", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "#00cc96"},
                {'range': [40, 70], 'color': "#ffa15e"},
                {'range': [70, 100], 'color': "#ff4b4b"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Streamlit UI
st.title(" Earthquake Risk Identifier System")
st.markdown("""
This application uses a machine learning model to predict the likelihood of high seismic risk 
based on geographical coordinates (latitude and longitude). The model was trained on historical 
Philippine earthquake data.
""")

# Create input columns (only 2 now)
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude", 
        value=8.4803,  # Default to CDO coordinates
        min_value=-90.0, 
        max_value=90.0, 
        format="%.4f",
        help="Enter latitude in decimal degrees (e.g., 8.4803 for CDO)"
    )

with col2:
    longitude = st.number_input(
        "Longitude", 
        value=124.6498,  # Default to CDO coordinates
        min_value=-180.0, 
        max_value=180.0, 
        format="%.4f",
        help="Enter longitude in decimal degrees (e.g., 124.6498 for CDO)"
    )

# Prediction button
if st.button("  Calculate Risk ", type="primary"):
    with st.spinner("Analyzing seismic risk..."):
        risk_prediction, risk_probability, zone_risk = predict_risk(latitude, longitude)
        
        # Display results
        st.divider()
        
        # --- NEW DASHBOARD LAYOUT ---
        col_kpi, col_gauge = st.columns([1, 1])

        with col_kpi:
            st.subheader("Analysis Result")
            if risk_prediction:
                st.markdown(f"<h1 style='color: #ff4b4b;'>⚠️ HIGH RISK</h1>", unsafe_allow_html=True)
                st.write(f"The location **({latitude}, {longitude})** is identified as a high-risk seismic zone.")
            else:
                st.markdown(f"<h1 style='color: #00cc96;'>✅ LOW RISK</h1>", unsafe_allow_html=True)
                st.write(f"The location **({latitude}, {longitude})** appears relatively stable based on historical patterns.")
            
            st.info(f"Zone Risk Score: {zone_risk:.1%}")

        with col_gauge:
            st.plotly_chart(create_gauge(risk_probability), use_container_width=True)

        # --- HISTORY TABLE ---
        st.subheader("📜 Nearby Historical Earthquakes (50km Radius)")
        nearby_quakes = get_nearby_quakes(latitude, longitude, raw_data)
        
        if not nearby_quakes.empty:
            st.dataframe(
                nearby_quakes[['Date_Time_PH', 'Magnitude', 'Depth_In_Km', 'Location']],
                use_container_width=True
            )
        else:
            st.caption("No significant historical records found within 50km.")
        
        # Additional information
        st.info("""
        **About this prediction:**
        - This model uses historical earthquake data to identify areas with high seismic activity patterns
        - Risk is determined based on shallow depth events (≤15km) with magnitude ≥4.0
        - The prediction considers geographical clustering patterns
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
    
    st.divider()
st.subheader("🗺️ Interactive Risk Map")

# Create Folium Map
m = folium.Map(location=[latitude, longitude], zoom_start=9, tiles="CartoDB positron")

# Add Marker
folium.Marker(
    [latitude, longitude], 
    popup="Selected Location", 
    icon=folium.Icon(color="red", icon="info-sign")
).add_to(m)

# Add Radius Circle
folium.Circle(
    radius=20000, # 20km
    location=[latitude, longitude],
    color="red",
    fill=True,
    fill_opacity=0.1
).add_to(m)

# Render Map
st_data = st_folium(m, width="100%", height=400)
# Footer
st.divider()
st.caption("Note: This application uses a machine learning model trained on historical data. Predictions should not be used as the sole basis for critical decisions.")