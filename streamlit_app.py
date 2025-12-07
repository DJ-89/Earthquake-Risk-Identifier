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

st.markdown("""
<style>
    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
    }
    /* Style the main title */
    h1 {
        color: #ff4b4b;
        font-weight: 700;
    }
    /* Make metric labels larger */
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

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
                {'range': [0, 30], 'color': "#00cc96"},
                {'range': [30, 60], 'color': "#ffa15e"},
                {'range': [60, 100], 'color': "#ff4b4b"}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None

# Streamlit UI
st.title(" Earthquake Risk Identifier System")
st.markdown("""
This application uses a machine learning model to assess the likelihood of high seismic risk based on geographical coordinates (latitude and longitude). 
The model was trained on historical Philippine earthquake data.""")


def set_coords(lat, lon):
    st.session_state.lat_input = lat
    st.session_state.lon_input = lon

# 2. Initialize defaults if they don't exist
if 'lat_input' not in st.session_state:
    st.session_state.lat_input = 8.4803
if 'lon_input' not in st.session_state:
    st.session_state.lon_input = 124.6498

# 3. Sidebar Code
with st.sidebar:
    st.header("📍 Input Parameters")
    st.write("Enter coordinates or choose a preset.")
    
    # IMPORTANT: Remove 'value=' and just use 'key='
    latitude = st.number_input(
        "Latitude", 
        key="lat_input",
        min_value=-90.0, max_value=90.0, format="%.4f"
    )
    
    longitude = st.number_input(
        "Longitude", 
        key="lon_input",
        min_value=-180.0, max_value=180.0, format="%.4f"
    )
    
    st.markdown("---")
    predict_btn = st.button("🚀 Analyze Risk", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("📍 Quick Load Locations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Manila", on_click=set_coords, args=(14.5995, 120.9842), use_container_width=True)
        st.button("Cagayan de Oro", on_click=set_coords, args=(8.4542, 124.6319), use_container_width=True)
    with col2:
        st.button("Cebu City", on_click=set_coords, args=(10.3157, 123.8854), use_container_width=True)
        st.button("Davao City", on_click=set_coords, args=(7.1907, 125.4553), use_container_width=True)



# --- DISPLAY RESULTS (Persistent) ---
if st.session_state.risk_result is not None:
    res = st.session_state.risk_result
    prob = res['probability']
    
    # 1. PRE-CALCULATE LOGIC (So it works for both Dashboard and Map)
    if prob >= 0.60:
        risk_label = "HIGH RISK"
        risk_color = "#ff4b4b"  # Hex for text
        color_code = "red"      # Name for Folium marker
        risk_msg = f"The location **({res['lat']}, {res['lon']})** is in a critical seismic zone."
    elif prob >= 0.30:
        risk_label = "MEDIUM RISK"
        risk_color = "#ffa15e"
        color_code = "orange"
        risk_msg = f"The location **({res['lat']}, {res['lon']})** shows moderate seismic activity patterns."
    else:
        risk_label = "LOW RISK"
        risk_color = "#00cc96"
        color_code = "green"
        risk_msg = f"The location **({res['lat']}, {res['lon']})** appears relatively stable."

    st.divider()

    # 2. CREATE TABS
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🗺️ Interactive Map", "📜 History Data"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        col_kpi, col_gauge = st.columns([1, 1])

        with col_kpi:
            st.subheader("Analysis Result")
            # Use the pre-calculated variables
            st.markdown(f"<h1 style='color: {risk_color};'>⚠️ {risk_label}</h1>", unsafe_allow_html=True)
            st.write(risk_msg)

            st.markdown("---")
            
            # Professional Metric Card
            st.metric(
                label="Zone Risk Score", 
                value=f"{res['zone_risk']:.1%}", 
                delta="Based on historical density", 
                delta_color="off"
            )

        with col_gauge:
            st.plotly_chart(create_gauge(prob), use_container_width=True)

    # --- TAB 2: MAP ---
    with tab2:
        st.subheader("Geographic Risk Visualization")
        
        m = folium.Map(location=[res['lat'], res['lon']], zoom_start=10, tiles="CartoDB positron")
        
        # Use 'color_code' determined above
        folium.Marker(
            [res['lat'], res['lon']], 
            popup="Analyzed Location", 
            icon=folium.Icon(color=color_code, icon="info-sign")
        ).add_to(m)
        
        folium.Circle(
            radius=20000, 
            location=[res['lat'], res['lon']],
            color=color_code,
            fill=True,
            fill_opacity=0.1
        ).add_to(m)
        
        st_folium(m, height=400, use_container_width=True)

    # --- TAB 3: HISTORY ---
    with tab3:
        st.subheader("Historical Earthquakes (50km Radius)")
        nearby_quakes = get_nearby_quakes(res['lat'], res['lon'], raw_data)
        
        if not nearby_quakes.empty:
            st.dataframe(
                nearby_quakes[['Date_Time_PH', 'Magnitude', 'Depth_In_Km', 'Location']], 
                use_container_width=True
            )
        else:
            st.caption("No significant historical records found within 50km.")

# Footer remains outside the if block
st.divider()
st.caption("Note: This risk assessment is for informational purposes only and must not be interpreted as a definitive forecast.")