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
from folium.plugins import HeatMap
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
    Predict seismic risk. Uses raw data density if zone lookup fails.
    """
    depth = 10.0  # Fixed default depth
    
    # 1. Create Input DataFrame
    input_df = pd.DataFrame({
        'Latitude': [lat],
        'Longitude': [lon],
        'Depth_In_Km': [depth]
    })
    
    # 2. Feature Engineering (Must match training)
    input_df['Latitude_abs'] = np.abs(input_df['Latitude'])
    input_df['Longitude_abs'] = np.abs(input_df['Longitude'])
    input_df['distance_from_center'] = np.sqrt(input_df['Latitude']**2 + input_df['Longitude']**2)
    input_df['depth_log'] = np.log1p(input_df['Depth_In_Km'])
    input_df['depth_normalized'] = input_df['Depth_In_Km'] / 100.0
    input_df['lat_long_interact'] = input_df['Latitude'] * input_df['Longitude']
    input_df['lat_depth'] = input_df['Latitude'] * input_df['Depth_In_Km']
    input_df['long_depth'] = input_df['Longitude'] * input_df['Depth_In_Km']
    
    # 3. DETERMINE ZONE RISK SCORE
    # Try to use the Lookup Table first
    use_fallback = True
    zone_risk = 0.5
    
    if not zone_risk_lookup.empty:
        # Check if we have the necessary coordinate columns to find the closest zone
        cols = zone_risk_lookup.reset_index().columns
        if 'Latitude_mean' in cols and 'Longitude_mean' in cols:
            zone_risk_lookup_reset = zone_risk_lookup.reset_index()
            distances = np.sqrt(
                (zone_risk_lookup_reset['Latitude_mean'] - lat)**2 + 
                (zone_risk_lookup_reset['Longitude_mean'] - lon)**2
            )
            closest_zone_idx = distances.idxmin()
            closest_zone = zone_risk_lookup_reset.loc[closest_zone_idx, 'seismic_zone']
            zone_risk = zone_risk_lookup.loc[closest_zone, 'zone_risk_score']
            use_fallback = False # We successfully found a zone

    # 4. SMART FALLBACK (If lookup failed, calculate from Raw Data)
    if use_fallback:
        # Calculate local density risk: (Sum of Mags within 100km) / Normalization Factor
        # This ensures the score CHANGES based on location
        
        # Approx 1 degree = 111km. Search radius ~1 degree.
        lat_min, lat_max = lat - 1.0, lat + 1.0
        lon_min, lon_max = lon - 1.0, lon + 1.0
        
        nearby = raw_data[
            (raw_data['Latitude'].between(lat_min, lat_max)) & 
            (raw_data['Longitude'].between(lon_min, lon_max))
        ]
        
        if not nearby.empty:
            # Calculate a risk score (0.0 to 1.0) based on earthquake density & magnitude
            # Heuristic: If total magnitude sum > 50, risk is 0.95 (High)
            total_mag = nearby['Magnitude'].sum()
            # Normalize to 0.1 - 0.9 range
            zone_risk = min(0.95, max(0.05, total_mag / 2000.0))
        else:
            zone_risk = 0.1  # Very low risk if no history
            
    # 5. Finalize Input
    input_df['zone_risk_score'] = zone_risk
    # Use 0 for zone ID if unknown, it has low impact compared to risk score
    input_df['seismic_zone'] = 0 
    
    # 6. Predict
    X_input = input_df[feature_columns]
    X_scaled = scaler.transform(X_input)
    
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
    st.session_state.lat_input = 0
if 'lon_input' not in st.session_state:
    st.session_state.lon_input = 0

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
    
    col1, col2,= st.columns(2)
    with col1:
        st.button("Manila", on_click=set_coords, args=(14.5995, 120.9842), use_container_width=True)
        st.button("Cagayan de Oro", on_click=set_coords, args=(8.4542, 124.6319), use_container_width=True)
        st.button("Palawan", on_click=set_coords, args=(9.457, 118.39), use_container_width=True)
    with col2:
        st.button("Cebu City", on_click=set_coords, args=(10.3157, 123.8854), use_container_width=True)
        st.button("Davao City", on_click=set_coords, args=(7.1907, 125.4553), use_container_width=True)


# --- PREDICTION LOGIC ---
# This block must be OUTSIDE the 'with st.sidebar:' indentation

if predict_btn:
    with st.spinner("Analyzing seismic risk..."):
        # 1. Get the latest coordinates directly from Session State
        # This fixes the "cannot analyze" error by ensuring we have the values
        current_lat = st.session_state.lat_input
        current_lon = st.session_state.lon_input
        
        # 2. Run Prediction
        risk_prediction, risk_probability, zone_risk = predict_risk(current_lat, current_lon)
        
        # 3. Save results
        st.session_state.risk_result = {
            'prediction': risk_prediction,
            'probability': risk_probability,
            'zone_risk': zone_risk,
            'lat': current_lat,
            'lon': current_lon
        }

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
    # --- TAB 1: DASHBOARD ---
    with tab1:
        # 1. Create columns inside the tab
        col_kpi, col_gauge = st.columns([1, 1])

        # 2. Fill the LEFT column
        with col_kpi:
            st.subheader("Analysis Result")
            
            # Status Box (Red/Orange/Green)
            if prob >= 0.60:
                with st.container(border=True): 
                    st.error("### ⚠️ HIGH RISK DETECTED")
                    st.write(risk_msg)
            elif prob >= 0.30:
                with st.container(border=True):
                    st.warning("### ⚠️ MEDIUM RISK ZONE")
                    st.write(risk_msg)
            else:
                with st.container(border=True):
                    st.success("### ✅ LOW RISK ZONE")
                    st.write(risk_msg)

            st.markdown("---")
            
            
            st.metric(
                label="Zone Risk Score", 
                value=f"{res['zone_risk']:.1%}", 
                delta="Based on historical density", 
                delta_color="off"
            )

        # 3. Fill the RIGHT column
        with col_gauge:
            st.plotly_chart(create_gauge(prob), use_container_width=True)


    st.markdown("---")
    st.subheader("📋 Recommended Safety Actions")
    
    # Define advice based on risk level
    if prob >= 0.60:
        with st.expander("🚨 CRITICAL PRECAUTIONS (Click to Expand)", expanded=True):
            st.markdown("""
            * **Structural Check:** Immediately consult a structural engineer to inspect your building's integrity.
            * **Emergency Kit:** Ensure a "Go Bag" is packed (water, non-perishable food, flashlight, first aid).
            * **Drills:** Conduct earthquake drills with family/employees weekly.
            * **Furniture:** Secure tall furniture (bookshelves, cabinets) to walls.
            """)
    elif prob >= 0.30:
        with st.expander("⚠️ PRECAUTIONARY MEASURES (Click to Expand)", expanded=True):
            st.markdown("""
            * **Review Hazards:** Check for hanging objects above beds or workspaces.
            * **Communication Plan:** Agree on a meeting point for your family in case of separation.
            * **Supplies:** Maintain a 3-day supply of food and water.
            """)
    else:
        with st.expander("✅ ROUTINE MAINTENANCE (Click to Expand)"):
            st.markdown("""
            * **Stay Informed:** Keep monitoring local PHIVOLCS advisories.
            * **Standard Prep:** Keep a basic first aid kit accessible.
            * **Insurance:** Review property insurance coverage for natural disasters.
            """)
        

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
        # Use column_config to create the visual bars
        st.dataframe(
            nearby_quakes[['Date_Time_PH', 'Magnitude', 'Depth_In_Km', 'Location']], 
            use_container_width=True,
            column_config={
                "Magnitude": st.column_config.ProgressColumn(
                    "Magnitude",
                    help="Earthquake Magnitude",
                    format="%.1f",
                    min_value=0,
                    max_value=10
                    # Note: The bar color will automatically match your 
                    # primaryColor (Red) defined in .streamlit/config.toml
                ),
                "Date_Time_PH": st.column_config.DatetimeColumn(
                    "Date", format="D MMM YYYY, h:mm a"
                )
            }
        )
    else:
        st.caption("No significant historical records found within 50km.")

# Footer remains outside the if block
st.divider()
st.caption("Note: This risk assessment is for informational purposes only and must not be interpreted as a definitive forecast.")