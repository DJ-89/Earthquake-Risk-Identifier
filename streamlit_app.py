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
import pydeck as pdk
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Earthquake Risk Identifier",
    page_icon=" 🛡️",
    layout="wide"
)

st.markdown("""
<style>
    /* --- EXISTING STYLES --- */
    .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #ff4b4b;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem;
    }

    /* --- STRONGER PULSE ANIMATION --- */
    /* Target the sidebar toggle button specifically */
    [data-testid="stSidebarCollapsedControl"] {
        animation: pulse 2s infinite !important;
        color: #ff4b4b !important; /* Force Red Color */
        border: 1px solid #ff4b4b !important; /* Add a red ring around it */
        border-radius: 50% !important;
        background-color: rgba(255, 75, 75, 0.1) !important; /* Faint red background */
    }
    
    /* Animation Keyframes */
    @keyframes pulse {
        0% { 
            transform: scale(1); 
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7);
        }
        70% { 
            transform: scale(1.1); 
            box-shadow: 0 0 0 10px rgba(255, 75, 75, 0);
        }
        100% { 
            transform: scale(1); 
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0);
        }
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

# --- LANDING PAGE CONTENT ---
if st.session_state.risk_result is None:
    st.markdown("---")
    
    # --- OPTION 1: QUICK STATS ---
    st.subheader("📊 Dataset Overview (Philippine Region)")

    # 1. Clean data for stats
    # Force 'Depth_In_Km' to numeric to avoid errors
    raw_data['Depth_In_Km'] = pd.to_numeric(raw_data['Depth_In_Km'], errors='coerce')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Earthquakes", f"{len(raw_data):,}")
    with col2:
        st.metric("Max Magnitude", f"{raw_data['Magnitude'].max()} Mw")
    with col3:
        avg_depth = raw_data['Depth_In_Km'].mean()
        st.metric("Avg. Depth", f"{avg_depth:.1f} km")
    with col4:
        st.metric("Data Source", "PHIVOLCS")

    st.markdown("---")

    # --- OPTION 2: MAP ---
    st.subheader("🗺️ Historical Seismic Activity")
    st.caption("Visualizing the density of recorded earthquake epicenters used for model training.")
    
    # 2. Clean data for Map
    map_data = raw_data[['Latitude', 'Longitude']].dropna()
    
    # FIX: Rename columns to lowercase 'latitude' and 'longitude' so st.map recognizes them
    map_data = map_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    
    st.map(map_data, zoom=4, color='#ff4b4b')


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
    

    # --- NEW: MOBILE CONTEXT BAR ---
    # This sits at the top of the main page so mobile users see their inputs
    with st.container():
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info(f"📍 **Analyzing:** {res['lat']:.4f}, {res['lon']:.4f}")
        with c2:
            # A visual hint pointing to the sidebar
            st.caption("↖ Open Sidebar to change")


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
        # 1. Top Section: Columns
        col_kpi, col_gauge = st.columns([1, 1])

        # Fill the LEFT column
        with col_kpi:
            st.subheader("Analysis Result")
            
            # Status Box
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

        # Fill the RIGHT column
        with col_gauge:
            st.plotly_chart(create_gauge(prob), use_container_width=True)


        # 2. Bottom Section: Safety Actions (NOW INSIDE TAB 1)
        # Note: These lines are indented to match 'col_kpi' and 'col_gauge'
        st.markdown("---")
        st.subheader("📋 Recommended Safety Actions")
        
        if prob >= 0.60:
            with st.expander("🚨 CRITICAL PRECAUTIONS (High Risk Zone)", expanded=True):
                st.markdown("""
                **Immediate Actions Required:**
                * **🏗️ Structural Inspection:** Contact a licensed civil engineer to inspect your building's foundation, beams, and columns for cracks or weaknesses immediately.
                * **🎒 Emergency "Go Bag":** Ensure every family member has a bag packed with 3 days of water, non-perishable food, flashlight, batteries, whistle, and a first-aid kit.
                * **🗄️ Secure Heavy Furniture:** Bolt bookshelves, cabinets, and water heaters to the wall to prevent them from toppling during shaking.
                * **👨‍👩‍👧‍👦 Family Drill:** Conduct "Drop, Cover, and Hold On" drills weekly. Designate a safe evacuation area outside away from power lines.
                * **📄 Document Protection:** Keep copies of important documents (IDs, land titles, insurance) in a waterproof container or cloud storage.
                """)
        elif prob >= 0.30:
            with st.expander("⚠️ PRECAUTIONARY MEASURES (Medium Risk Zone)", expanded=True):
                st.markdown("""
                **Standard Readiness Steps:**
                * **🖼️ Hazard Hunt:** Check your rooms for heavy objects hanging over beds or sofas (e.g., mirrors, shelves). Relocate them to safer spots.
                * **📍 Communication Plan:** Agree on a specific meeting point (e.g., a nearby park) if family members are separated when an earthquake hits.
                * **🔋 Backup Power:** Have a power bank fully charged and a battery-operated radio to monitor news if electricity is cut off.
                * **🥫 Supply Buffer:** Maintain a pantry with at least 3 days' worth of ready-to-eat food and 4 liters of water per person.
                """)
        else:
            with st.expander("✅ ROUTINE MAINTENANCE (Low Risk Zone)"):
                st.markdown("""
                **Maintenance & Awareness:**
                * **📢 Stay Informed:** Download the PHIVOLCS app or follow local disaster management pages for occasional advisories.
                * **🩹 Basic First Aid:** Keep a simple first aid kit accessible (bandages, antiseptic, pain relievers) and check expiration dates annually.
                * **🏠 Property Insurance:** Review your home insurance policy to see if it covers "Acts of Nature" or earthquake damage, just in case.
                * **🔦 Utility Check:** Know exactly where your main gas, water, and electric shut-off valves are located and how to turn them off.
                """)


# --- TAB 2: MAP (PyDeck with Heatmap + 50km Radius) ---
    with tab2:
        st.subheader("Geographic Risk Visualization")
        st.caption("Heatmap shows density of historical earthquakes. The circle represents a 50km radius.")

        # 1. Prepare Data
        lat, lon = res['lat'], res['lon']
        
        # Filter nearby data for the heatmap
        if 'raw_data' in locals():
            nearby_data = raw_data[
                (raw_data['Latitude'].between(lat - 1, lat + 1)) & 
                (raw_data['Longitude'].between(lon - 1, lon + 1))
            ].dropna()
        else:
            nearby_data = pd.DataFrame()

        # 2. Render PyDeck Chart
        if not nearby_data.empty:
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=lat,
                    longitude=lon,
                    zoom=9,
                    pitch=0,
                ),
                layers=[
                    # Layer 1: The Heatmap (Historical Density)
                    pdk.Layer(
                        'HeatmapLayer',
                        data=nearby_data,
                        get_position='[Longitude, Latitude]',
                        opacity=0.6,
                        threshold=0.05,
                        radiusPixels=30
                    ),
                    # Layer 2: The 50km Radius Circle
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
                        get_position='[lon, lat]',
                        get_radius=50000,          # CHANGED: 50,000 meters = 50km
                        get_fill_color=[255, 75, 75, 30], # Red with high transparency
                        stroked=True,
                        get_line_color=[255, 75, 75, 100], # Solid red border
                        line_width_min_pixels=2
                    ),
                    # Layer 3: The Exact Location Pin (Center Dot)
                    pdk.Layer(
                        'ScatterplotLayer',
                        data=pd.DataFrame({'lat': [lat], 'lon': [lon]}),
                        get_position='[lon, lat]',
                        get_fill_color=[255, 255, 255, 255], 
                        get_radius=500,  
                        stroked=True,
                        get_line_color=[255, 0, 0, 255],
                        line_width_min_pixels=3
                    )
                ]
            ))
        else:
            st.warning("No historical data found in this immediate area to generate a heatmap.")

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