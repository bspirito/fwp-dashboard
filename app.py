import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from scipy import stats
except ImportError:
    st.error("Missing dependency: scipy. Please run 'pip install scipy'")
import os
import datetime
import time
import re
import numpy as np

# Set page config
st.set_page_config(page_title="FWP Water Quality Intelligence", layout="wide")

# Constants
DATA_URL = "https://docs.google.com/spreadsheets/d/1cke90cT3WiPNHHMaR2HzEfg28Af0uXeo-VUPLe-0jmU/export?format=csv&gid=0"
LIMITS_URL = "https://docs.google.com/spreadsheets/d/1cke90cT3WiPNHHMaR2HzEfg28Af0uXeo-VUPLe-0jmU/export?format=csv&gid=350045879"
# Path to the weather data file
WEATHER_FILE = os.path.join(os.path.dirname(__file__), "winchester_weather_history.csv")

# --- Hardcoded Fallbacks ---
MASTER_RANGES = [
    {'Parameter': 'E. Coli', 'Min': 0.0, 'Max': 235.0, 'Unit': 'CFU/100mL', 'Standard': 'EPA Recreational'},
    {'Parameter': 'Fecal Coliform', 'Min': 0.0, 'Max': 200.0, 'Unit': 'cfu/100mL', 'Standard': 'MassDEP Class B'},
    {'Parameter': 'Total Phosphorus', 'Min': 0.0, 'Max': 50.0, 'Unit': 'ug/L', 'Standard': 'Eutrophic threshold'},
    {'Parameter': 'Total Nitrogen', 'Min': 0.0, 'Max': 360.0, 'Unit': 'ug/L', 'Standard': '0.36 mg/L limit'}
]

@st.cache_data(ttl=600)
def load_limits():
    try:
        df_l = pd.read_csv(LIMITS_URL)
        df_l.columns = [c.strip() for c in df_l.columns]
        return df_l
    except Exception as e:
        st.error(f"Could not load limits from Google Sheet: {e}")
        return pd.DataFrame(MASTER_RANGES)

@st.cache_data(ttl=600)
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=['Coordinates', 'Result', 'Sample Date', 'Parameter'])
        df['Unit'] = df['Unit'].fillna('N/A')

        def parse_coords(c):
            try:
                parts = str(c).split(',')
                return float(parts[0].strip()), float(parts[1].strip())
            except: return None, None

        df['lat'], df['lon'] = zip(*df['Coordinates'].apply(parse_coords))
        df = df.dropna(subset=['lat', 'lon'])

        def clean_result(val):
            if pd.isna(val) or val == '': return 0.0
            s = str(val).strip().replace(',', '').upper()
            if 'ND' in s or 'NON-DETECT' in s: return 0.0
            s_clean = re.sub(r'[^\d\.-]', '', s)
            try:
                res = float(s_clean)
                return res if np.isfinite(res) else 0.0
            except: return 0.0

        df['Result_Clean'] = df['Result'].apply(clean_result)
        
        # --- NEW: Unit Normalization ---
        # If unit is mg/L but limit is typically in ug/L (N and P variants), normalize to ug/L
        ug_params = ['Total Phosphorus', 'Nitrate Nitrogen', 'Ammonia Nitrogen', 'Total Nitrogen', 'Nitrite Nitrogen', 'Total Kjeldahl Nitrogen']
        mask_mgl = (df['Unit'].str.contains('mg/L', case=False, na=False)) & (df['Parameter'].isin(ug_params))
        df.loc[mask_mgl, 'Result_Clean'] = df.loc[mask_mgl, 'Result_Clean'] * 1000
        # ---
        df['Date'] = pd.to_datetime(df['Sample Date'], errors='coerce')
        mask = df['Date'].isna()
        df.loc[mask, 'Date'] = pd.to_datetime(df.loc[mask, 'Sample Date'].astype(str), format='%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        return df.sort_values('Date'), datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        return pd.DataFrame(), f"Error: {e}"

@st.cache_data
def load_weather():
    if os.path.exists(WEATHER_FILE):
        w_df = pd.read_csv(WEATHER_FILE)
        w_df['Date'] = pd.to_datetime(w_df['Date'])
        return w_df
    return pd.DataFrame()

# --- Data Loading ---
df, last_updated = load_data()
weather_df = load_weather()

if df.empty:
    st.error(f"Failed to load data. {last_updated}")
    st.stop()

all_params = sorted(df['Parameter'].unique().tolist())

if 'ref_df' not in st.session_state:
    st.session_state.ref_df = load_limits()

TOXIC_SPECIES = ['Cyanobacteria (Blue/Green) Dolichospermum', 'Cyanobacteria (Blue/Green) Microcystis', 'Cyanobacteria (Blue/Green) Raphidiopsis']

# --- HEADER: GLOBAL CONTROLS ---
st.title("🛡️ Winter Pond Environmental Intelligence")

col_p, col_f, col_r = st.columns([2, 2, 1])

with col_p:
    active_param = st.selectbox("Select Parameter", all_params, index=all_params.index("Total Phosphorus") if "Total Phosphorus" in all_params else 0)

with col_f:
    time_filter = st.selectbox("Data Filter", ["Most Recent", "Specific Date", "Custom Range", "All Data"])

# Dynamic date logic
active_param_df = df[df['Parameter'] == active_param]
active_unique_dates = sorted(active_param_df['Date_Str'].unique().tolist(), reverse=True)

active_filtered_df = pd.DataFrame()
display_date = ""

if time_filter == "Most Recent":
    latest_date = active_unique_dates[0] if active_unique_dates else "N/A"
    active_filtered_df = active_param_df[active_param_df['Date_Str'] == latest_date]
    display_date = f"Most Recent: {latest_date}"
elif time_filter == "Specific Date":
    selected_date = st.selectbox("Select Month Day Year", active_unique_dates)
    active_filtered_df = active_param_df[active_param_df['Date_Str'] == selected_date]
    display_date = selected_date
elif time_filter == "Custom Range":
    dr = st.date_input("Select Range", [active_param_df['Date'].min(), active_param_df['Date'].max()])
    if len(dr) == 2:
        active_filtered_df = active_param_df[(active_param_df['Date'] >= pd.to_datetime(dr[0])) & (active_param_df['Date'] <= pd.to_datetime(dr[1]))]
        display_date = f"{dr[0]} to {dr[1]}"
else:
    active_filtered_df = active_param_df.copy()
    display_date = "All Recorded Data"

with col_r:
    if st.button("Refresh Live Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption(f"Sync: {last_updated}")

# --- TABS ---
tab_report, tab_map, tab_trend, tab_corr, tab_algae, tab_ref = st.tabs([
    "🏆 Health Report Card", "🗺️ Interactive Maps", "📈 Results Trend", "🔗 Correlation", "🌿 Algae community", "📋 References"
])

# --- TAB: REPORT CARD ---
with tab_report:
    st.header("🛡️ Watershed Health Analysis")
    
    # Year Selector for Report Card
    report_years = sorted(df['Year'].unique().tolist(), reverse=True)
    sel_report_year = st.selectbox("Select Report Year", report_years)
    
    st.subheader(f"📊 {sel_report_year} Report Card")
    report_df = df[df['Year'] == sel_report_year]
    
    m1, m2, m3, m4 = st.columns(4)
    m5, m6, m7, m8 = st.columns(4)

    def calculate_score(df, params, limit, higher_is_better=False):
        sub = df[df['Parameter'].isin(params)]
        if sub.empty: return None
        if higher_is_better:
            pass_rate = (sub['Result_Clean'] >= limit).mean() * 100
        else:
            pass_rate = (sub['Result_Clean'] <= limit).mean() * 100
        return pass_rate

    def get_letter(score):
        if score is None: return "N/A", "gray"
        if score >= 90: return "A", "green"
        if score >= 80: return "B", "blue"
        if score >= 70: return "C", "orange"
        if score >= 60: return "D", "red"
        return "F", "darkred"

    def get_limit(p_name, default_v):
        row = st.session_state.ref_df[st.session_state.ref_df['Parameter'] == p_name]
        return row['Max'].iloc[0] if not row.empty else default_v

    def calculate_score_dynamic(df, param_map, higher_is_better=False):
        # param_map is {param_name: default_limit}
        found_data = []
        for p, default_limit in param_map.items():
            sub = df[df['Parameter'] == p]
            if not sub.empty:
                limit = get_limit(p, default_limit)
                if higher_is_better:
                    score = (sub['Result_Clean'] >= limit).mean() * 100
                else:
                    score = (sub['Result_Clean'] <= limit).mean() * 100
                found_data.append((score, len(sub)))
        
        if not found_data: return None, 0
        
        # Weighted average based on sample count
        total_samples = sum(s[1] for s in found_data)
        avg_score = sum(s[0] * s[1] for s in found_data) / total_samples
        return avg_score, total_samples

    # 1. Bacteria (Contact Safety)
    bac_map = {'E. Coli': 235, 'E. Coil': 235, 'Fecal Coliform': 200, 'Total Coliform': 10000}
    bac_score, bac_count = calculate_score_dynamic(report_df, bac_map)
    g, c = get_letter(bac_score)
    m1.metric("Contact Safety", g, f"{bac_count} samples" if bac_count else "No Data")
    
    # 2. Nutrients (Trophic State)
    nut_map = {'Total Phosphorus': 50, 'Total Nitrogen': 360, 'Nitrate Nitrogen': 120000, 'Ammonia Nitrogen': 500}
    nut_score, nut_count = calculate_score_dynamic(report_df, nut_map)
    g, c = get_letter(nut_score)
    m2.metric("Nutrient Levels", g, f"{nut_count} samples" if nut_count else "No Data")

    # 3. Clarity (Aesthetics/Safety)
    clar_map = {'Water Clarity Secchi Disc': 1.22, 'Secchi Depth': 1.22, 'Secchi': 1.22}
    clar_score, clar_count = calculate_score_dynamic(report_df, clar_map, True)
    g, c = get_letter(clar_score)
    m3.metric("Water Clarity", g, f"{clar_count} samples" if clar_count else "No Data")

    # 4. Bloom Risk
    bloom_map = {'Cyanobacteria (Blue/Green Algae)': 70000}
    bloom_score, bloom_count = calculate_score_dynamic(report_df, bloom_map)
    g, c = get_letter(bloom_score)
    m4.metric("Bloom Safety", g, f"{bloom_count} samples" if bloom_count else "No Data")

    # 5. Dissolved Oxygen
    do_map = {'Disolved Oxygen': 5.0, 'Surface Disolved Oxygen': 5.0, 'Surface Dissolved Oxygen': 5.0}
    do_score, do_count = calculate_score_dynamic(report_df, do_map, True)
    g, c = get_letter(do_score)
    m5.metric("Oxygen Support", g, f"{do_count} samples" if do_count else "No Data")

    # 6. Algae Density
    chl_map = {'Total Chlorophyll a': 50, 'Chlorophyll a': 50, 'Chlorophyll A': 50, 'Total Chlorophyll': 50}
    chl_score, chl_count = calculate_score_dynamic(report_df, chl_map)
    g, c = get_letter(chl_score)
    m6.metric("Algae Density", g, f"{chl_count} samples" if chl_count else "No Data")

    # 7. Chemical Stability
    ph_map = {'pH': 8.5} # Logic for range handled below
    ph_sub = report_df[report_df['Parameter'] == 'pH']
    ph_score, ph_count = None, 0
    if not ph_sub.empty:
        ph_count = len(ph_sub)
        ph_score = ((ph_sub['Result_Clean'] >= 6.5) & (ph_sub['Result_Clean'] <= 8.5)).mean() * 100
    g, c = get_letter(ph_score)
    m7.metric("pH Stability", g, f"{ph_count} samples" if ph_count else "No Data")

    # 8. Overall Health Index
    all_scores = [s for s in [bac_score, nut_score, clar_score, bloom_score, do_score, chl_score, ph_score] if s is not None]
    overall = np.mean(all_scores) if all_scores else None
    g, c = get_letter(overall)
    m8.metric("Overall Pond Grade", g, f"Index: {overall:.1f}" if overall else "No Data")

    st.divider()
    
    st.subheader("📈 Health Trends Through the Years")
    st.write("How has the overall Health Index changed since 1989?")
    
    # Historical Grade Logic
    yearly_grades = []
    for y in sorted(df['Year'].unique()):
        ydf = df[df['Year'] == y]
        y_scores = [
            calculate_score(ydf, ['E. Coli', 'E. Coil', 'Fecal Coliform'], 235),
            calculate_score(ydf, ['Total Phosphorus'], 50),
            calculate_score(ydf, ['Water Clarity Secchi Disc', 'Secchi Depth', 'Secchi'], 1.22, True),
            calculate_score(ydf, ['Cyanobacteria (Blue/Green Algae)'], 70000),
            calculate_score(ydf, ['Disolved Oxygen'], 5.0, True),
            calculate_score(ydf, ['Total Chlorophyll a', 'Chlorophyll a'], 50)
        ]
        valid = [s for s in y_scores if s is not None]
        if valid:
            yearly_grades.append({'Year': y, 'Health Index': np.mean(valid)})
    
    hist_grade_df = pd.DataFrame(yearly_grades)
    fig_hist_grade = px.line(hist_grade_df, x='Year', y='Health Index', markers=True, title="Winter Pond Health Index Trend")
    fig_hist_grade.add_hrect(y0=90, y1=100, fillcolor="green", opacity=0.1, annotation_text="Excellent (A)")
    fig_hist_grade.add_hrect(y0=70, y1=90, fillcolor="orange", opacity=0.1, annotation_text="Concern (C/B)")
    fig_hist_grade.add_hrect(y0=0, y1=70, fillcolor="red", opacity=0.1, annotation_text="Poor (F/D)")
    st.plotly_chart(fig_hist_grade, use_container_width=True)

    # NEW: Grade Key and Descriptions
    st.subheader("📋 Understanding the Health Index")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        st.markdown("""
        ### **The Grade Key**
        *   🟢 **90 - 100 (Excellent/A):** Pristine conditions. All parameters (Bacteria, Nutrients, Clarity) are well within safe Massachusetts standards.
        *   🔵 **80 - 89 (Good/B):** Healthy ecosystem with minor, infrequent exceedances. Fully safe for recreational use.
        *   🟠 **70 - 79 (Fair/C):** Ecological concern. High nutrients or low clarity are frequently detected. Use caution during summer peaks.
        *   🔴 **Below 70 (Poor/D-F):** Impaired state. Significant bacteria spikes or toxic bloom conditions are common. 
        """)
        
    with d_col2:
        st.markdown("""
        ### **How it's aggregated**
        The index is a **weighted composite score**. It doesn't just look at one parameter, it balances:
        1.  **Public Health** (E. Coli & Algae Toxins)
        2.  **Ecosystem Support** (Oxygen & pH)
        3.  **Visual Aesthetics** (Clarity & Chlorophyll)
        
        *Note: If a year shows 'N/A', it means fewer than 3 key categories were sampled during that period.*
        """)

    with st.expander("📖 Description of New Grading Elements"):
        st.markdown("""
        *   **Oxygen Support**: Measures if Dissolved Oxygen is above **5.0 mg/L**. Oxygen is critical for fish and healthy pond bacteria.
        *   **Algae Density**: Uses Chlorophyll a. High levels (>50 µg/L) indicate excessive algae biomass, even if toxic species aren't present.
        *   **pH Stability**: Tracks if the pond stays between **6.5 and 8.5**. Values outside this range can be stressful for aquatic life.
        *   **Health Index**: A weighted average of all available metrics for that year.
        """)

    st.divider()
    
    # Seasonal Pond Dynamics (Previous Logic)
    st.subheader("🗓️ Typical Seasonal Stress Profile")
    stress_params = ['Total Phosphorus', 'Total Nitrogen', 'E. Coil', 'Cyanobacteria (Blue/Green Algae)']
    stress_ref = st.session_state.ref_df[st.session_state.ref_df['Parameter'].isin(stress_params)]
    if not stress_ref.empty:
        temp_stress_df = df[df['Parameter'].isin(stress_params)].copy()
        def get_internal_name(p): return p
        temp_stress_df['Ref_Parameter'] = temp_stress_df['Parameter']
        stress_df = temp_stress_df.merge(st.session_state.ref_df[['Parameter', 'Max']].rename(columns={'Parameter': 'Ref_Parameter'}), on='Ref_Parameter')
        stress_df['Norm_Score'] = stress_df['Result_Clean'] / stress_df['Max']
        seasonal_df = stress_df.groupby('Month')['Norm_Score'].mean().reset_index()
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        seasonal_df['Month'] = seasonal_df['Month'].map(month_map)
        fig_seasonal = px.line(seasonal_df, x='Month', y='Norm_Score', title="Seasonal Stress Profile", markers=True)
        fig_seasonal.update_traces(line_color='teal', fill='tozeroy')
        st.plotly_chart(fig_seasonal, use_container_width=True)

# --- TAB: MAPS ---
with tab_map:
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1: v_mode = st.radio("Map Engine", ["2D", "3D"], horizontal=True)
    with m_col2:
        map_styles = {"Road": "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png", "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Dark": "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"}
        sel_style = st.selectbox("Background", list(map_styles.keys()))
    with m_col3: m_scale = st.slider("Scale", 1, 100, 30)

    if not active_filtered_df.empty:
        center = [active_filtered_df['lat'].mean(), active_filtered_df['lon'].mean()]
        if v_mode == "2D":
            m = folium.Map(location=center, zoom_start=17, tiles=None)
            folium.TileLayer(tiles=map_styles[sel_style], attr="Tiles", name=sel_style).add_to(m)
            max_v = active_param_df['Result_Clean'].max() or 1
            for _, row in active_filtered_df.iterrows():
                norm = max(0, min(1, row['Result_Clean']/max_v))
                color = '#%02x%02x%02x' % (int(norm * 255), 0, int((1-norm) * 255))
                folium.CircleMarker(location=[row['lat'], row['lon']], radius=5 + (norm * m_scale), popup=f"{row['Location']}: {row['Result_Clean']}", color=color, fill=True).add_to(m)
            st_folium(m, height=600, use_container_width=True, key=f"folium-{active_param}-{sel_style}")
        else:
            max_v = active_param_df['Result_Clean'].max() or 1
            active_filtered_df['color'] = active_filtered_df['Result_Clean'].apply(lambda x: [int((x/max_v)*255), 0, int((1-(x/max_v))*255), 160])
            
            # Column Layer
            column_layer = pdk.Layer(
                "ColumnLayer", active_filtered_df, get_position=["lon", "lat"],
                get_elevation="Result_Clean", elevation_scale=(m_scale * 10.0) / max_v,
                radius=20, get_fill_color="color", pickable=True, auto_highlight=True
            )
            
            # NEW: Text Layer for labels
            text_layer = pdk.Layer(
                "TextLayer",
                active_filtered_df,
                get_position=["lon", "lat"],
                get_text="Location",
                get_size=16,
                get_color=[255, 255, 255] if sel_style == "Dark" else [0, 0, 0],
                get_alignment_baseline="'bottom'",
            )

            st.pydeck_chart(pdk.Deck(
                layers=[column_layer, text_layer],
                initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=16, pitch=45),
                tooltip={"html": "<b>{Location}</b><br>Result: {Result_Clean} " + (active_param_df['Unit'].iloc[0] if not active_param_df.empty else "")},
                map_provider="carto",
                map_style="light" if sel_style == "Road" else "dark"
            ), key=f"deck-{active_param}-{sel_style}")

# --- TAB: TREND ---
with tab_trend:
    st.header(f"{active_param} Trends with Weather Context")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    show_temp = w_col1.checkbox("🌡️ Show Temperature", value=False)
    show_rain = w_col2.checkbox("🌧️ Show Rainfall (Bar)", value=False)
    show_rain_ma = w_col3.checkbox("📈 Show Rain Moving Avg", value=False)
    temp_unit = w_col4.radio("Temp Unit", ["Fahrenheit", "Celsius"], horizontal=True)

    if not active_param_df.empty:
        plot_df = active_param_df.sort_values('Date')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for loc in plot_df['Location'].unique():
            loc_df = plot_df[plot_df['Location'] == loc]
            fig.add_trace(go.Scatter(x=loc_df['Date'], y=loc_df['Result_Clean'], mode='lines+markers', name=f"{loc}"), secondary_y=False)
        
        # --- NEW: Weather Context Logic ---
        if (show_temp or show_rain or show_rain_ma) and not weather_df.empty:
            w_start, w_end = plot_df['Date'].min(), plot_df['Date'].max()
            w_plot_df = weather_df[(weather_df['Date'] >= w_start) & (weather_df['Date'] <= w_end)].copy()
            
            if not w_plot_df.empty:
                if show_temp:
                    t_col = 'tavg_f' if temp_unit == "Fahrenheit" else 'Temp_C'
                    fig.add_trace(go.Scatter(x=w_plot_df['Date'], y=w_plot_df[t_col], name=f"Temp ({temp_unit[0]})", 
                                             line=dict(color='orange', width=1, dash='dot'), opacity=0.5), secondary_y=True)
                
                if show_rain:
                    fig.add_trace(go.Bar(x=w_plot_df['Date'], y=w_plot_df['prcp_in'], name="Rain (in)", 
                                         marker_color='lightblue', opacity=0.4), secondary_y=True)
                
                if show_rain_ma:
                    w_plot_df['rain_ma'] = w_plot_df['prcp_in'].rolling(window=7).mean()
                    fig.add_trace(go.Scatter(x=w_plot_df['Date'], y=w_plot_df['rain_ma'], name="Rain 7d Avg", 
                                             line=dict(color='blue', width=2), opacity=0.6), secondary_y=True)

        # --- Universal Threshold Highlighting ---
        ref_active = st.session_state.ref_df[st.session_state.ref_df['Parameter'] == active_param]
        
        if not ref_active.empty:
            limit = ref_active['Max'].iloc[0]
            y_max = max(limit * 1.5, plot_df['Result_Clean'].max() * 1.2, 1.0)
            
            # Special Case: Oxygen and Secchi (Higher is BETTER)
            if 'Oxygen' in active_param or 'Secchi' in active_param:
                # Green above limit, Red below
                fig.add_hrect(y0=limit, y1=y_max, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Healthy Zone", annotation_position="top left", layer="below")
                fig.add_hrect(y0=0, y1=limit, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Impaired Zone", annotation_position="bottom left", layer="below")
                fig.add_hline(y=limit, line_dash="dash", line_color="black", annotation_text=f"Standard: {limit}")
            
            # Special Case: pH (Safe Range)
            elif active_param == 'pH':
                fig.add_hrect(y0=6.5, y1=8.5, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Optimal pH", layer="below")
                fig.add_hrect(y0=0, y1=6.5, fillcolor="red", opacity=0.1, line_width=0, layer="below")
                fig.add_hrect(y0=8.5, y1=14, fillcolor="red", opacity=0.1, line_width=0, layer="below")
            
            # Standard Case: Nutrients/Bacteria (Lower is BETTER)
            else:
                # Green below limit, Red above
                fig.add_hrect(y0=0, y1=limit, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Safe Zone", annotation_position="bottom left", layer="below")
                fig.add_hrect(y0=limit, y1=y_max, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Exceedance Zone", annotation_position="top left", layer="below")
                fig.add_hline(y=limit, line_dash="dash", line_color="red", annotation_text=f"Limit: {limit}")

        # Determine Secondary Y-Axis Title
        y2_title = ""
        parts = []
        if show_temp: parts.append(f"Temp ({temp_unit[0]})")
        if show_rain: parts.append("Rain (in)")
        y2_title = " & ".join(parts)

        fig.update_layout(
            title=f"{active_param} Over Time",
            xaxis_title="Date",
            yaxis_title=f"Result ({active_param_df['Unit'].iloc[0]})",
            yaxis2_title=y2_title,
            hovermode="x unified",
            height=600
        )

        # Invert Y-axis for Secchi Depth only
        if 'Secchi' in active_param:
            fig.update_layout(yaxis=dict(autorange='reversed'))

        st.plotly_chart(fig, use_container_width=True)

        if not active_filtered_df.empty:
            st.divider()
            st.subheader(f"Local Comparison: {display_date}")
            compare_df = active_filtered_df.groupby('Location', as_index=False).agg({'Result_Clean': 'mean'})
            bar_fig = px.bar(compare_df, x='Location', y='Result_Clean', color='Result_Clean', text_auto='.2f', title=f"Site Performance ({active_param})", color_continuous_scale='RdYlGn_r' if not ref_active.empty else 'Viridis')
            if not ref_active.empty: bar_fig.add_hline(y=ref_active['Max'].iloc[0], line_dash="dash", line_color="red")
            st.plotly_chart(bar_fig, use_container_width=True)
            st.subheader("Statistical Summary")
            st.table(active_param_df.groupby('Location')['Result_Clean'].agg(['count', 'min', 'max', 'mean']).round(2))

# --- TAB: CORRELATION ---
with tab_corr:
    st.header("Relationship Analysis")
    c1, c2 = st.columns(2)
    with c1: p1 = st.selectbox("X-Axis", all_params, index=all_params.index("Temperature") if "Temperature" in all_params else 0)
    with c2: p2 = st.selectbox("Y-Axis", all_params, index=all_params.index("Total Phosphorus") if "Total Phosphorus" in all_params else 1)
    corr_data = df[df['Parameter'].isin([p1, p2])].pivot_table(index=['Date', 'Location'], columns='Parameter', values='Result_Clean').dropna()
    if not corr_data.empty:
        st.plotly_chart(px.scatter(corr_data, x=p1, y=p2, trendline="ols"), use_container_width=True)
        st.info(f"Correlation (R): {corr_data[p1].corr(corr_data[p2]):.2f}")
    else: st.info("No overlapping data.")

# --- TAB: ALGAE ---
with tab_algae:
    st.header("Algae Community")
    algae_all = df[df['Parameter'].str.contains('Cyanobacteria|Chlorophyta|Algae|Diatoms', case=False)]
    if not algae_all.empty:
        latest_year = df['Year'].max()
        st.plotly_chart(px.sunburst(algae_all[algae_all['Year']==latest_year], path=['Parameter'], values='Result_Clean', title="Current Composition"), use_container_width=True)
        toxic_df = algae_all[algae_all['Parameter'].isin(TOXIC_SPECIES)]
        if not toxic_df.empty:
            st.plotly_chart(px.bar(toxic_df, x='Date', y='Result_Clean', color='Parameter', title="Toxic Watch"), use_container_width=True)

# --- TAB: REF ---
with tab_ref:
    st.header("Standards Editor")
    col_sync1, col_sync2 = st.columns(2)
    if col_sync1.button("Sync Limits from Google Sheet", use_container_width=True):
        st.session_state.ref_df = load_limits()
        st.rerun()
    if col_sync2.button("Restore Winchester Defaults", use_container_width=True):
        st.session_state.ref_df = pd.DataFrame(MASTER_RANGES)
        st.rerun()
    st.data_editor(st.session_state.ref_df, num_rows="dynamic", use_container_width=True)
