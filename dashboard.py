"""
Streamlit Dashboard for Locomotive Sensor Analytics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
from config import (
    ANOMALIES_DIR, DASHBOARD_CONFIG, INDIA_CENTER, INDIA_ZOOM
)

# Page configuration
st.set_page_config(**DASHBOARD_CONFIG)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load anomaly data with caching"""
    anomalies_path = str(ANOMALIES_DIR / "anomalies.parquet")
    try:
        df = pd.read_parquet(anomalies_path)
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def main():
    # Load data
    df = load_data()
    
    if df is None:
        st.error("⚠️ No data available. Please run the pipeline first.")
        st.info("""
        **Steps to generate data:**
        1. Run `python data_ingestion.py` to convert CSV to Parquet
        2. Run `python feature_engineering.py` to create features
        3. Run `python anomaly_detection.py` to detect anomalies
        4. Refresh this dashboard
        """)
        return
    
    # ========================================================================
    # SIDEBAR - Filters
    # ========================================================================
    st.sidebar.title("🚂 Locomotive Analytics")
    st.sidebar.markdown("---")
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    else:
        df_filtered = df
    
    # Locomotive filter
    all_locos = sorted(df_filtered['locoid'].unique())
    selected_locos = st.sidebar.multiselect(
        "🚂 Select Locomotives",
        options=all_locos,
        default=all_locos[:min(5, len(all_locos))]
    )
    
    if selected_locos:
        df_filtered = df_filtered[df_filtered['locoid'].isin(selected_locos)]
    
    # Anomaly score threshold
    anomaly_threshold = st.sidebar.slider(
        "⚠️ Anomaly Score Threshold",
        min_value=0,
        max_value=int(df_filtered['anomaly_score'].max()),
        value=2
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Data Range:** {min_date} to {max_date}")
    st.sidebar.info(f"**Total Locomotives:** {len(all_locos)}")
    
    # ========================================================================
    # MAIN CONTENT - Page Navigation
    # ========================================================================
    st.title("🚂 Locomotive Sensor Analytics Dashboard")
    
    page = st.radio(
        "Navigate",
        ["📊 Overview", "📈 Time Series Analysis", "🗺️ Geographic View", 
         "⚠️ Anomaly Details", "📉 Data Quality"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ========================================================================
    # PAGE 1: Overview
    # ========================================================================
    if page == "📊 Overview":
        st.header("📊 Overview Dashboard")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🚂 Locomotives Reporting",
                len(df_filtered['locoid'].unique())
            )
        
        with col2:
            anomaly_count = (df_filtered['anomaly_score'] >= anomaly_threshold).sum()
            anomaly_pct = (anomaly_count / len(df_filtered)) * 100
            st.metric(
                "⚠️ Anomalies Detected",
                f"{anomaly_count:,}",
                f"{anomaly_pct:.1f}%"
            )
        
        with col3:
            total_faults = df_filtered['fault_count'].sum()
            st.metric(
                "🔧 Total Faults",
                f"{int(total_faults):,}"
            )
        
        with col4:
            avg_temp = df_filtered['temp_motor1_1_mean'].mean()
            st.metric(
                "🌡️ Avg Motor Temp",
                f"{avg_temp:.1f}°C"
            )
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Anomaly Trend Over Time")
            daily_anomalies = df_filtered.groupby('date').agg({
                'is_anomaly': 'sum',
                'locoid': 'count'
            }).reset_index()
            daily_anomalies['anomaly_rate'] = (daily_anomalies['is_anomaly'] / daily_anomalies['locoid']) * 100
            
            fig = px.line(
                daily_anomalies,
                x='date',
                y='anomaly_rate',
                title='Daily Anomaly Rate (%)',
                labels={'anomaly_rate': 'Anomaly Rate (%)', 'date': 'Date'}
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("🚂 Anomalies by Locomotive")
            loco_anomalies = df_filtered.groupby('locoid')['is_anomaly'].sum().reset_index()
            loco_anomalies = loco_anomalies.sort_values('is_anomaly', ascending=False).head(10)
            
            fig = px.bar(
                loco_anomalies,
                x='locoid',
                y='is_anomaly',
                title='Top 10 Locomotives by Anomaly Count',
                labels={'is_anomaly': 'Anomaly Count', 'locoid': 'Locomotive ID'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # Temperature distribution
        st.subheader("🌡️ Motor Temperature Distribution")
        fig = px.box(
            df_filtered,
            x='locoid',
            y='temp_motor1_1_mean',
            title='Motor Temperature Distribution by Locomotive',
            labels={'temp_motor1_1_mean': 'Temperature (°C)', 'locoid': 'Locomotive ID'}
        )
        st.plotly_chart(fig, width='stretch')
    
    # ========================================================================
    # PAGE 2: Time Series Analysis
    # ========================================================================
    elif page == "📈 Time Series Analysis":
        st.header("📈 Time Series Analysis")
        
        if not selected_locos:
            st.warning("Please select at least one locomotive from the sidebar.")
            return
        
        selected_loco = st.selectbox("Select Locomotive for Detailed View", selected_locos)
        loco_data = df_filtered[df_filtered['locoid'] == selected_loco].sort_values('ts')
        
        # Temperature plot
        st.subheader("🌡️ Motor Temperature Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loco_data['ts'],
            y=loco_data['temp_motor1_1_mean'],
            mode='lines',
            name='Motor 1-1 Temp',
            line=dict(color='blue')
        ))
        
        # Mark anomalies
        anomalies = loco_data[loco_data['anomaly_score'] >= anomaly_threshold]
        fig.add_trace(go.Scatter(
            x=anomalies['ts'],
            y=anomalies['temp_motor1_1_mean'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Current plot
        st.subheader("⚡ Current Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loco_data['ts'],
            y=loco_data['current_u_mean'],
            mode='lines',
            name='Current U',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=anomalies['ts'],
            y=anomalies['current_u_mean'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Current (A)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Pressure plot
        st.subheader("💨 Pressure Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loco_data['ts'],
            y=loco_data['pressure_tr1_mean'],
            mode='lines',
            name='Pressure TR1',
            line=dict(color='purple')
        ))
        
        fig.add_trace(go.Scatter(
            x=anomalies['ts'],
            y=anomalies['pressure_tr1_mean'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Pressure (bar)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    
    # ========================================================================
    # PAGE 3: Geographic View
    # ========================================================================
    elif page == "🗺️ Geographic View":
        st.header("🗺️ Geographic View - India Map")
        
        # Get latest position for each locomotive
        latest_positions = df_filtered.sort_values('ts').groupby('locoid').last().reset_index()
        latest_positions = latest_positions[
            (latest_positions['avg_lat'].notna()) & 
            (latest_positions['avg_lon'].notna())
        ]
        
        if len(latest_positions) == 0:
            st.warning("No GPS data available for selected locomotives.")
            return
        
        # Create map
        m = folium.Map(location=INDIA_CENTER, zoom_start=INDIA_ZOOM)
        
        for _, row in latest_positions.iterrows():
            # Color based on anomaly status
            color = 'red' if row['is_anomaly'] else 'green'
            
            # Popup content
            popup_html = f"""
            <b>Locomotive ID:</b> {row['locoid']}<br>
            <b>Temperature:</b> {row['temp_motor1_1_mean']:.1f}°C<br>
            <b>Speed:</b> {row['avg_speed']:.1f} km/h<br>
            <b>Anomaly Score:</b> {row['anomaly_score']}<br>
            <b>Last Update:</b> {row['ts']}
            """
            
            folium.CircleMarker(
                location=[row['avg_lat'], row['avg_lon']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        folium_static(m, width=1200, height=600)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 Green: Normal operation
        - 🔴 Red: Anomaly detected
        """)
    
    # ========================================================================
    # PAGE 4: Anomaly Details
    # ========================================================================
    elif page == "⚠️ Anomaly Details":
        st.header("⚠️ Anomaly Details")
        
        anomalies = df_filtered[df_filtered['anomaly_score'] >= anomaly_threshold]
        
        st.metric("Total Anomalies", len(anomalies))
        
        # Top anomalies table
        st.subheader("🔝 Top Anomalies")
        top_anomalies = anomalies.nlargest(20, 'anomaly_score')[[
            'ts', 'locoid', 'anomaly_score', 'anomaly_types',
            'temp_motor1_1_mean', 'current_u_mean', 'pressure_tr1_mean'
        ]]
        st.dataframe(top_anomalies, width='stretch')
        
        # Anomaly score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Anomaly Score Distribution")
            fig = px.histogram(
                anomalies,
                x='anomaly_score',
                nbins=20,
                title='Distribution of Anomaly Scores'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("🏷️ Top Anomaly Types")
            # Parse anomaly types
            all_types = []
            for types_str in anomalies['anomaly_types'].dropna():
                all_types.extend(types_str.split(';'))
            
            if all_types:
                type_counts = pd.Series(all_types).value_counts().head(10)
                fig = px.bar(
                    x=type_counts.values,
                    y=type_counts.index,
                    orientation='h',
                    title='Most Common Anomaly Types',
                    labels={'x': 'Count', 'y': 'Anomaly Type'}
                )
                st.plotly_chart(fig, width='stretch')
    
    # ========================================================================
    # PAGE 5: Data Quality
    # ========================================================================
    elif page == "📉 Data Quality":
        st.header("📉 Data Quality Metrics")
        
        # Sampling rate
        st.subheader("📊 Sampling Rate by Locomotive")
        sampling_rate = df_filtered.groupby('locoid')['sample_count'].mean().reset_index()
        sampling_rate.columns = ['locoid', 'avg_samples_per_min']
        
        fig = px.bar(
            sampling_rate,
            x='locoid',
            y='avg_samples_per_min',
            title='Average Samples per Minute',
            labels={'avg_samples_per_min': 'Samples/min', 'locoid': 'Locomotive ID'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # GPS availability
        st.subheader("🛰️ GPS Data Availability")
        gps_avail = df_filtered.groupby('locoid')['gps_availability'].mean().reset_index()
        gps_avail['gps_availability'] *= 100
        
        fig = px.bar(
            gps_avail,
            x='locoid',
            y='gps_availability',
            title='GPS Availability by Locomotive (%)',
            labels={'gps_availability': 'GPS Availability (%)', 'locoid': 'Locomotive ID'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Missing data percentage
        st.subheader("❓ Missing Data Analysis")
        missing_cols = ['temp_motor1_1_mean', 'current_u_mean', 'pressure_tr1_mean', 'battery_volt_mean']
        missing_data = []
        
        for col in missing_cols:
            if col in df_filtered.columns:
                missing_pct = (df_filtered[col].isna().sum() / len(df_filtered)) * 100
                missing_data.append({'Sensor': col, 'Missing (%)': missing_pct})
        
        missing_df = pd.DataFrame(missing_data)
        fig = px.bar(
            missing_df,
            x='Sensor',
            y='Missing (%)',
            title='Missing Data Percentage by Sensor'
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
