"""
Configuration file for locomotive sensor data analysis pipeline
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
INPUT_DIR = Path(r"H:\april_data")
PARQUET_DIR = Path(r"H:\april_parquet")
FEATURES_DIR = Path(r"H:\april_features")
ANOMALIES_DIR = Path(r"H:\april_anomalies")

# Create directories if they don't exist
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
ANOMALIES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================
CHUNK_SIZE = 1_000_000  # Rows per chunk for CSV reading
AGGREGATION_INTERVAL = "1min"  # Time interval for feature aggregation

# ============================================================================
# SENSOR THRESHOLDS (for rule-based anomaly detection)
# ============================================================================
THRESHOLDS = {
    # Temperature thresholds (°C)
    'temp_motor_max': 120,
    'temp_motor_rate_max': 5,  # °C per minute
    
    # Current thresholds (A)
    'current_std_max': 50,
    
    # Battery thresholds (V)
    'battery_min': 90,
    
    # Speed thresholds (km/h)
    'speed_jump_max': 20,  # km/h per minute
    
    # Pressure thresholds (bar)
    'pressure_min': 4,
    'pressure_max': 10,
}

# ============================================================================
# ANOMALY DETECTION PARAMETERS
# ============================================================================
MAD_THRESHOLD = 3.5  # Median Absolute Deviation threshold
ISOLATION_FOREST_CONTAMINATION = 0.01  # Expected proportion of anomalies

# ============================================================================
# DTYPE MAPPING FOR EFFICIENT MEMORY USAGE
# ============================================================================
dtype_dict = {
    # GPS & identifiers
    'latitude': 'float32',
    'longitude': 'float32',
    'altitude': 'float32',
    'gpsspeed': 'float32',
    'devicetime': 'str',
    'locoid': 'int32',
    'faultnum': 'int16',
    
    # Temperature sensors (°C)
    'xtempmotor1_1': 'float32',
    'xtempmotor1_2': 'float32',
    'xtempmotor2_1': 'float32',
    'xtempmotor2_2': 'float32',
    'xtempmotor3_1': 'float32',
    'xtempmotor3_2': 'float32',
    'xatmp1oeltr_1': 'float32',
    'xatmp1oeltr_2': 'float32',
    'xatmp2oeltr_1': 'float32',
    'xatmp2oeltr_2': 'float32',
    'xatmp1oelsr_1': 'float32',
    'xatmp1oelsr_2': 'float32',
    'xatmp2oelsr_1': 'float32',
    'xatmp2oelsr_2': 'float32',
    
    # Current sensors (A)
    'xuprim_1': 'float32',
    'xiprim_1': 'float32',
    'xaibur': 'float32',
    
    # Pressure sensors (bar)
    'xadrucktr_1': 'float32',
    'xadrucktr_2': 'float32',
    'xadrucksr_1': 'float32',
    'xadrucksr_2': 'float32',
    'xprautobkln': 'float32',
    'xpressurecv_1': 'float32',
    'xpressurecv_2': 'float32',
    
    # Energy & odometer
    'xenergkwh_plus': 'float32',
    'xenergkwh_minus': 'float32',
    'odometerK': 'float32',
    'odometerM': 'float32',
    'odometerG': 'float32',
    
    # Battery & Speed
    'xu_battery': 'float32',
    'xspeedloco': 'float32',
    'xte_be_loco': 'float32',
}

# Flag columns (binary) - will be detected automatically
FLAG_DTYPE = 'int8'

# ============================================================================
# FEATURE COLUMNS FOR ANOMALY DETECTION
# ============================================================================
ANOMALY_FEATURES = [
    'temp_motor1_1_mean',
    'temp_motor1_1_max',
    'temp_motor1_1_std',
    'current_u_mean',
    'current_u_std',
    'pressure_tr1_mean',
    'pressure_tr1_std',
    'energy_consumption',
    'avg_speed',
    'battery_volt_mean',
]

# ============================================================================
# STREAMLIT DASHBOARD SETTINGS
# ============================================================================
DASHBOARD_CONFIG = {
    'page_title': 'Locomotive Sensor Analytics',
    'page_icon': '🚂',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# Map settings for India
INDIA_CENTER = [20.5937, 78.9629]
INDIA_ZOOM = 5
