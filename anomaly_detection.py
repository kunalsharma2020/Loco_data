"""
Anomaly detection module: Multi-layer anomaly detection on sensor data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from config import (
    FEATURES_DIR, ANOMALIES_DIR, THRESHOLDS, 
    MAD_THRESHOLD, ISOLATION_FOREST_CONTAMINATION, ANOMALY_FEATURES
)


def calculate_mad(series):
    """Calculate Median Absolute Deviation"""
    median = series.median()
    mad = np.abs(series - median).median()
    return median, mad


def detect_anomalies(features_path=None, output_path=None):
    """
    Multi-layer anomaly detection on aggregated features
    
    Parameters:
    -----------
    features_path : str, optional
        Path to input features Parquet file
    output_path : str, optional
        Path for output anomalies Parquet file
    """
    if features_path is None:
        features_path = str(FEATURES_DIR / "features_1min.parquet")
    
    if output_path is None:
        output_path = str(ANOMALIES_DIR / "anomalies.parquet")
    
    print(f"\n{'='*60}")
    print(f"Running anomaly detection...")
    print(f"{'='*60}")
    print(f"📂 Input:  {features_path}")
    print(f"📂 Output: {output_path}")
    
    # Load features
    print("\n⚙️  Loading features...")
    df = pd.read_parquet(features_path)
    print(f"   Loaded {len(df):,} rows, {len(df['locoid'].unique())} locomotives")
    
    # Initialize anomaly columns
    df['anomaly_rule'] = 0
    df['anomaly_mad'] = 0
    df['anomaly_ml'] = 0
    df['anomaly_types'] = ''
    
    # ========================================================================
    # LAYER 1: Rule-based anomaly detection
    # ========================================================================
    print("\n⚙️  Layer 1: Rule-based detection...")
    
    anomaly_flags = []
    
    # 1. High temperature
    temp_high = df['temp_motor1_1_max'] > THRESHOLDS['temp_motor_max']
    df.loc[temp_high, 'anomaly_rule'] = 1
    df.loc[temp_high, 'anomaly_types'] += 'HIGH_TEMP;'
    anomaly_flags.append(('High Temperature', temp_high.sum()))
    
    # 2. Rapid temperature change
    temp_rate_high = df['temp_motor1_1_rate'].abs() > THRESHOLDS['temp_motor_rate_max']
    df.loc[temp_rate_high, 'anomaly_rule'] = 1
    df.loc[temp_rate_high, 'anomaly_types'] += 'TEMP_SPIKE;'
    anomaly_flags.append(('Temperature Spike', temp_rate_high.sum()))
    
    # 3. High current variability
    current_unstable = df['current_u_std'] > THRESHOLDS['current_std_max']
    df.loc[current_unstable, 'anomaly_rule'] = 1
    df.loc[current_unstable, 'anomaly_types'] += 'CURRENT_UNSTABLE;'
    anomaly_flags.append(('Current Instability', current_unstable.sum()))
    
    # 4. Low battery
    battery_low = df['battery_volt_min'] < THRESHOLDS['battery_min']
    df.loc[battery_low, 'anomaly_rule'] = 1
    df.loc[battery_low, 'anomaly_types'] += 'LOW_BATTERY;'
    anomaly_flags.append(('Low Battery', battery_low.sum()))
    
    # 5. Sudden speed jump
    df['speed_change'] = df.groupby('locoid')['avg_speed'].diff().abs()
    speed_jump = df['speed_change'] > THRESHOLDS['speed_jump_max']
    df.loc[speed_jump, 'anomaly_rule'] = 1
    df.loc[speed_jump, 'anomaly_types'] += 'SPEED_JUMP;'
    anomaly_flags.append(('Speed Jump', speed_jump.sum()))
    
    # 6. Pressure anomalies
    pressure_low = df['pressure_tr1_min'] < THRESHOLDS['pressure_min']
    df.loc[pressure_low, 'anomaly_rule'] = 1
    df.loc[pressure_low, 'anomaly_types'] += 'LOW_PRESSURE;'
    anomaly_flags.append(('Low Pressure', pressure_low.sum()))
    
    print("   Rule-based anomalies detected:")
    for name, count in anomaly_flags:
        print(f"     • {name}: {count:,}")
    
    # ========================================================================
    # LAYER 2: Statistical anomaly detection (MAD)
    # ========================================================================
    print("\n⚙️  Layer 2: Statistical detection (MAD)...")
    
    mad_features = ['temp_motor1_1_mean', 'current_u_mean', 'pressure_tr1_mean']
    mad_anomalies = 0
    
    for loco in df['locoid'].unique():
        loco_mask = df['locoid'] == loco
        loco_data = df.loc[loco_mask]
        
        for feature in mad_features:
            if feature in df.columns and loco_data[feature].notna().sum() > 10:
                median, mad = calculate_mad(loco_data[feature].dropna())
                
                if mad > 0:
                    z_score = np.abs((loco_data[feature] - median) / (1.4826 * mad))
                    anomaly_mask = z_score > MAD_THRESHOLD
                    
                    df.loc[loco_mask & anomaly_mask, 'anomaly_mad'] = 1
                    df.loc[loco_mask & anomaly_mask, 'anomaly_types'] += f'MAD_{feature.upper()};'
                    mad_anomalies += anomaly_mask.sum()
    
    print(f"   MAD anomalies detected: {mad_anomalies:,}")
    
    # ========================================================================
    # LAYER 3: ML-based anomaly detection (Isolation Forest)
    # ========================================================================
    print("\n⚙️  Layer 3: ML-based detection (Isolation Forest)...")
    
    # Select features for ML
    ml_features = [f for f in ANOMALY_FEATURES if f in df.columns]
    print(f"   Using {len(ml_features)} features for ML detection")
    
    ml_anomalies = 0
    
    for loco in df['locoid'].unique():
        loco_mask = df['locoid'] == loco
        loco_data = df.loc[loco_mask, ml_features].copy()
        
        # Only analyze when locomotive is moving
        moving_mask = df.loc[loco_mask, 'pct_moving'] > 0.5
        loco_data_moving = loco_data[moving_mask]
        
        if len(loco_data_moving) > 50:  # Need sufficient data
            # Handle missing values
            loco_data_moving = loco_data_moving.fillna(loco_data_moving.median())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(loco_data_moving)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=ISOLATION_FOREST_CONTAMINATION,
                random_state=42,
                n_jobs=-1
            )
            predictions = iso_forest.fit_predict(X_scaled)
            
            # Mark anomalies (-1 from Isolation Forest)
            anomaly_indices = loco_data_moving.index[predictions == -1]
            df.loc[anomaly_indices, 'anomaly_ml'] = 1
            df.loc[anomaly_indices, 'anomaly_types'] += 'ML_ISOLATION;'
            ml_anomalies += len(anomaly_indices)
    
    print(f"   ML anomalies detected: {ml_anomalies:,}")
    
    # ========================================================================
    # Combine anomaly scores
    # ========================================================================
    print("\n⚙️  Combining anomaly scores...")
    
    df['anomaly_score'] = (
        df['anomaly_rule'] * 3 +      # Rule-based: weight 3
        df['anomaly_mad'] * 2 +        # Statistical: weight 2
        df['anomaly_ml'] * 1           # ML: weight 1
    )
    
    df['is_anomaly'] = (df['anomaly_score'] >= 2).astype(int)
    
    # Clean up anomaly_types
    df['anomaly_types'] = df['anomaly_types'].str.rstrip(';')
    
    # ========================================================================
    # Save results
    # ========================================================================
    print(f"\n💾 Saving anomalies to {output_path}...")
    df.to_parquet(output_path, compression='zstd', index=False)
    
    # Summary statistics
    total_anomalies = df['is_anomaly'].sum()
    anomaly_pct = (total_anomalies / len(df)) * 100
    
    print(f"\n{'='*60}")
    print(f"✓ Anomaly detection complete!")
    print(f"{'='*60}")
    print(f"  Total records:        {len(df):,}")
    print(f"  Anomalies detected:   {total_anomalies:,} ({anomaly_pct:.2f}%)")
    print(f"  Rule-based:           {df['anomaly_rule'].sum():,}")
    print(f"  Statistical (MAD):    {df['anomaly_mad'].sum():,}")
    print(f"  ML (Isolation):       {df['anomaly_ml'].sum():,}")
    print(f"{'='*60}\n")
    
    return df


if __name__ == "__main__":
    detect_anomalies()
