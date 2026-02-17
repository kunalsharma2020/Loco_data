"""
Descriptive analysis module: Generate summary reports and statistics
"""
import duckdb
import pandas as pd
from pathlib import Path
from config import FEATURES_DIR


def descriptive_analysis(features_path=None):
    """
    Generate descriptive statistics and reports from aggregated features
    
    Parameters:
    -----------
    features_path : str, optional
        Path to input features Parquet file
    """
    if features_path is None:
        features_path = str(FEATURES_DIR / "features_1min.parquet")
    
    print(f"\n{'='*60}")
    print(f"Running descriptive analysis...")
    print(f"{'='*60}")
    print(f"📂 Input: {features_path}")
    
    con = duckdb.connect(database=':memory:')
    
    # Load features
    con.execute(f"CREATE TABLE features AS SELECT * FROM read_parquet('{features_path}')")
    
    reports = {}
    
    # ========================================================================
    # 1. Data Coverage Report
    # ========================================================================
    print("\n📊 1. Data Coverage by Locomotive")
    print("-" * 60)
    
    coverage = con.execute("""
        SELECT 
            locoid,
            COUNT(*) AS total_records,
            MIN(ts) AS first_seen,
            MAX(ts) AS last_seen,
            ROUND(AVG(sample_count), 1) AS avg_samples_per_min,
            ROUND(AVG(gps_availability) * 100, 1) AS gps_availability_pct
        FROM features
        GROUP BY locoid
        ORDER BY locoid
    """).df()
    
    print(coverage.to_string(index=False))
    reports['coverage'] = coverage
    
    # ========================================================================
    # 2. Sensor Range Summary
    # ========================================================================
    print("\n📊 2. Sensor Range Summary (All Locomotives)")
    print("-" * 60)
    
    sensor_ranges = con.execute("""
        SELECT 
            'Motor Temp 1-1' AS sensor,
            ROUND(MIN(temp_motor1_1_mean), 1) AS min_val,
            ROUND(AVG(temp_motor1_1_mean), 1) AS avg_val,
            ROUND(MAX(temp_motor1_1_max), 1) AS max_val,
            '°C' AS unit
        FROM features WHERE temp_motor1_1_mean IS NOT NULL
        UNION ALL
        SELECT 
            'Motor Temp 2-1',
            ROUND(MIN(temp_motor2_1_mean), 1),
            ROUND(AVG(temp_motor2_1_mean), 1),
            ROUND(MAX(temp_motor2_1_max), 1),
            '°C'
        FROM features WHERE temp_motor2_1_mean IS NOT NULL
        UNION ALL
        SELECT 
            'Current U',
            ROUND(MIN(current_u_mean), 1),
            ROUND(AVG(current_u_mean), 1),
            ROUND(MAX(current_u_max), 1),
            'A'
        FROM features WHERE current_u_mean IS NOT NULL
        UNION ALL
        SELECT 
            'Pressure TR1',
            ROUND(MIN(pressure_tr1_min), 2),
            ROUND(AVG(pressure_tr1_mean), 2),
            ROUND(MAX(pressure_tr1_mean), 2),
            'bar'
        FROM features WHERE pressure_tr1_mean IS NOT NULL
        UNION ALL
        SELECT 
            'Battery Voltage',
            ROUND(MIN(battery_volt_min), 1),
            ROUND(AVG(battery_volt_mean), 1),
            ROUND(MAX(battery_volt_mean), 1),
            'V'
        FROM features WHERE battery_volt_mean IS NOT NULL
        UNION ALL
        SELECT 
            'Speed',
            ROUND(MIN(avg_speed), 1),
            ROUND(AVG(avg_speed), 1),
            ROUND(MAX(max_speed), 1),
            'km/h'
        FROM features WHERE avg_speed IS NOT NULL
    """).df()
    
    print(sensor_ranges.to_string(index=False))
    reports['sensor_ranges'] = sensor_ranges
    
    # ========================================================================
    # 3. Fault Summary
    # ========================================================================
    print("\n📊 3. Fault Summary")
    print("-" * 60)
    
    fault_summary = con.execute("""
        SELECT 
            locoid,
            SUM(fault_count) AS total_faults,
            COUNT(CASE WHEN fault_count > 0 THEN 1 END) AS intervals_with_faults,
            MAX(max_faultnum) AS highest_fault_code
        FROM features
        GROUP BY locoid
        HAVING total_faults > 0
        ORDER BY total_faults DESC
    """).df()
    
    if len(fault_summary) > 0:
        print(fault_summary.to_string(index=False))
        reports['fault_summary'] = fault_summary
    else:
        print("No faults detected in the dataset.")
        reports['fault_summary'] = pd.DataFrame()
    
    # ========================================================================
    # 4. Operational Regime Analysis
    # ========================================================================
    print("\n📊 4. Operational Regime Analysis")
    print("-" * 60)
    
    regime_analysis = con.execute("""
        SELECT 
            locoid,
            COUNT(*) AS total_intervals,
            ROUND(AVG(CASE WHEN pct_moving > 0.5 THEN 1 ELSE 0 END) * 100, 1) AS pct_time_moving,
            ROUND(AVG(avg_speed), 1) AS avg_speed_overall,
            ROUND(AVG(CASE WHEN pct_moving > 0.5 THEN avg_speed END), 1) AS avg_speed_when_moving,
            ROUND(SUM(energy_consumption), 1) AS total_energy_kwh,
            ROUND(SUM(distance_km), 1) AS total_distance_km,
            ROUND(AVG(energy_efficiency), 3) AS avg_energy_efficiency
        FROM features
        GROUP BY locoid
        ORDER BY locoid
    """).df()
    
    print(regime_analysis.to_string(index=False))
    reports['regime_analysis'] = regime_analysis
    
    # ========================================================================
    # 5. Temperature Distribution by Locomotive
    # ========================================================================
    print("\n📊 5. Temperature Distribution (Motor 1-1)")
    print("-" * 60)
    
    temp_distribution = con.execute("""
        SELECT 
            locoid,
            ROUND(MIN(temp_motor1_1_mean), 1) AS min_temp,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY temp_motor1_1_mean), 1) AS q25_temp,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY temp_motor1_1_mean), 1) AS median_temp,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY temp_motor1_1_mean), 1) AS q75_temp,
            ROUND(MAX(temp_motor1_1_max), 1) AS max_temp,
            ROUND(STDDEV(temp_motor1_1_mean), 1) AS std_temp
        FROM features
        WHERE temp_motor1_1_mean IS NOT NULL
        GROUP BY locoid
        ORDER BY locoid
    """).df()
    
    print(temp_distribution.to_string(index=False))
    reports['temp_distribution'] = temp_distribution
    
    # ========================================================================
    # 6. Data Quality Metrics
    # ========================================================================
    print("\n📊 6. Data Quality Metrics")
    print("-" * 60)
    
    data_quality = con.execute("""
        SELECT 
            locoid,
            COUNT(*) AS total_records,
            ROUND(AVG(CASE WHEN temp_motor1_1_mean IS NULL THEN 1 ELSE 0 END) * 100, 1) AS pct_missing_temp,
            ROUND(AVG(CASE WHEN current_u_mean IS NULL THEN 1 ELSE 0 END) * 100, 1) AS pct_missing_current,
            ROUND(AVG(CASE WHEN pressure_tr1_mean IS NULL THEN 1 ELSE 0 END) * 100, 1) AS pct_missing_pressure,
            ROUND(AVG(gps_availability) * 100, 1) AS avg_gps_availability
        FROM features
        GROUP BY locoid
        ORDER BY locoid
    """).df()
    
    print(data_quality.to_string(index=False))
    reports['data_quality'] = data_quality
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"✓ Descriptive analysis complete!")
    print(f"{'='*60}\n")
    
    con.close()
    
    return reports


if __name__ == "__main__":
    reports = descriptive_analysis()
