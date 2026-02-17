"""
Feature Engineering Module
Creates aggregated features from raw Parquet data using DuckDB.
"""

import duckdb
from pathlib import Path
from config import (
    PARQUET_DIR, FEATURES_DIR, AGGREGATION_INTERVAL
)

def create_aggregated_features():
    """
    Aggregate raw sensor data into time-bucketed features.
    Uses DuckDB for efficient out-of-core processing.
    """
    print("\n" + "="*60)
    print("Creating aggregated features...")
    print("="*60)
    
    parquet_glob = str(PARQUET_DIR / "date=*" / "part-*.parquet")
    output_path = FEATURES_DIR / "features_1min.parquet"
    
    print(f"📂 Input:  {parquet_glob}")
    print(f"📂 Output: {output_path}")
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        conn = duckdb.connect(database=':memory:')
        
        print("\n⚙️  Running aggregation query (this may take several minutes)...\n")
        
        # Main aggregation query
        query = f"""
        CREATE TABLE features_1min AS
        SELECT
            locoid,
            time_bucket(INTERVAL '{AGGREGATION_INTERVAL}', devicetime) AS ts,
            
            -- GPS & Movement
            AVG(gpsspeed) AS avg_speed,
            MAX(gpsspeed) AS max_speed,
            AVG(xspeedloco) AS avg_speed_loco,
            MAX(xspeedloco) AS max_speed_loco,
            AVG(CASE WHEN gpsspeed > 5 THEN 1 ELSE 0 END) AS pct_moving,
            AVG(latitude) AS avg_lat,
            AVG(longitude) AS avg_lon,
            COUNT(CASE WHEN latitude IS NOT NULL THEN 1 END) / COUNT(*)::FLOAT AS gps_availability,
            
            -- Temperature sensors (motor)
            AVG(xtempmotor1_1) AS temp_motor1_1_mean,
            MAX(xtempmotor1_1) AS temp_motor1_1_max,
            STDDEV(xtempmotor1_1) AS temp_motor1_1_std,
            AVG(xtempmotor1_2) AS temp_motor1_2_mean,
            MAX(xtempmotor1_2) AS temp_motor1_2_max,
            AVG(xtempmotor2_1) AS temp_motor2_1_mean,
            MAX(xtempmotor2_1) AS temp_motor2_1_max,
            AVG(xtempmotor2_2) AS temp_motor2_2_mean,
            MAX(xtempmotor2_2) AS temp_motor2_2_max,
            AVG(xtempmotor3_1) AS temp_motor3_1_mean,
            MAX(xtempmotor3_1) AS temp_motor3_1_max,
            AVG(xtempmotor3_2) AS temp_motor3_2_mean,
            MAX(xtempmotor3_2) AS temp_motor3_2_max,
            
            -- Temperature sensors (transformer/rectifier)
            AVG(xatmp1oeltr_1) AS temp_trans1_1_mean,
            MAX(xatmp1oeltr_1) AS temp_trans1_1_max,
            AVG(xatmp1oeltr_2) AS temp_trans1_2_mean,
            MAX(xatmp1oeltr_2) AS temp_trans1_2_max,
            AVG(xatmp2oeltr_1) AS temp_trans2_1_mean,
            MAX(xatmp2oeltr_1) AS temp_trans2_1_max,
            AVG(xatmp2oeltr_2) AS temp_trans2_2_mean,
            MAX(xatmp2oeltr_2) AS temp_trans2_2_max,
            
            -- Current sensors
            AVG(xuprim_1) AS current_u_mean,
            MAX(xuprim_1) AS current_u_max,
            STDDEV(xuprim_1) AS current_u_std,
            AVG(xiprim_1) AS current_i_mean,
            MAX(xiprim_1) AS current_i_max,
            AVG(xaibur) AS current_bur_mean,
            MAX(xaibur) AS current_bur_max,
            
            -- Pressure sensors
            AVG(xadrucktr_1) AS pressure_tr1_mean,
            MIN(xadrucktr_1) AS pressure_tr1_min,
            STDDEV(xadrucktr_1) AS pressure_tr1_std,
            AVG(xadrucktr_2) AS pressure_tr2_mean,
            MIN(xadrucktr_2) AS pressure_tr2_min,
            AVG(xadrucksr_1) AS pressure_sr1_mean,
            AVG(xadrucksr_2) AS pressure_sr2_mean,
            AVG(xprautobkln) AS pressure_brake_mean,
            AVG(xpressurecv_1) AS pressure_cv1_mean,
            AVG(xpressurecv_2) AS pressure_cv2_mean,
            
            -- Energy & Odometer (delta per interval)
            MAX(xenergkwh_plus) - MIN(xenergkwh_plus) AS energy_consumption,
            MAX(odometerK) - MIN(odometerK) AS distance_km,
            
            -- Battery & Temperature
            AVG(xu_battery) AS battery_volt_mean,
            MIN(xu_battery) AS battery_volt_min,
            AVG(xte_be_loco) AS temp_be_loco_mean,
            
            -- Faults
            SUM(CASE WHEN faultnum > 0 THEN 1 ELSE 0 END) AS fault_count,
            MAX(faultnum) AS max_faultnum,
            
            -- Data quality
            COUNT(*) AS sample_count
            
        FROM read_parquet('{parquet_glob}')
        WHERE devicetime IS NOT NULL
        GROUP BY locoid, ts
        ORDER BY locoid, ts;
        """
        
        conn.execute(query)
        
        # Add derived features using a separate query with window functions
        print("⚙️  Adding derived features (energy efficiency, temperature rate)...\n")
        
        derived_query = """
        CREATE TABLE features_final AS
        SELECT 
            *,
            -- Energy efficiency (kWh per km)
            CASE 
                WHEN distance_km > 0.1 THEN energy_consumption / distance_km 
                ELSE NULL 
            END AS energy_efficiency,
            
            -- Temperature rate of change (°C per minute)
            temp_motor1_1_mean - LAG(temp_motor1_1_mean) 
                OVER (PARTITION BY locoid ORDER BY ts) AS temp_motor1_1_rate
        FROM features_1min;
        """
        
        conn.execute(derived_query)
        
        # Export to Parquet
        print("💾 Writing features to Parquet...\n")
        conn.execute(f"""
            COPY features_final 
            TO '{output_path}' 
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """)
        
        # Get summary stats
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT locoid) as unique_locos,
                MIN(ts) as start_time,
                MAX(ts) as end_time
            FROM features_final;
        """).fetchone()
        
        conn.close()
        
        print("✓ Feature engineering complete!")
        print(f"  • Total feature rows: {stats[0]:,}")
        print(f"  • Unique locomotives: {stats[1]}")
        print(f"  • Time range: {stats[2]} to {stats[3]}")
        print(f"  • Output: {output_path}\n")
        
    except Exception as e:
        print(f"✗ Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    create_aggregated_features()