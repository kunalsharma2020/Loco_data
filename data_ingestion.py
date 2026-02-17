"""
Data ingestion module: Convert CSV files to partitioned Parquet format
"""
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
from config import INPUT_DIR, PARQUET_DIR, CHUNK_SIZE, dtype_dict, FLAG_DTYPE


def detect_flag_columns(df):
    """Detect binary flag columns (columns with only 0/1 values)"""
    flag_cols = []
    for col in df.columns:
        if col not in dtype_dict:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                flag_cols.append(col)
    return flag_cols


def clean_gps_data(df):
    """Clean invalid GPS coordinates (0,0) by setting them to NaN"""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        invalid_gps = (df['latitude'] == 0) & (df['longitude'] == 0)
        df.loc[invalid_gps, ['latitude', 'longitude', 'altitude']] = np.nan
    return df


def convert_csv_to_parquet(csv_path, out_dir, chunksize=CHUNK_SIZE):
    """
    Convert a single CSV file to partitioned Parquet format
    
    Parameters:
    -----------
    csv_path : str
        Path to input CSV file
    out_dir : str
        Output directory for Parquet files
    chunksize : int
        Number of rows to read per chunk
    """
    csv_name = Path(csv_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {csv_name}")
    print(f"{'='*60}")
    
    try:
        # Read first chunk to detect flag columns
        first_chunk = pd.read_csv(
    csv_path,
    nrows=1000,
    encoding="cp1252",
    encoding_errors="replace"
    )
        flag_cols = detect_flag_columns(first_chunk)
        
        # Update dtype_dict with flag columns
        full_dtype = dtype_dict.copy()
        for col in flag_cols:
            full_dtype[col] = FLAG_DTYPE
        
        # Process CSV in chunks
        chunk_iter = pd.read_csv(
    csv_path,
    chunksize=chunksize,
    dtype={k: v for k, v in full_dtype.items() if k != 'devicetime'},
    parse_dates=['devicetime'],
    low_memory=False,
    encoding="cp1252",
    encoding_errors="replace"
    )
        
        total_rows = 0
        for i, chunk in enumerate(tqdm(chunk_iter, desc=f"Converting {csv_name}")):
            # Extract date for partitioning
            chunk['date'] = pd.to_datetime(chunk['devicetime']).dt.date
            
            # Clean GPS data
            chunk = clean_gps_data(chunk)
            
            # Write partitioned Parquet
            for date, group in chunk.groupby('date'):
                partition_path = Path(out_dir) / f"date={date}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                output_file = partition_path / f"part-{csv_name}-{i:04d}.parquet"
                group.drop('date', axis=1).to_parquet(
                    output_file,
                    engine='pyarrow',
                    compression='zstd',
                    index=False
                )
            
            total_rows += len(chunk)
        
        print(f"✓ Converted {total_rows:,} rows from {csv_name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {csv_name}: {str(e)}")
        return False


def convert_all_csvs():
    """Convert all CSV files in INPUT_DIR to Parquet format"""
    csv_files = sorted(glob.glob(str(INPUT_DIR / "*.csv")))
    
    if not csv_files:
        print(f"⚠ No CSV files found in {INPUT_DIR}")
        return
    
    print(f"\n🚂 Found {len(csv_files)} CSV file(s) to convert")
    print(f"📂 Input:  {INPUT_DIR}")
    print(f"📂 Output: {PARQUET_DIR}")
    print(f"\nStarting conversion...\n")
    
    success_count = 0
    for csv_file in csv_files:
        if convert_csv_to_parquet(csv_file, str(PARQUET_DIR)):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete: {success_count}/{len(csv_files)} files successful")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Run the conversion
    convert_all_csvs()
