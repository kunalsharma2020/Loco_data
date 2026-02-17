"""
Main pipeline runner: Execute the complete analysis pipeline
"""
import sys
from datetime import datetime
from data_ingestion import convert_all_csvs
from feature_engineering import create_aggregated_features
from anomaly_detection import detect_anomalies
from descriptive_analysis import descriptive_analysis


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_full_pipeline():
    """Run the complete analysis pipeline"""
    start_time = datetime.now()
    
    print_banner("🚂 LOCOMOTIVE SENSOR DATA ANALYSIS PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Data Ingestion
        print_banner("STEP 1/4: Data Ingestion (CSV → Parquet)")
        convert_all_csvs()
        
        # Step 2: Feature Engineering
        print_banner("STEP 2/4: Feature Engineering")
        success = create_aggregated_features()
        if not success:
            print("❌ Feature engineering failed. Stopping pipeline.")
            return False
        
        # Step 3: Anomaly Detection
        print_banner("STEP 3/4: Anomaly Detection")
        detect_anomalies()
        
        # Step 4: Descriptive Analysis
        print_banner("STEP 4/4: Descriptive Analysis")
        descriptive_analysis()
        
        # Pipeline complete
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_banner("✅ PIPELINE COMPLETE")
        print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        print("\n" + "="*70)
        print("\n📊 Next Steps:")
        print("   1. Review the descriptive analysis output above")
        print("   2. Launch the dashboard: streamlit run dashboard.py")
        print("   3. Explore anomalies and sensor trends")
        print("\n" + "="*70 + "\n")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_step(step_name):
    """Run a specific pipeline step"""
    steps = {
        'ingest': ('Data Ingestion', convert_all_csvs),
        'features': ('Feature Engineering', create_aggregated_features),
        'anomalies': ('Anomaly Detection', detect_anomalies),
        'analysis': ('Descriptive Analysis', descriptive_analysis),
    }
    
    if step_name not in steps:
        print(f"❌ Unknown step: {step_name}")
        print(f"Available steps: {', '.join(steps.keys())}")
        return False
    
    step_title, step_func = steps[step_name]
    print_banner(f"Running: {step_title}")
    
    try:
        step_func()
        print(f"\n✅ {step_title} complete\n")
        return True
    except Exception as e:
        print(f"\n❌ {step_title} failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific step
        step = sys.argv[1].lower()
        run_step(step)
    else:
        # Run full pipeline
        run_full_pipeline()
