import os
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd
import pandas as pd
import joblib
from data.load import load_data
from features.build_features import engineer_features, get_preprocessing_pipeline, save_intermediate

@hydra.main(config_path="../conf/pipeline", config_name="titanic", version_base="1.3")
def process_data(cfg):
    print("Starting data processing stage...")

    # Adjust paths for Hydra's working directory
    os.chdir(get_original_cwd())

    # Create necessary directories
    os.makedirs(cfg.pipeline.data.interim_dir, exist_ok=True)

    # Check if the data file exists
    data_path = cfg.pipeline.data.raw_data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Current working directory:", os.getcwd())
        return

    # Load data
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    print(f"Data loaded: {df.shape}")

    # Engineer features
    df = engineer_features(df)
    print("Features engineered")
    print(f"Dataframe columns after engineering: {df.columns.tolist()}")
    
    # Save feature-engineered data
    save_intermediate(df, cfg.pipeline.data.feature_engineered_path)
    print(f"Saved feature-engineered data to {cfg.pipeline.data.feature_engineered_path}")

    # Prepare features and target
    feature_columns = cfg.pipeline.features.numerical + cfg.pipeline.features.categorical
    print(f"Expected features: {feature_columns}")
    if not all(col in df.columns for col in feature_columns):
        print(f"Error: Some features not found in DataFrame. Available columns: {df.columns.tolist()}")
        return
        
    X = df[feature_columns]
    if "Survived" not in df.columns:
        print("Error: 'Survived' column not found in the dataframe")
        print(f"Available columns: {df.columns.tolist()}")
        return
        
    y = df["Survived"]
    print(f"Features and target prepared: X shape={X.shape}, y shape={y.shape}")

    # Preprocessing pipeline
    pipeline = get_preprocessing_pipeline(
        cfg.pipeline.features.numerical, cfg.pipeline.features.categorical
    )
    save_intermediate(pipeline, cfg.pipeline.data.processed_pipeline_path)
    print(f"Saved preprocessing pipeline to {cfg.pipeline.data.processed_pipeline_path}")

if __name__ == "__main__":
    process_data()