
import os
from pathlib import Path
import traceback

from data.load import load_data
from features.build_features import (
    engineer_features,
    get_preprocessing_pipeline,
    save_intermediate,
)
import hydra
from hydra.utils import get_original_cwd
from models.train_model import train_model
import pandas as pd


@hydra.main(config_path="../../conf/pipeline", config_name="titanic", version_base="1.3")
def main(cfg):
    print("Starting pipeline...")

    # Adjust paths to account for Hydra's working directory
    os.chdir(get_original_cwd())
    
    # Create necessary directories
    os.makedirs(cfg.pipeline.data.interim_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.pipeline.models[0].path), exist_ok=True)

    # Check if the data file exists
    data_path = cfg.pipeline.data.raw_data_path
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Current working directory:", os.getcwd())
        print("Please make sure the data file exists at the specified path")
        return

    try:
        # Load data
        print(f"Loading data from {data_path}")
        df = load_data(data_path)
        print(f"Data loaded: {df.shape}")

        # Engineer features
        df = engineer_features(df)
        print("Features engineered")
        print(f"Dataframe columns after engineering: {df.columns.tolist()}")
        
        save_intermediate(df, cfg.pipeline.data.feature_engineered_path)
        print(f"Saved feature-engineered data to {cfg.pipeline.data.feature_engineered_path}")

        # Prepare features and target
        X = df[cfg.pipeline.features.numerical + cfg.pipeline.features.categorical]
        
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

        # Train and evaluate models
        for config in cfg.pipeline.models:
            print(f"Training {config.name}...")
            model, accuracy, report = train_model(
                X,
                y,
                pipeline,
                config.name,
                config.params,
                config.path,
            )
            print(f"{config.name} - Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
            print(f"Successfully trained and saved model to {config.path}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Executing main.py")
    main()
