import os
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd
import pandas as pd
import joblib
from models.train_model import train_model

@hydra.main(config_path="../conf/pipeline", config_name="titanic", version_base="1.3")
def main(cfg):
    print("Starting model training stage...")

    # Adjust paths for Hydra's working directory
    os.chdir(get_original_cwd())

    # Create models directory
    os.makedirs(os.path.dirname(cfg.pipeline.models[0].path), exist_ok=True)

    # Check if required files exist
    feature_engineered_path = cfg.pipeline.data.feature_engineered_path
    pipeline_path = cfg.pipeline.data.processed_pipeline_path
    
    if not Path(feature_engineered_path).exists():
        print(f"Error: Feature engineered data not found at {feature_engineered_path}")
        print("Please run the processing stage first")
        return
        
    if not Path(pipeline_path).exists():
        print(f"Error: Preprocessing pipeline not found at {pipeline_path}")
        print("Please run the processing stage first")
        return

    try:
        # Load processed data and pipeline
        print(f"Loading feature engineered data from {feature_engineered_path}")
        df = joblib.load(feature_engineered_path)
        print(f"Data loaded: {df.shape}")
        
        print(f"Loading preprocessing pipeline from {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        print("Pipeline loaded")

        # Prepare features and target
        feature_columns = cfg.pipeline.features.numerical + cfg.pipeline.features.categorical
        print(f"Using features: {feature_columns}")
        
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

        # Train and evaluate models
        for config in cfg.pipeline.models:
            print(f"Training {config.name}...")
            model, accuracy, report = train_model(
                X,
                y,
                pipeline,
                config.name,
                config.path,
                config.params
                
            )
            print(f"{config.name} - Accuracy: {accuracy:.4f}")
            print(f"Classification Report:\n{report}")
            print(f"Successfully trained and saved model to {config.path}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()