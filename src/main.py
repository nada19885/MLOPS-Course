import os
from pathlib import Path
import traceback

from data.load import load_data
from features.build_features import (
    engineer_features,
    get_preprocessing_pipeline,
    save_intermediate,
)
from models.train_model import train_model
import pandas as pd


def main():
    print("Starting pipeline...")
    
    # Create necessary directories if they don't exist
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Configuration (hard-coded for Lab 0)
    data_path = "data/raw/train.csv"
    
    # Check if the data file exists
    if not Path(data_path).exists():
        print(f"Error: Data file not found at {data_path}")
        print("Current working directory:", os.getcwd())
        print("Please make sure the data file exists at the specified path")
        return
        
    numerical_features = ["Age", "SibSp", "Parch", "Fare", "family_size", "is_alone"]
    categorical_features = ["Pclass", "Sex", "Embarked"]
    output_paths = {
        "feature_engineered": "data/interim/feature_engineered.pkl",
        "processed": "data/interim/processed.pkl",
    }
    model_configs = [
        {
            "name": "logistic",
            "params": {},
            "path": "models/model_logistic.pkl",
        },
        {
            "name": "random_forest",
            "params": {"n_estimators": 100, "max_depth": 5},
            "path": "models/model_random_forest.pkl",
        },
    ]

    try:
        # Load data
        print(f"Loading data from {data_path}")
        df = load_data(data_path)
        print(f"Data loaded: {df.shape}")
        
        df = engineer_features(df)
        print("Features engineered")
        print(f"Dataframe columns after engineering: {df.columns.tolist()}")
        
        save_intermediate(df, output_paths["feature_engineered"])
        print(f"Saved feature-engineered data to {output_paths['feature_engineered']}")

        # Prepare features and target
        X = df[numerical_features + categorical_features]
        
        # Check if Survived column exists
        if "Survived" not in df.columns:
            print("Error: 'Survived' column not found in the dataframe")
            print(f"Available columns: {df.columns.tolist()}")
            return
            
        y = df["Survived"]
        print(f"Features and target prepared: X shape={X.shape}, y shape={y.shape}")

        # Preprocessing pipeline
        pipeline = get_preprocessing_pipeline(numerical_features, categorical_features)
        save_intermediate(pipeline, output_paths["processed"])
        print(f"Saved preprocessing pipeline to {output_paths['processed']}")

        # Train and evaluate models
        for config in model_configs:
            print(f"Training {config['name']}...")
            model, accuracy, report = train_model(
                X,
                y,
                pipeline,
                config["name"],
                config["params"],
                config["path"],
            )
            print(f"{config['name']} - Accuracy: {accuracy}")
            print(f"Classification Report:\n{report}")
            print("Successfully trained and saved model to", config["path"])
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Executing main.py")
    main()

def main():
    print("Running main function.")

if __name__ == "__main__":
    main()
