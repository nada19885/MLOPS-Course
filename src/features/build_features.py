import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from the input dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("Starting feature engineering")
    df = df.copy()
    
    # Create family_size and is_alone features
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    
    print(f"Feature engineering completed. New columns: {df.columns.tolist()}")
    return df


def get_preprocessing_pipeline(numerical_features, categorical_cols):
    """
    Create a preprocessing pipeline for numerical and categorical features.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_cols: List of categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    print(f"Creating preprocessing pipeline for {len(numerical_features)} numerical and {len(categorical_cols)} categorical features")
    
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def save_intermediate(obj, path: str):
    """
    Save intermediate object as a pickle file.
    
    Args:
        obj: Object to save
        path: Path to save the object
    """
    print(f"Saving object to {path}")
    joblib.dump(obj, path)
    print("Object saved successfully")