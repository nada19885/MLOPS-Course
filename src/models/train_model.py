import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


def train_model(X, y, pipeline,model_name,model_path, params={}):
    """
    Train and evaluate a machine learning model.
    
    Args:
        X: Features dataframe
        y: Target series
        pipeline: Preprocessing pipeline
        model_name: Type of model to train ('logistic' or 'random_forest')
        params: Model parameters
        model_path: Path to save the trained model
        
    Returns:
        tuple: (model, accuracy, classification_report)
    """
    print(f"Starting model training for {model_name} with params {params}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == 'logistic':
        model = LogisticRegression(**params)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(**params)
    else:
        raise ValueError("Model name not recognized. Use 'logistic' or 'random_forest'.")
    
    print(f"Created {model_name} model")
    full_model = Pipeline(steps=[('preprocessor', pipeline), ('model', model)])

    print("Fitting model...")
    full_model.fit(X_train, y_train)
    print("Model fitted. Generating predictions...")
    
    y_pred = full_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Saving model to {model_path}")
    joblib.dump(full_model, model_path)
    print("Model saved successfully")
    
    return full_model, accuracy, report