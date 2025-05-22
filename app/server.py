from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
from pathlib import Path
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Predictor",
    description="API to predict passenger survival on the Titanic",
    version="1.0.0"
)

# Load the model once at startup
model_path = "models/model_logistic.pkl"
if not Path(model_path).exists():
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define input schema with validation
class Passenger(BaseModel):
    PassengerId: int = Field(..., description="Passenger ID")
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Name: str = Field(..., description="Passenger name")
    Sex: str = Field(..., pattern="^(male|female)$", description="Gender (male or female)")
    Age: Optional[float] = Field(None, ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Ticket: str = Field(..., description="Ticket number")
    Fare: Optional[float] = Field(None, ge=0, description="Passenger fare")
    Cabin: Optional[str] = Field(None, description="Cabin number")
    Embarked: Optional[str] = Field(None, pattern="^[CSQ]$", description="Port of embarkation (C, S, or Q)")

    class Config:
        schema_extra = {
            "example": {
                "PassengerId": 1,
                "Pclass": 3,
                "Name": "Braund, Mr. Owen Harris",
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Ticket": "A/5 21171",
                "Fare": 7.25,
                "Cabin": None,
                "Embarked": "S"
            }
        }

class PredictionResponse(BaseModel):
    passenger_id: int
    prediction: int
    survival_status: str
    probability: dict
    confidence: float

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as in training
    """
    df = df.copy()
    
    # Create family_size and is_alone features
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    
    return df

@app.get("/")
def read_root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Welcome to the Titanic Survival Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST request to predict survival",
            "health": "/health - GET request to check API health",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    try:
        # Quick model test
        test_data = pd.DataFrame({
            "Pclass": [3], "Sex": ["male"], "Age": [22.0], 
            "SibSp": [1], "Parch": [0], "Fare": [7.25], 
            "Embarked": ["S"], "family_size": [2], "is_alone": [0]
        })
        _ = model.predict_proba(test_data)
        return {"status": "healthy", "model": "loaded"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict", response_model=PredictionResponse)
def predict_survival(passenger: Passenger):
    """
    Predict survival probability for a Titanic passenger
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([passenger.dict()])
        
        # Apply feature engineering
        input_df = engineer_features(input_df)
        
        # Select only the features used in training
        feature_columns = [
            "Pclass", "Sex", "Age", "SibSp", "Parch", 
            "Fare", "Embarked", "family_size", "is_alone"
        ]
        
        # Handle missing values for prediction
        if input_df["Age"].isna().any():
            input_df["Age"].fillna(input_df["Age"].mean() if not input_df["Age"].isna().all() else 30, inplace=True)
        if input_df["Fare"].isna().any():
            input_df["Fare"].fillna(input_df["Fare"].mean() if not input_df["Fare"].isna().all() else 15, inplace=True)
        if input_df["Embarked"].isna().any():
            input_df["Embarked"].fillna("S", inplace=True)
        
        model_input = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(model_input)[0]
        probabilities = model.predict_proba(model_input)[0]
        
        # Calculate confidence (max probability)
        confidence = float(max(probabilities))
        
        return PredictionResponse(
            passenger_id=passenger.PassengerId,
            prediction=int(prediction),
            survival_status="Survived" if prediction == 1 else "Did not survive",
            probability={
                "Not Survived": float(probabilities[0]),
                "Survived": float(probabilities[1])
            },
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """
    Get information about the loaded model
    """
    try:
        model_info = {
            "model_type": type(model.named_steps['model']).__name__,
            "features_used": [
                "Pclass", "Sex", "Age", "SibSp", "Parch", 
                "Fare", "Embarked", "family_size", "is_alone"
            ],
            "preprocessing_steps": list(model.named_steps.keys()),
            "model_file": model_path
        }
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)