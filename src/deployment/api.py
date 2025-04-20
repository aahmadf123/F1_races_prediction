from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="F1 Race Winner Prediction API",
    description="API for predicting F1 race winners using ensemble models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionInput(BaseModel):
    driver: str
    team: str
    circuit: str
    date: str
    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float
    rolling_position: float
    rolling_points: float
    team_rolling_points: float

class PredictionOutput(BaseModel):
    predicted_position: float
    confidence: float
    model_contributions: Dict[str, float]

class ModelService:
    def __init__(self):
        self.models_path = Path(os.getenv('MODEL_REGISTRY_PATH', 'models'))
        self.models = {}
        self.scalers = {}
        self.load_latest_models()
        
    def load_latest_models(self):
        """Load the latest ensemble models and scalers."""
        try:
            # Find latest model directory
            model_dirs = sorted(self.models_path.glob('ensemble_*'))
            if not model_dirs:
                raise ValueError("No models found in registry")
                
            latest_dir = model_dirs[-1]
            
            # Load models
            for model_file in latest_dir.glob('*_model.joblib'):
                model_name = model_file.stem.replace('_model', '')
                self.models[model_name] = joblib.load(model_file)
                
            # Load scalers
            scalers_dir = Path(os.getenv('FEATURE_STORE_PATH', 'features')) / 'scalers'
            for scaler_file in scalers_dir.glob('*_scaler.joblib'):
                feature_name = scaler_file.stem.replace('_scaler', '')
                self.scalers[feature_name] = joblib.load(scaler_file)
                
            logger.info(f"Loaded models from {latest_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def preprocess_input(self, input_data: PredictionInput) -> pd.DataFrame:
        """Preprocess input data for prediction."""
        # Create feature dictionary
        features = {
            'rolling_position': input_data.rolling_position,
            'rolling_points': input_data.rolling_points,
            'team_rolling_points': input_data.team_rolling_points,
            'temperature': input_data.temperature,
            'humidity': input_data.humidity,
            'wind_speed': input_data.wind_speed,
            'precipitation': input_data.precipitation
        }
        
        # Create interaction terms
        features['temp_humidity'] = features['temperature'] * features['humidity']
        features['wind_precip'] = features['wind_speed'] * features['precipitation']
        
        # Add time features
        date = datetime.strptime(input_data.date, '%Y-%m-%d')
        features['month'] = date.month
        features['day_of_week'] = date.weekday()
        
        # Scale features
        scaled_features = {}
        for feature, value in features.items():
            if feature in self.scalers:
                scaled_features[f'{feature}_scaled'] = self.scalers[feature].transform([[value]])[0][0]
            else:
                scaled_features[feature] = value
                
        return pd.DataFrame([scaled_features])
        
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """Generate prediction using ensemble models."""
        try:
            # Preprocess input
            X = self.preprocess_input(input_data)
            
            # Get predictions from each model
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(X)[0]
                predictions[name] = pred
                
            # Calculate ensemble prediction (weighted average)
            weights = {'xgboost': 0.4, 'lightgbm': 0.3, 'catboost': 0.3}
            ensemble_pred = sum(pred * weights[name] for name, pred in predictions.items())
            
            # Calculate confidence (inverse of prediction spread)
            pred_std = np.std(list(predictions.values()))
            confidence = 1 / (1 + pred_std)
            
            return PredictionOutput(
                predicted_position=float(ensemble_pred),
                confidence=float(confidence),
                model_contributions=predictions
            )
            
        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize model service
model_service = ModelService()

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Generate race position prediction."""
    return model_service.predict(input_data)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": list(model_service.models.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 