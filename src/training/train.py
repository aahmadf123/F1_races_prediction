import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import mlflow
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTraining:
    def __init__(self, data_path: str = 'data', models_path: str = 'models'):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.features_dir = self.data_path / 'features'
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(f"file:{self.models_path}/mlruns")
        
    def prepare_data(self, input_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for training."""
        df = pd.read_parquet(self.features_dir / input_file)
        
        # Define features and target
        feature_cols = [
            'rolling_position_scaled', 'rolling_points_scaled',
            'team_rolling_points_scaled', 'circuit_position_mean_scaled',
            'circuit_position_std_scaled', 'temperature_scaled',
            'humidity_scaled', 'wind_speed_scaled', 'precipitation_scaled',
            'temp_humidity', 'wind_precip', 'month', 'day_of_week'
        ]
        
        target_col = 'position'
        
        X = df[feature_cols]
        y = df[target_col]
        
        return X, y
        
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, 
                      params: Optional[Dict] = None) -> xgb.XGBRegressor:
        """Train XGBoost model with optional hyperparameter tuning."""
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
            
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        return model
        
    def train_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                       params: Optional[Dict] = None) -> lgb.LGBMRegressor:
        """Train LightGBM model with optional hyperparameter tuning."""
        if params is None:
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
            
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        return model
        
    def train_catboost(self, X: pd.DataFrame, y: pd.Series,
                       params: Optional[Dict] = None) -> cat.CatBoostRegressor:
        """Train CatBoost model with optional hyperparameter tuning."""
        if params is None:
            params = {
                'loss_function': 'MAE',
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6
            }
            
        model = cat.CatBoostRegressor(**params, verbose=False)
        model.fit(X, y)
        
        return model
        
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Optuna objective function for hyperparameter tuning."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.train_xgboost(X_train, y_train, params)
            pred = model.predict(X_val)
            score = mean_absolute_error(y_val, pred)
            scores.append(score)
            
        return np.mean(scores)
        
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble of models."""
        logger.info("Starting ensemble model training")
        
        # Train individual models
        xgb_model = self.train_xgboost(X, y)
        lgb_model = self.train_lightgbm(X, y)
        cat_model = self.train_catboost(X, y)
        
        models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'catboost': cat_model
        }
        
        return models
        
    def evaluate_models(self, models: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance."""
        results = {}
        
        for name, model in models.items():
            pred = model.predict(X)
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            
            results[name] = {
                'mae': mae,
                'rmse': rmse
            }
            
        return results
        
    def save_models(self, models: Dict):
        """Save trained models."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = self.models_path / f'ensemble_{timestamp}'
        save_dir.mkdir(exist_ok=True)
        
        for name, model in models.items():
            model_path = save_dir / f'{name}_model.joblib'
            joblib.dump(model, model_path)
            
    def run_training_pipeline(self, input_file: str):
        """Run the complete model training pipeline."""
        logger.info("Starting model training pipeline")
        
        try:
            # Prepare data
            X, y = self.prepare_data(input_file)
            
            # Train models
            models = self.train_ensemble(X, y)
            
            # Evaluate models
            results = self.evaluate_models(models, X, y)
            
            # Log results with MLflow
            with mlflow.start_run():
                mlflow.log_params({
                    'input_file': input_file,
                    'n_features': X.shape[1],
                    'n_samples': X.shape[0]
                })
                
                for model_name, metrics in results.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f'{model_name}_{metric_name}', value)
                        
            # Save models
            self.save_models(models)
            
            logger.info("Model training pipeline completed successfully")
            return models, results
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {str(e)}")
            raise
            
if __name__ == "__main__":
    # Example usage
    training = ModelTraining()
    models, results = training.run_training_pipeline('engineered_2023.parquet') 