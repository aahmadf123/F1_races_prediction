import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, data_path: str = 'data'):
        self.data_path = Path(data_path)
        self.processed_dir = self.data_path / 'processed'
        self.features_dir = self.data_path / 'features'
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scalers
        self.scalers = {}
        
    def calculate_driver_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate rolling performance metrics for drivers."""
        # Sort by date and driver
        df = df.sort_values(['date', 'driver'])
        
        # Calculate rolling average position
        df['rolling_position'] = df.groupby('driver')['position'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Calculate rolling points
        df['rolling_points'] = df.groupby('driver')['points'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        
        return df
        
    def calculate_team_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate rolling performance metrics for teams."""
        # Sort by date and team
        df = df.sort_values(['date', 'team'])
        
        # Calculate rolling team points
        df['team_rolling_points'] = df.groupby('team')['points'].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        
        # Calculate team position trend
        df['team_position_trend'] = df.groupby('team')['position'].transform(
            lambda x: x.diff().rolling(window, min_periods=1).mean()
        )
        
        return df
        
    def calculate_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate circuit-specific features."""
        # Group by circuit to calculate statistics
        circuit_stats = df.groupby('circuit').agg({
            'position': ['mean', 'std'],
            'points': ['mean', 'sum']
        }).reset_index()
        
        # Flatten column names
        circuit_stats.columns = ['circuit', 'circuit_position_mean', 
                               'circuit_position_std', 'circuit_points_mean',
                               'circuit_points_sum']
        
        # Merge back to main dataframe
        df = df.merge(circuit_stats, on='circuit', how='left')
        
        return df
        
    def calculate_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process weather data into features."""
        # Extract relevant weather features
        weather_features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        
        # Create interaction terms
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['wind_precip'] = df['wind_speed'] * df['precipitation']
        
        return df
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        return df
        
    def scale_features(self, df: pd.DataFrame, features_to_scale: List[str]) -> pd.DataFrame:
        """Scale numerical features."""
        for feature in features_to_scale:
            if feature not in self.scalers:
                self.scalers[feature] = StandardScaler()
                df[f'{feature}_scaled'] = self.scalers[feature].fit_transform(
                    df[[feature]]
                )
            else:
                df[f'{feature}_scaled'] = self.scalers[feature].transform(
                    df[[feature]]
                )
                
        return df
        
    def save_scalers(self):
        """Save fitted scalers for future use."""
        scalers_path = self.features_dir / 'scalers'
        scalers_path.mkdir(exist_ok=True)
        
        for feature, scaler in self.scalers.items():
            joblib.dump(scaler, scalers_path / f'{feature}_scaler.joblib')
            
    def run_feature_engineering(self, input_file: str) -> pd.DataFrame:
        """Run the complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")
        
        try:
            # Load processed data
            df = pd.read_parquet(self.processed_dir / input_file)
            
            # Apply feature engineering steps
            df = self.calculate_driver_form(df)
            df = self.calculate_team_form(df)
            df = self.calculate_circuit_features(df)
            df = self.calculate_weather_features(df)
            df = self.create_time_features(df)
            
            # Scale numerical features
            numerical_features = [
                'rolling_position', 'rolling_points', 'team_rolling_points',
                'circuit_position_mean', 'circuit_position_std',
                'temperature', 'humidity', 'wind_speed', 'precipitation'
            ]
            
            df = self.scale_features(df, numerical_features)
            
            # Save engineered features
            output_file = self.features_dir / f'engineered_{input_file}'
            df.to_parquet(output_file)
            
            # Save scalers
            self.save_scalers()
            
            logger.info("Feature engineering pipeline completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise
            
if __name__ == "__main__":
    # Example usage
    feature_engineering = FeatureEngineering()
    df = feature_engineering.run_feature_engineering('processed_2023.parquet') 