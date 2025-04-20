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
    def __init__(self):
        self.data_dir = Path('data')
        self.features_dir = self.data_dir / 'features'
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: list, context: str) -> bool:
        """Validate dataframe has required columns and non-empty data."""
        if df.empty:
            logger.error(f"{context}: Empty dataframe provided")
            return False
            
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"{context}: Missing required columns: {missing_cols}")
            return False
            
        return True
        
    def process_qualifying_data(self, qualifying_data: pd.DataFrame) -> pd.DataFrame:
        """Process qualifying data to create relevant features."""
        logger.info("Processing qualifying data")
        
        required_columns = ['year', 'round', 'driver', 'team', 'qualifying_position', 'qualifying_time', 'circuit']
        if not self._validate_dataframe(qualifying_data, required_columns, "Qualifying data"):
            return pd.DataFrame()
            
        try:
            # Handle missing or invalid qualifying times
            qualifying_data['qualifying_time'] = pd.to_numeric(qualifying_data['qualifying_time'], errors='coerce')
            
            # Calculate gap to pole position
            qualifying_data['gap_to_pole'] = qualifying_data.groupby(['year', 'round'])['qualifying_time'].transform(
                lambda x: x - x.min()
            )
            
            # Calculate gaps to cars ahead and behind
            qualifying_data = qualifying_data.sort_values(['year', 'round', 'qualifying_position'])
            qualifying_data['gap_to_car_ahead'] = qualifying_data.groupby(['year', 'round'])['qualifying_time'].diff()
            qualifying_data['gap_to_car_behind'] = qualifying_data.groupby(['year', 'round'])['qualifying_time'].diff(-1)
            
            # Calculate qualifying performance relative to team average
            team_avg = qualifying_data.groupby(['year', 'round', 'team'])['qualifying_time'].transform('mean')
            qualifying_data['team_quali_performance'] = qualifying_data['qualifying_time'] - team_avg
            
            # Calculate historical qualifying performance at each circuit
            circuit_history = qualifying_data.groupby(['driver', 'circuit'])['qualifying_position'].agg([
                'mean',
                'std',
                'min',
                'max',
                'count'
            ]).reset_index()
            
            # Handle potential division by zero in consistency calculations
            circuit_history['std_quali_pos'] = circuit_history['std_quali_pos'].fillna(0)
            circuit_history['circuit_consistency'] = 1 / (1 + circuit_history['std_quali_pos'])
            
            # Rename columns for clarity
            circuit_history.columns = [
                'driver', 'circuit', 'avg_quali_pos', 'std_quali_pos',
                'best_quali_pos', 'worst_quali_pos', 'circuit_experience'
            ]
            
            # Merge circuit history back to qualifying data
            qualifying_features = qualifying_data.merge(
                circuit_history,
                on=['driver', 'circuit'],
                how='left'
            )
            
            # Calculate recent form (last 3 races)
            recent_form = qualifying_data.sort_values(['driver', 'year', 'round']).groupby('driver').rolling(
                window=3, min_periods=1
            )['qualifying_position'].agg(['mean', 'std']).reset_index()
            
            recent_form.columns = ['driver', 'index', 'recent_quali_pos_mean', 'recent_quali_pos_std']
            
            # Calculate qualifying momentum (improvement in last 3 races)
            qualifying_data['quali_momentum'] = qualifying_data.sort_values(['driver', 'year', 'round']).groupby('driver')['qualifying_position'].transform(
                lambda x: x.diff().rolling(3, min_periods=1).mean()
            )
            
            # Calculate team qualifying strength
            team_strength = qualifying_data.groupby(['year', 'round', 'team'])['qualifying_position'].mean().reset_index()
            team_strength = team_strength.rename(columns={'qualifying_position': 'team_quali_strength'})
            
            # Merge team strength back
            qualifying_features = qualifying_features.merge(
                team_strength,
                on=['year', 'round', 'team'],
                how='left'
            )
            
            # Calculate qualifying position vs team strength
            qualifying_features['quali_vs_team'] = qualifying_features['qualifying_position'] - qualifying_features['team_quali_strength']
            
            # Merge recent form back to qualifying features
            qualifying_features = qualifying_features.merge(
                recent_form[['driver', 'recent_quali_pos_mean', 'recent_quali_pos_std']],
                on='driver',
                how='left'
            )
            
            # Handle potential division by zero in consistency calculations
            qualifying_features['recent_quali_pos_std'] = qualifying_features['recent_quali_pos_std'].fillna(0)
            qualifying_features['quali_consistency'] = 1 / (1 + qualifying_features['recent_quali_pos_std'])
            
            # Calculate position group performance
            try:
                qualifying_features['position_group'] = pd.qcut(
                    qualifying_features['qualifying_position'],
                    q=4,
                    labels=['front', 'upper_mid', 'lower_mid', 'back']
                )
            except ValueError as e:
                logger.warning(f"Error in position group calculation: {str(e)}. Using alternative grouping.")
                qualifying_features['position_group'] = pd.cut(
                    qualifying_features['qualifying_position'],
                    bins=[0, 3, 7, 12, float('inf')],
                    labels=['front', 'upper_mid', 'lower_mid', 'back']
                )
            
            group_performance = qualifying_features.groupby(['driver', 'position_group'])['qualifying_position'].agg(['mean', 'std']).reset_index()
            group_performance['std'] = group_performance['std'].fillna(0)
            group_performance['group_consistency'] = 1 / (1 + group_performance['std'])
            
            # Merge group performance back
            qualifying_features = qualifying_features.merge(
                group_performance[['driver', 'position_group', 'mean', 'group_consistency']],
                on=['driver', 'position_group'],
                how='left'
            )
            
            # Select final features
            quali_features = qualifying_features[[
                'year', 'round', 'driver', 'team',
                'qualifying_position', 'gap_to_pole',
                'gap_to_car_ahead', 'gap_to_car_behind',
                'team_quali_performance', 'team_quali_strength',
                'quali_vs_team', 'quali_momentum',
                'avg_quali_pos', 'std_quali_pos',
                'best_quali_pos', 'worst_quali_pos',
                'circuit_experience', 'circuit_consistency',
                'recent_quali_pos_mean', 'recent_quali_pos_std',
                'quali_consistency', 'position_group',
                'group_consistency'
            ]]
            
            # Fill any remaining NaN values with appropriate defaults
            quali_features = quali_features.fillna({
                'gap_to_pole': 0,
                'gap_to_car_ahead': 0,
                'gap_to_car_behind': 0,
                'team_quali_performance': 0,
                'team_quali_strength': quali_features['qualifying_position'].mean(),
                'quali_vs_team': 0,
                'quali_momentum': 0,
                'circuit_experience': 0,
                'circuit_consistency': 0.5,
                'group_consistency': 0.5
            })
            
            return quali_features
            
        except Exception as e:
            logger.error(f"Error processing qualifying data: {str(e)}")
            return pd.DataFrame()
        
    def process_race_data(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """Process race data to create relevant features."""
        logger.info("Processing race data")
        
        required_columns = ['year', 'round', 'driver', 'team', 'position', 'circuit', 'lap_time']
        if not self._validate_dataframe(race_data, required_columns, "Race data"):
            return pd.DataFrame()
            
        try:
            # Handle missing or invalid positions
            race_data['position'] = pd.to_numeric(race_data['position'], errors='coerce')
            
            # Calculate driver form (last 3 races)
            driver_form = race_data.sort_values(['driver', 'year', 'round']).groupby('driver').rolling(
                window=3, min_periods=1
            )['position'].agg(['mean', 'std']).reset_index()
            
            driver_form.columns = ['driver', 'index', 'recent_pos_mean', 'recent_pos_std']
            
            # Calculate team form
            team_form = race_data.sort_values(['team', 'year', 'round']).groupby('team').rolling(
                window=3, min_periods=1
            )['position'].agg(['mean', 'std']).reset_index()
            
            team_form.columns = ['team', 'index', 'team_pos_mean', 'team_pos_std']
            
            # Calculate race momentum
            race_data['race_momentum'] = race_data.sort_values(['driver', 'year', 'round']).groupby('driver')['position'].transform(
                lambda x: x.diff().rolling(3, min_periods=1).mean()
            )
            
            # Calculate circuit experience
            circuit_experience = race_data.groupby(['driver', 'circuit'])['position'].count().reset_index()
            circuit_experience = circuit_experience.rename(columns={'position': 'race_circuit_experience'})
            
            # Calculate position group performance
            try:
                race_data['position_group'] = pd.qcut(
                    race_data['position'],
                    q=4,
                    labels=['front', 'upper_mid', 'lower_mid', 'back']
                )
            except ValueError as e:
                logger.warning(f"Error in position group calculation: {str(e)}. Using alternative grouping.")
                race_data['position_group'] = pd.cut(
                    race_data['position'],
                    bins=[0, 3, 7, 12, float('inf')],
                    labels=['front', 'upper_mid', 'lower_mid', 'back']
                )
            
            group_performance = race_data.groupby(['driver', 'position_group'])['position'].agg(['mean', 'std']).reset_index()
            group_performance['std'] = group_performance['std'].fillna(0)
            group_performance['race_group_consistency'] = 1 / (1 + group_performance['std'])
            
            # Calculate gaps to cars ahead and behind
            race_data = race_data.sort_values(['year', 'round', 'position'])
            race_data['gap_to_car_ahead'] = race_data.groupby(['year', 'round'])['lap_time'].diff()
            race_data['gap_to_car_behind'] = race_data.groupby(['year', 'round'])['lap_time'].diff(-1)
            
            # Merge form features back to race data
            race_features = race_data.merge(
                driver_form[['driver', 'recent_pos_mean', 'recent_pos_std']],
                on='driver',
                how='left'
            ).merge(
                team_form[['team', 'team_pos_mean', 'team_pos_std']],
                on='team',
                how='left'
            ).merge(
                circuit_experience,
                on=['driver', 'circuit'],
                how='left'
            ).merge(
                group_performance[['driver', 'position_group', 'mean', 'race_group_consistency']],
                on=['driver', 'position_group'],
                how='left'
            )
            
            # Handle potential division by zero in consistency calculations
            race_features['recent_pos_std'] = race_features['recent_pos_std'].fillna(0)
            race_features['race_consistency'] = 1 / (1 + race_features['recent_pos_std'])
            
            # Calculate position stability
            position_stability = race_data.groupby('driver')['position'].agg([
                lambda x: (x.diff() == 0).mean(),
                lambda x: (x.diff() > 0).mean(),
                lambda x: (x.diff() < 0).mean()
            ]).reset_index()
            
            position_stability.columns = ['driver', 'position_stability', 'position_gain_rate', 'position_loss_rate']
            
            # Merge position stability features
            race_features = race_features.merge(
                position_stability,
                on='driver',
                how='left'
            )
            
            # Fill any remaining NaN values with appropriate defaults
            race_features = race_features.fillna({
                'gap_to_car_ahead': 0,
                'gap_to_car_behind': 0,
                'race_momentum': 0,
                'race_circuit_experience': 0,
                'race_consistency': 0.5,
                'race_group_consistency': 0.5,
                'position_stability': 0.5,
                'position_gain_rate': 0.25,
                'position_loss_rate': 0.25
            })
            
            return race_features
            
        except Exception as e:
            logger.error(f"Error processing race data: {str(e)}")
            return pd.DataFrame()
        
    def run_feature_engineering(
        self,
        race_file: str,
        qualifying_file: Optional[str] = None,
        weather_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Run the complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline")
        
        try:
            # Load race data
            if not Path(race_file).exists():
                logger.error(f"Race file not found: {race_file}")
                return pd.DataFrame()
                
            race_data = pd.read_parquet(race_file)
            
            # Process race data
            race_features = self.process_race_data(race_data)
            if race_features.empty:
                logger.error("Failed to process race data")
                return pd.DataFrame()
            
            # Process qualifying data if available
            if qualifying_file:
                if not Path(qualifying_file).exists():
                    logger.warning(f"Qualifying file not found: {qualifying_file}")
                    features = race_features
                else:
                    qualifying_data = pd.read_parquet(qualifying_file)
                    quali_features = self.process_qualifying_data(qualifying_data)
                    
                    if not quali_features.empty:
                        # Merge qualifying features with race features
                        features = race_features.merge(
                            quali_features,
                            on=['year', 'round', 'driver', 'team'],
                            how='left'
                        )
                    else:
                        logger.warning("No qualifying features generated, using only race features")
                        features = race_features
            else:
                features = race_features
                
            # Process weather data if available
            if weather_file:
                if not Path(weather_file).exists():
                    logger.warning(f"Weather file not found: {weather_file}")
                else:
                    weather_data = pd.read_parquet(weather_file)
                    # Add weather processing logic here
                    features = features.merge(
                        weather_data,
                        on=['year', 'round'],
                        how='left'
                    )
                    
            # Scale numerical features
            scaler = StandardScaler()
            numerical_features = features.select_dtypes(include=[np.number]).columns
            features[numerical_features] = scaler.fit_transform(features[numerical_features])
            
            # Save processed features
            output_file = self.features_dir / 'processed_features.parquet'
            features.to_parquet(output_file)
            
            logger.info(f"Feature engineering completed. Features saved to {output_file}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    fe = FeatureEngineering()
    features = fe.run_feature_engineering(
        race_file='data/processed/processed_2023.parquet',
        qualifying_file='data/processed/qualifying_2023.parquet'
    ) 