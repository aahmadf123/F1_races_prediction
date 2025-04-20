import os
import logging
from datetime import datetime
from pathlib import Path
import fastf1
import requests
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataIngestion:
    def __init__(self):
        self.data_dir = Path(os.getenv('DATA_PATH', 'data'))
        self.raw_dir = self.data_dir / 'raw'
        self.interim_dir = self.data_dir / 'interim'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.interim_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize FastF1
        fastf1.Cache.enable_cache(str(self.raw_dir / 'cache'))
        
    def fetch_ergast_data(self, year: int) -> pd.DataFrame:
        """Fetch race data from Ergast API."""
        url = f"http://ergast.com/api/f1/{year}/results.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Save raw data
            raw_file = self.raw_dir / f'ergast_{year}.json'
            with open(raw_file, 'w') as f:
                f.write(response.text)
                
            # Process and return DataFrame
            races = data['MRData']['RaceTable']['Races']
            processed_data = []
            
            for race in races:
                race_data = {
                    'year': year,
                    'round': race['round'],
                    'race_name': race['raceName'],
                    'date': race['date'],
                    'circuit': race['Circuit']['circuitName'],
                    'country': race['Circuit']['Location']['country'],
                    'results': race['Results']
                }
                processed_data.append(race_data)
                
            return pd.DataFrame(processed_data)
            
        except Exception as e:
            logger.error(f"Error fetching Ergast data for {year}: {str(e)}")
            raise
            
    def fetch_fastf1_data(self, year: int, session_type: str = 'R') -> pd.DataFrame:
        """Fetch detailed timing data from FastF1."""
        try:
            session = fastf1.get_session(year, 1, session_type)
            session.load()
            
            # Get lap times
            lap_times = session.laps
            
            # Save raw data
            raw_file = self.raw_dir / f'fastf1_{year}_{session_type}.parquet'
            lap_times.to_parquet(raw_file)
            
            return lap_times
            
        except Exception as e:
            logger.error(f"Error fetching FastF1 data for {year}: {str(e)}")
            raise
            
    def fetch_weather_data(self, lat: float, lon: float, date: str) -> Dict:
        """Fetch weather data from OpenWeatherMap API."""
        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            raise ValueError("OpenWeatherMap API key not found in environment variables")
            
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'date': date
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Save raw data
            raw_file = self.raw_dir / f'weather_{date}.json'
            with open(raw_file, 'w') as f:
                f.write(response.text)
                
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            raise
            
    def run_ingestion_pipeline(self, years: List[int]):
        """Run the complete data ingestion pipeline."""
        logger.info("Starting data ingestion pipeline")
        
        for year in years:
            try:
                # Fetch data from all sources
                ergast_data = self.fetch_ergast_data(year)
                fastf1_data = self.fetch_fastf1_data(year)
                
                # Save processed data
                processed_file = self.processed_dir / f'processed_{year}.parquet'
                ergast_data.to_parquet(processed_file)
                
                logger.info(f"Successfully processed data for {year}")
                
            except Exception as e:
                logger.error(f"Error processing data for {year}: {str(e)}")
                continue
                
        logger.info("Data ingestion pipeline completed")

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    years = list(range(2018, datetime.now().year + 1))
    ingestion.run_ingestion_pipeline(years) 