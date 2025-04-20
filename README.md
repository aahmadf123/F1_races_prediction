# F1 Race Winner Prediction Pipeline

A production-grade machine learning pipeline for predicting Formula 1 race winners using advanced ML techniques and real-time data.

## Overview

This project implements a robust prediction system for Formula 1 race outcomes, incorporating:
- Real-time data ingestion from multiple sources (Ergast, FastF1, Weather APIs)
- Advanced feature engineering and feature store
- Ensemble machine learning models with uncertainty quantification
- MLOps best practices and monitoring
- Production-ready API deployment

## Project Structure

```
f1_winner_predictions/
├── data/               # Data storage (raw, interim, processed)
├── features/           # Feature store definitions
├── models/            # Model registry
├── notebooks/         # EDA and prototyping
├── src/              # Source code
├── tests/            # Test suite
├── dashboards/       # Monitoring dashboards
└── scripts/          # Utility scripts
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv f1env
source f1env/bin/activate  # On Windows: f1env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Data Ingestion
```bash
python src/ingestion/fetch_data.py
```

### Model Training
```bash
python src/training/train.py
```

### API Server
```bash
python src/deployment/api.py
```

### Dashboard
```bash
streamlit run dashboards/main.py
```

## Development

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation as needed
- Use conventional commits

## License

This project is licensed under the MIT License - see the LICENSE file for details.
