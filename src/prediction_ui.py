import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")

from src.transform.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1PredictionUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("F1 Race Prediction")
        self.window.geometry("1000x800")
        
        # Set dark mode theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure dark mode colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#e0e0e0"
        self.accent_color = "#007acc"
        self.highlight_color = "#3c3c3c"
        self.text_color = "#ffffff"
        
        # Apply dark mode colors
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("TButton", background=self.accent_color, foreground=self.text_color)
        self.style.configure("TEntry", fieldbackground=self.highlight_color, foreground=self.text_color)
        self.style.configure("TCombobox", fieldbackground=self.highlight_color, foreground=self.text_color)
        self.style.configure("TLabelframe", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("TLabelframe.Label", background=self.bg_color, foreground=self.fg_color)
        
        # Configure the main window
        self.window.configure(bg=self.bg_color)
        
        # Initialize components
        self.fe = FeatureEngineering()
        self.model = None
        self.load_model()
        self.load_data()
        
        # Create UI elements
        self.create_widgets()
        
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = Path('models/race_prediction_model.joblib')
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model file not found. Using dummy predictions.")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
            
    def load_data(self):
        """Load and process necessary data"""
        try:
            # Load latest race and qualifying data
            self.features = self.fe.run_feature_engineering(
                race_file='data/processed/processed_2023.parquet',
                qualifying_file='data/processed/qualifying_2023.parquet'
            )
            
            # Get unique values for dropdowns
            self.circuits = sorted(self.features['circuit'].unique())
            self.drivers = sorted(self.features['driver'].unique())
            self.teams = sorted(self.features['team'].unique())
            
            # Load weather data if available
            weather_file = 'data/processed/weather_2023.parquet'
            if Path(weather_file).exists():
                self.weather_data = pd.read_parquet(weather_file)
                self.weather_conditions = sorted(self.weather_data['condition'].unique())
            else:
                self.weather_data = None
                self.weather_conditions = ['Dry', 'Wet', 'Mixed']
                
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.circuits = []
            self.drivers = []
            self.teams = []
            self.weather_conditions = ['Dry', 'Wet', 'Mixed']
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="F1 Race Prediction", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Race Information Frame
        race_frame = ttk.LabelFrame(main_frame, text="Race Information", padding=10)
        race_frame.pack(fill="x", padx=10, pady=5)
        
        # Circuit Selection
        ttk.Label(race_frame, text="Circuit:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.circuit_var = tk.StringVar()
        circuit_dropdown = ttk.Combobox(race_frame, textvariable=self.circuit_var, width=30)
        circuit_dropdown['values'] = self.circuits
        circuit_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Weather Selection
        ttk.Label(race_frame, text="Weather:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.weather_var = tk.StringVar(value="Dry")
        weather_dropdown = ttk.Combobox(race_frame, textvariable=self.weather_var, width=15)
        weather_dropdown['values'] = self.weather_conditions
        weather_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Temperature
        ttk.Label(race_frame, text="Temperature (°C):").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.temp_var = tk.StringVar(value="25")
        temp_entry = ttk.Entry(race_frame, textvariable=self.temp_var, width=5)
        temp_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        
        # Qualifying Results Frame
        quali_frame = ttk.LabelFrame(main_frame, text="Qualifying Results", padding=10)
        quali_frame.pack(fill="x", padx=10, pady=5)
        
        # Create a canvas with scrollbar for qualifying entries
        canvas = tk.Canvas(quali_frame, bg=self.bg_color, highlightthickness=0)
        scrollbar = ttk.Scrollbar(quali_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout for qualifying entries
        ttk.Label(scrollable_frame, text="Driver").grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(scrollable_frame, text="Team").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(scrollable_frame, text="Position").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(scrollable_frame, text="Time (s)").grid(row=0, column=3, padx=5, pady=2)
        
        # Driver entries
        self.driver_entries = {}
        self.team_entries = {}
        self.quali_pos_entries = {}
        self.quali_time_entries = {}
        
        for i, driver in enumerate(self.drivers):
            row = i + 1
            
            # Find team for this driver
            driver_team = self.features[self.features['driver'] == driver]['team'].iloc[0] if not self.features.empty else ""
            
            # Driver name (read-only)
            driver_label = ttk.Label(scrollable_frame, text=driver)
            driver_label.grid(row=row, column=0, padx=5, pady=2, sticky="w")
            
            # Team dropdown
            team_var = tk.StringVar(value=driver_team)
            team_dropdown = ttk.Combobox(scrollable_frame, textvariable=team_var, width=15)
            team_dropdown['values'] = self.teams
            team_dropdown.grid(row=row, column=1, padx=5, pady=2)
            self.team_entries[driver] = team_var
            
            # Qualifying position
            pos_var = tk.StringVar()
            pos_entry = ttk.Entry(scrollable_frame, textvariable=pos_var, width=5)
            pos_entry.grid(row=row, column=2, padx=5, pady=2)
            self.quali_pos_entries[driver] = pos_var
            
            # Qualifying time
            time_var = tk.StringVar()
            time_entry = ttk.Entry(scrollable_frame, textvariable=time_var, width=8)
            time_entry.grid(row=row, column=3, padx=5, pady=2)
            self.quali_time_entries[driver] = time_var
            
            self.driver_entries[driver] = {
                'team': team_var,
                'position': pos_var,
                'time': time_var
            }
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Additional Race Factors Frame
        factors_frame = ttk.LabelFrame(main_frame, text="Additional Race Factors", padding=10)
        factors_frame.pack(fill="x", padx=10, pady=5)
        
        # Tire degradation
        ttk.Label(factors_frame, text="Tire Degradation:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.tire_var = tk.StringVar(value="Medium")
        tire_dropdown = ttk.Combobox(factors_frame, textvariable=self.tire_var, width=15)
        tire_dropdown['values'] = ["Low", "Medium", "High"]
        tire_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Track evolution
        ttk.Label(factors_frame, text="Track Evolution:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.track_var = tk.StringVar(value="Medium")
        track_dropdown = ttk.Combobox(factors_frame, textvariable=self.track_var, width=15)
        track_dropdown['values'] = ["Low", "Medium", "High"]
        track_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Safety car probability
        ttk.Label(factors_frame, text="Safety Car Probability:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.safety_car_var = tk.StringVar(value="Low")
        safety_car_dropdown = ttk.Combobox(factors_frame, textvariable=self.safety_car_var, width=15)
        safety_car_dropdown['values'] = ["Low", "Medium", "High"]
        safety_car_dropdown.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        
        # Predict Button
        predict_button = ttk.Button(main_frame, text="Predict Race Result", command=self.predict_race)
        predict_button.pack(pady=10)
        
        # Results Display
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Text results tab
        self.text_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text="Text Results")
        
        self.results_text = tk.Text(self.text_tab, height=10, width=50, bg=self.highlight_color, fg=self.text_color)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Chart tab
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="Visualization")
        
        # Confidence tab
        self.confidence_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.confidence_tab, text="Confidence Analysis")
        
    def prepare_prediction_data(self, circuit, qualifying_positions):
        """Prepare data for prediction"""
        try:
            # Create a DataFrame with the current race data
            current_race_data = []
            
            for driver, entries in self.driver_entries.items():
                try:
                    quali_pos = int(entries['position'].get()) if entries['position'].get() else None
                    quali_time = float(entries['time'].get()) if entries['time'].get() else None
                    
                    if quali_pos is not None:
                        # Get historical data for this driver at this circuit
                        driver_history = self.features[
                            (self.features['driver'] == driver) & 
                            (self.features['circuit'] == circuit)
                        ]
                        
                        # Get recent form (last 3 races)
                        recent_form = self.features[self.features['driver'] == driver].sort_values(
                            ['year', 'round'], ascending=False
                        ).head(3)
                        
                        # Create entry for this driver
                        entry = {
                            'driver': driver,
                            'team': entries['team'].get(),
                            'circuit': circuit,
                            'qualifying_position': quali_pos,
                            'qualifying_time': quali_time,
                            'weather': self.weather_var.get(),
                            'temperature': float(self.temp_var.get()),
                            'tire_degradation': self.tire_var.get(),
                            'track_evolution': self.track_var.get(),
                            'safety_car_probability': self.safety_car_var.get(),
                            
                            # Historical performance
                            'circuit_experience': len(driver_history),
                            'avg_quali_pos': driver_history['qualifying_position'].mean() if not driver_history.empty else None,
                            'best_quali_pos': driver_history['qualifying_position'].min() if not driver_history.empty else None,
                            'recent_pos_mean': recent_form['position'].mean() if not recent_form.empty else None,
                            'recent_pos_std': recent_form['position'].std() if not recent_form.empty else None,
                            
                            # Team performance
                            'team_pos_mean': self.features[self.features['team'] == entries['team'].get()]['position'].mean(),
                            'team_pos_std': self.features[self.features['team'] == entries['team'].get()]['position'].std(),
                        }
                        
                        current_race_data.append(entry)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing data for {driver}: {str(e)}")
                    continue
            
            # Convert to DataFrame
            prediction_data = pd.DataFrame(current_race_data)
            
            # Fill missing values
            prediction_data = prediction_data.fillna({
                'circuit_experience': 0,
                'avg_quali_pos': prediction_data['qualifying_position'].mean(),
                'best_quali_pos': prediction_data['qualifying_position'].min(),
                'recent_pos_mean': prediction_data['qualifying_position'].mean(),
                'recent_pos_std': 0,
                'team_pos_mean': prediction_data['qualifying_position'].mean(),
                'team_pos_std': 0
            })
            
            # Add derived features
            prediction_data['race_consistency'] = 1 / (1 + prediction_data['recent_pos_std'])
            prediction_data['team_consistency'] = 1 / (1 + prediction_data['team_pos_std'])
            
            # Calculate position group
            try:
                prediction_data['position_group'] = pd.qcut(
                    prediction_data['qualifying_position'],
                    q=4,
                    labels=['front', 'upper_mid', 'lower_mid', 'back']
                )
            except ValueError:
                prediction_data['position_group'] = pd.cut(
                    prediction_data['qualifying_position'],
                    bins=[0, 3, 7, 12, float('inf')],
                    labels=['front', 'upper_mid', 'lower_mid', 'back']
                )
            
            # Add weather encoding
            weather_encoding = {'Dry': 0, 'Wet': 1, 'Mixed': 0.5}
            prediction_data['weather_encoded'] = prediction_data['weather'].map(weather_encoding)
            
            # Add factor encodings
            factor_encoding = {'Low': 0, 'Medium': 0.5, 'High': 1}
            prediction_data['tire_degradation_encoded'] = prediction_data['tire_degradation'].map(factor_encoding)
            prediction_data['track_evolution_encoded'] = prediction_data['track_evolution'].map(factor_encoding)
            prediction_data['safety_car_probability_encoded'] = prediction_data['safety_car_probability'].map(factor_encoding)
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            return pd.DataFrame()
        
    def make_prediction(self, prediction_data):
        """Make prediction using the trained model"""
        try:
            if self.model is None:
                # Use a simple heuristic if no model is available
                logger.warning("No model available, using heuristic prediction")
                
                # Simple heuristic: combine qualifying position with historical performance
                prediction_data['predicted_position'] = (
                    prediction_data['qualifying_position'] * 0.6 + 
                    prediction_data['avg_quali_pos'] * 0.2 + 
                    prediction_data['recent_pos_mean'] * 0.2
                )
                
                # Add some randomness for positions
                prediction_data['predicted_position'] += np.random.normal(0, 0.5, len(prediction_data))
                
                # Ensure positions are positive integers
                prediction_data['predicted_position'] = prediction_data['predicted_position'].clip(lower=1)
                
                # Calculate confidence based on consistency
                prediction_data['confidence'] = (
                    prediction_data['race_consistency'] * 0.6 + 
                    prediction_data['team_consistency'] * 0.4
                )
                
                # Normalize confidence to 0-1 range
                prediction_data['confidence'] = prediction_data['confidence'].clip(0, 1)
                
            else:
                # Use the trained model
                # Select features for prediction
                feature_columns = [
                    'qualifying_position', 'circuit_experience', 'avg_quali_pos',
                    'best_quali_pos', 'recent_pos_mean', 'recent_pos_std',
                    'team_pos_mean', 'team_pos_std', 'race_consistency',
                    'team_consistency', 'weather_encoded', 'temperature',
                    'tire_degradation_encoded', 'track_evolution_encoded',
                    'safety_car_probability_encoded'
                ]
                
                # Make prediction
                X = prediction_data[feature_columns]
                prediction_data['predicted_position'] = self.model.predict(X)
                
                # Get prediction probabilities if available
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)
                    # Use the probability of the predicted class as confidence
                    prediction_data['confidence'] = np.max(probabilities, axis=1)
                else:
                    # Calculate confidence based on consistency
                    prediction_data['confidence'] = (
                        prediction_data['race_consistency'] * 0.6 + 
                        prediction_data['team_consistency'] * 0.4
                    )
            
            # Sort by predicted position
            prediction_data = prediction_data.sort_values('predicted_position')
            
            # Round predicted positions to integers
            prediction_data['predicted_position'] = prediction_data['predicted_position'].round().astype(int)
            
            # Format confidence as percentage
            prediction_data['confidence_pct'] = (prediction_data['confidence'] * 100).round(1)
            
            return prediction_data
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return pd.DataFrame()
        
    def predict_race(self):
        """Make predictions for the race"""
        try:
            # Get input values
            circuit = self.circuit_var.get()
            if not circuit:
                messagebox.showerror("Error", "Please select a circuit")
                return
                
            # Collect qualifying positions
            qualifying_positions = {}
            for driver, entries in self.driver_entries.items():
                try:
                    pos = int(entries['position'].get()) if entries['position'].get() else None
                    qualifying_positions[driver] = pos
                except ValueError:
                    continue
            
            # Check if we have enough data
            if len(qualifying_positions) < 5:
                messagebox.showerror("Error", "Please enter qualifying positions for at least 5 drivers")
                return
            
            # Create prediction input data
            prediction_data = self.prepare_prediction_data(circuit, qualifying_positions)
            
            if prediction_data.empty:
                messagebox.showerror("Error", "Failed to prepare prediction data")
                return
            
            # Make prediction
            predicted_results = self.make_prediction(prediction_data)
            
            if predicted_results.empty:
                messagebox.showerror("Error", "Failed to generate predictions")
                return
            
            # Display results
            self.display_results(predicted_results)
            
            # Create visualizations
            self.create_charts(predicted_results)
            
            # Show confidence analysis
            self.show_confidence_analysis(predicted_results)
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        
    def display_results(self, results):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Predicted Race Results for {self.circuit_var.get()}\n\n")
        
        # Display top 10 with confidence
        for i, (_, row) in enumerate(results.head(10).iterrows(), 1):
            position = self.get_position_text(i)
            self.results_text.insert(tk.END, 
                f"{position}: {row['driver']} ({row['team']}) - Confidence: {row['confidence_pct']}%\n")
        
        # Display qualifying vs predicted
        self.results_text.insert(tk.END, "\nQualifying vs Predicted Positions:\n")
        self.results_text.insert(tk.END, "Driver".ljust(15) + "Qual".ljust(8) + "Pred".ljust(8) + "Change\n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")
        
        for _, row in results.iterrows():
            quali_pos = row['qualifying_position']
            pred_pos = row['predicted_position']
            change = pred_pos - quali_pos
            change_text = f"+{change}" if change > 0 else str(change)
            
            self.results_text.insert(tk.END, 
                f"{row['driver'][:15].ljust(15)}{quali_pos}".ljust(8) + 
                f"{pred_pos}".ljust(8) + f"{change_text}\n")
    
    def get_position_text(self, position):
        """Convert position number to text (1st, 2nd, etc.)"""
        if position == 1:
            return "1st"
        elif position == 2:
            return "2nd"
        elif position == 3:
            return "3rd"
        else:
            return f"{position}th"
    
    def create_charts(self, results):
        """Create visualization charts"""
        # Clear previous charts
        for widget in self.chart_tab.winfo_children():
            widget.destroy()
        
        # Create figure with dark background
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        fig.patch.set_facecolor(self.bg_color)
        
        # Plot 1: Top 10 predicted positions
        top_10 = results.head(10)
        drivers = top_10['driver']
        positions = top_10['predicted_position']
        confidences = top_10['confidence']
        
        bars = ax1.barh(drivers, positions, color=self.accent_color, alpha=0.7)
        
        # Add confidence as text
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f"{confidences[i]*100:.1f}%", 
                    va='center', color=self.text_color)
        
        ax1.set_xlabel('Position', color=self.text_color)
        ax1.set_ylabel('Driver', color=self.text_color)
        ax1.set_title('Top 10 Predicted Finishers', color=self.text_color)
        ax1.set_facecolor(self.bg_color)
        ax1.tick_params(colors=self.text_color)
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Plot 2: Qualifying vs Predicted positions
        quali_pos = results['qualifying_position']
        pred_pos = results['predicted_position']
        drivers = results['driver']
        
        x = np.arange(len(drivers))
        width = 0.35
        
        ax2.bar(x - width/2, quali_pos, width, label='Qualifying', color='#4CAF50', alpha=0.7)
        ax2.bar(x + width/2, pred_pos, width, label='Predicted', color='#2196F3', alpha=0.7)
        
        ax2.set_xlabel('Driver', color=self.text_color)
        ax2.set_ylabel('Position', color=self.text_color)
        ax2.set_title('Qualifying vs Predicted Positions', color=self.text_color)
        ax2.set_xticks(x)
        ax2.set_xticklabels(drivers, rotation=45, ha='right')
        ax2.legend()
        ax2.set_facecolor(self.bg_color)
        ax2.tick_params(colors=self.text_color)
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Add to UI
        canvas = FigureCanvasTkAgg(fig, master=self.chart_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def show_confidence_analysis(self, results):
        """Show confidence analysis"""
        # Clear previous content
        for widget in self.confidence_tab.winfo_children():
            widget.destroy()
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(self.bg_color)
        
        # Plot confidence distribution
        confidences = results['confidence']
        drivers = results['driver']
        
        bars = ax.barh(drivers, confidences, color=self.accent_color, alpha=0.7)
        
        # Add confidence percentage as text
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{confidences[i]*100:.1f}%", 
                    va='center', color=self.text_color)
        
        ax.set_xlabel('Confidence', color=self.text_color)
        ax.set_ylabel('Driver', color=self.text_color)
        ax.set_title('Prediction Confidence by Driver', color=self.text_color)
        ax.set_facecolor(self.bg_color)
        ax.tick_params(colors=self.text_color)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Add to UI
        canvas = FigureCanvasTkAgg(fig, master=self.confidence_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = F1PredictionUI()
    app.run() 