# app.py - Enhanced version with Weather API and ML model
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import requests
import traceback
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KDTree

app = Flask(__name__)
app.secret_key = 'traffic_safety_advanced_predictor_key'

# OpenWeatherMap API key - replace with your actual key
OPENWEATHER_API_KEY = "8dad3db309e50de33c8cdefbe69cec74"  # Replace with your OpenWeatherMap API key

# Global variables for models and data
accident_model = None
weather_model = None
merged_data = None
location_tree = None
date_columns = None
scaler = None
feature_columns = None
categorical_columns = None
numerical_columns = None

# Path to save trained models
MODEL_DIR = 'models'


# Custom JSON encoder for handling non-serializable types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return int(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(CustomJSONEncoder, self).default(obj)


# Use custom JSON encoder for Flask
app.json_encoder = CustomJSONEncoder


def load_data():
    """Load the merged traffic accident data and prepare models"""
    global merged_data, location_tree, date_columns, accident_model, weather_model
    global feature_columns, categorical_columns, numerical_columns, scaler

    try:
        print("Loading merged traffic data...")

        # Check if data file exists
        if os.path.exists('merged_traffic_accident_data.csv'):
            merged_data = pd.read_csv('merged_traffic_accident_data.csv', low_memory=False)
            print(f"Loaded {len(merged_data)} records from merged data")

            # Check for location columns
            if 'accident_lat' in merged_data.columns and 'accident_lon' in merged_data.columns:
                # Create spatial index for fast queries
                coords = merged_data[['accident_lat', 'accident_lon']].dropna().values
                if len(coords) > 0:
                    location_tree = KDTree(coords)
                    print(f"Created spatial index with {len(coords)} points")

            # Identify date columns
            date_columns = [col for col in merged_data.columns if 'date' in col.lower() or 'datum' in col.lower()]
            for col in date_columns:
                try:
                    merged_data[f"{col}_dt"] = pd.to_datetime(merged_data[col], errors='coerce')
                    print(f"Converted {col} to datetime")
                except:
                    pass

            # Load or train models
            load_or_train_models()

        else:
            print("Warning: merged_traffic_accident_data.csv not found!")
            print("Creating dummy data for testing...")

            # Create dummy data
            create_dummy_data()

            # Train simple models on dummy data
            load_or_train_models()

        print("Data and model initialization complete")

    except Exception as e:
        print(f"Error in data loading: {e}")
        print(traceback.format_exc())

        # Create dummy data as fallback
        create_dummy_data()

        # Train simple models on dummy data
        load_or_train_models()


def create_dummy_data():
    """Create dummy data for testing when real data is unavailable"""
    global merged_data, location_tree, date_columns

    # Create dummy data centered around Brno
    num_records = 1000
    center_lat, center_lon = 49.1951, 16.6068  # Brno coordinates

    # Generate random dates between 2018-2023
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = (end_date - start_date).days

    # Random dates
    np.random.seed(42)  # For reproducibility
    random_dates = [start_date + timedelta(days=np.random.randint(0, date_range)) for _ in range(num_records)]

    # Random coordinates around center
    random_lats = [center_lat + np.random.uniform(-0.05, 0.05) for _ in range(num_records)]
    random_lons = [center_lon + np.random.uniform(-0.05, 0.05) for _ in range(num_records)]

    # Random weather and traffic data
    temp_avg_values = [np.random.uniform(-10, 30) for _ in range(num_records)]
    rain_values = [np.random.uniform(0, 20) for _ in range(num_records)]
    car_counts = [np.random.randint(500, 10000) for _ in range(num_records)]
    truck_counts = [np.random.randint(50, 1000) for _ in range(num_records)]

    # Road conditions (categorical)
    road_conditions = np.random.choice(['dry', 'wet', 'snow', 'ice'], size=num_records)

    # Create target variable (accident severity - binary for simplicity)
    # More severe accidents in bad weather and high traffic
    severity_scores = []
    for i in range(num_records):
        base_score = 0
        # Higher in extreme temperatures
        if temp_avg_values[i] < 0 or temp_avg_values[i] > 25:
            base_score += 0.2
        # Higher in rain
        if rain_values[i] > 5:
            base_score += 0.3
        # Higher in bad road conditions
        if road_conditions[i] in ['snow', 'ice', 'wet']:
            base_score += 0.25
        # Higher in high traffic
        if car_counts[i] > 5000 or truck_counts[i] > 500:
            base_score += 0.15

        # Add randomness
        base_score += np.random.uniform(-0.2, 0.2)
        severity_scores.append(min(1, max(0, base_score)))

    # Convert to binary (severe vs not severe)
    accident_severity = [1 if score > 0.5 else 0 for score in severity_scores]

    # Create DataFrame
    merged_data = pd.DataFrame({
        'accident_date': random_dates,
        'accident_lat': random_lats,
        'accident_lon': random_lons,
        'temp_avg': temp_avg_values,
        'rain_cumulative': rain_values,
        'car_count': car_counts,
        'truck_count': truck_counts,
        'road_condition': road_conditions,
        'accident_severity': accident_severity,
        'severity_score': severity_scores  # Continuous severity score for regression
    })

    # Add datetime column
    merged_data['accident_date_dt'] = merged_data['accident_date']
    date_columns = ['accident_date']

    # Create spatial index
    coords = merged_data[['accident_lat', 'accident_lon']].values
    location_tree = KDTree(coords)

    print(f"Created dummy dataset with {len(merged_data)} records")


def load_or_train_models():
    """Load existing models or train new ones"""
    global accident_model, weather_model, merged_data, feature_columns
    global categorical_columns, numerical_columns, scaler

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Define model file paths
    accident_model_path = os.path.join(MODEL_DIR, 'accident_model.pkl')
    weather_model_path = os.path.join(MODEL_DIR, 'weather_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

    # Try to load existing models
    models_loaded = False

    try:
        if os.path.exists(accident_model_path) and os.path.exists(weather_model_path) and os.path.exists(scaler_path):
            print("Loading existing models...")
            accident_model = joblib.load(accident_model_path)
            weather_model = joblib.load(weather_model_path)
            scaler = joblib.load(scaler_path)

            # Load feature columns from model
            try:
                feature_columns = accident_model.feature_names_in_.tolist()
                models_loaded = True
                print("Models loaded successfully")
            except:
                # If model doesn't have feature_names_in_ attribute
                print("Could not retrieve feature names from model")
    except Exception as e:
        print(f"Error loading models: {e}")

    # Train models if not loaded
    if not models_loaded:
        print("Training new models...")

        # Identify potential feature columns
        if merged_data is not None:
            # Identify numerical columns
            numerical_columns = merged_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Remove target columns and spatial coordinates from features
            exclude_cols = ['accident_severity', 'severity_score',
                            'accident_lat', 'accident_lon']
            numerical_columns = [col for col in numerical_columns
                                 if col not in exclude_cols and not col.startswith('accident_')]

            # Identify categorical columns
            categorical_columns = merged_data.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_columns = [col for col in categorical_columns
                                   if col not in exclude_cols and not col.startswith('accident_')]

            # Combined feature columns
            feature_columns = numerical_columns + categorical_columns

            # Fill missing values for training
            training_data = merged_data.copy()
            for col in numerical_columns:
                if col in training_data.columns:
                    training_data[col] = training_data[col].fillna(training_data[col].median())

            for col in categorical_columns:
                if col in training_data.columns:
                    training_data[col] = training_data[col].fillna(training_data[col].mode()[0])

            # Create X and y for training
            # For accident severity prediction
            if 'accident_severity' in training_data.columns:
                X = training_data[feature_columns]
                y_cls = training_data['accident_severity']

                # Create preprocessing pipeline
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ])

                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ])

                # Create a column transformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numerical_columns),
                        ('cat', categorical_transformer, categorical_columns)
                    ])

                # Create and train models
                accident_model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ])

                # Train accident severity model
                try:
                    accident_model.fit(X, y_cls)
                    joblib.dump(accident_model, accident_model_path)
                    print("Trained and saved accident severity model")
                except Exception as e:
                    print(f"Error training accident model: {e}")
                    # Create a dummy model
                    accident_model = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
                    ])
                    X_dummy = X.head(100)
                    y_dummy = y_cls.head(100)
                    accident_model.fit(X_dummy, y_dummy)
                    joblib.dump(accident_model, accident_model_path)
                    print("Created fallback accident model")

            # For weather prediction
            if 'temp_avg' in training_data.columns:
                # Select features for weather prediction
                weather_features = [col for col in feature_columns
                                    if col not in ['temp_avg', 'rain_cumulative', 'wind_speed', 'humidity']]

                if weather_features:
                    X_weather = training_data[weather_features]
                    y_weather = training_data[['temp_avg']]

                    # Weather model preprocessor
                    weather_preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, [col for col in numerical_columns if col in weather_features]),
                            ('cat', categorical_transformer,
                             [col for col in categorical_columns if col in weather_features])
                        ])

                    # Create weather model
                    weather_model = Pipeline(steps=[
                        ('preprocessor', weather_preprocessor),
                        ('regressor', GradientBoostingRegressor(n_estimators=50, random_state=42))
                    ])

                    # Train weather model
                    try:
                        weather_model.fit(X_weather, y_weather)
                        joblib.dump(weather_model, weather_model_path)
                        print("Trained and saved weather prediction model")
                    except Exception as e:
                        print(f"Error training weather model: {e}")
                        # Create dummy weather model
                        weather_model = Pipeline(steps=[
                            ('preprocessor', weather_preprocessor),
                            ('regressor', GradientBoostingRegressor(n_estimators=10, random_state=42))
                        ])
                        X_weather_dummy = X_weather.head(100)
                        y_weather_dummy = y_weather.head(100)
                        weather_model.fit(X_weather_dummy, y_weather_dummy)
                        joblib.dump(weather_model, weather_model_path)
                        print("Created fallback weather model")

            # Save the scaler separately for reuse
            if 'preprocessor' in accident_model.named_steps:
                try:
                    scaler = accident_model.named_steps['preprocessor']
                    joblib.dump(scaler, scaler_path)
                    print("Saved data preprocessor")
                except Exception as e:
                    print(f"Error saving preprocessor: {e}")


def get_real_weather_forecast(lat, lon, date_str):
    """Get real weather forecast from OpenWeatherMap API"""
    try:
        # Parse the input date
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
        current_date = datetime.now().date()

        # Get days difference
        days_diff = (forecast_date.date() - current_date).days

        # Check if date is in the future and within forecast range (5 days)
        if 0 <= days_diff <= 5:
            # Use OpenWeatherMap 5-day forecast API
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"

            response = requests.get(url)
            if response.status_code == 200:
                forecast_data = response.json()

                # Find forecast entries closest to the requested date
                target_date_str = forecast_date.strftime('%Y-%m-%d')
                matching_forecasts = [item for item in forecast_data['list']
                                      if item['dt_txt'].startswith(target_date_str)]

                if matching_forecasts:
                    # Calculate daily averages and extremes
                    temps = [item['main']['temp'] for item in matching_forecasts]
                    temp_min = min([item['main']['temp_min'] for item in matching_forecasts])
                    temp_max = max([item['main']['temp_max'] for item in matching_forecasts])
                    temp_avg = sum(temps) / len(temps)

                    humidities = [item['main']['humidity'] for item in matching_forecasts]
                    humidity = sum(humidities) / len(humidities)

                    wind_speeds = [item['wind']['speed'] for item in matching_forecasts]
                    wind_speed = sum(wind_speeds) / len(wind_speeds)

                    # Determine weather condition (use the afternoon forecast if available)
                    afternoon_forecasts = [item for item in matching_forecasts if '12:00' in item['dt_txt']]
                    weather_item = afternoon_forecasts[0] if afternoon_forecasts else matching_forecasts[0]
                    weather_condition = weather_item['weather'][0]['main'].lower()

                    # Convert API weather condition to our internal format
                    condition_mapping = {
                        'clear': 'clear',
                        'clouds': 'clear',  # Mostly clear
                        'rain': 'light_rain',
                        'drizzle': 'light_rain',
                        'thunderstorm': 'heavy_rain',
                        'snow': 'snow',
                        'mist': 'fog',
                        'fog': 'fog',
                        'haze': 'fog'
                    }

                    condition = condition_mapping.get(weather_condition, 'clear')

                    # Calculate precipitation
                    precipitation = 0
                    for item in matching_forecasts:
                        if 'rain' in item and '3h' in item['rain']:
                            precipitation += item['rain']['3h']

                    # Check for snow
                    is_snow = any('snow' in item for item in matching_forecasts)

                    return {
                        'temp_avg': round(temp_avg, 1),
                        'temp_min': round(temp_min, 1),
                        'temp_max': round(temp_max, 1),
                        'humidity': round(humidity),
                        'wind_speed': round(wind_speed, 1),
                        'precipitation': round(precipitation, 1),
                        'condition': condition,
                        'is_snow': 1 if is_snow else 0,
                        'source': 'openweathermap'
                    }

        # Fallback to model-based prediction if we can't get real forecast
        print("Falling back to model-based weather prediction")
        return get_model_weather_forecast(lat, lon, date_str)

    except Exception as e:
        print(f"Error getting real weather forecast: {e}")
        print(traceback.format_exc())
        return get_model_weather_forecast(lat, lon, date_str)


def get_model_weather_forecast(lat, lon, date_str):
    """Generate weather forecast using historical data and/or prediction model"""
    try:
        # Parse the input date
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
        month = forecast_date.month - 1  # 0-indexed
        day_of_week = forecast_date.weekday()  # 0=Monday, 6=Sunday

        # If we have a trained weather model, use it
        if weather_model is not None and merged_data is not None:
            # Create a feature set for prediction
            X = pd.DataFrame({
                'month': [month],
                'day_of_week': [day_of_week],
                'lat': [lat],
                'lon': [lon]
            })

            # Try model prediction
            try:
                # Predict temperature
                temp_avg = weather_model.predict(X)[0]

                # Add randomness to other weather factors based on month
                # Temperature variation
                temp_range = 8 if month in [11, 0, 1, 2] else 6  # Wider range in winter
                temp_min = temp_avg - np.random.uniform(1, temp_range / 2)
                temp_max = temp_avg + np.random.uniform(1, temp_range / 2)

                # Precipitation probability by month
                precip_prob = [0.4, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4][month]
                precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0

                # Snow condition
                is_snow = precipitation > 0 and temp_avg < 2 and month in [11, 0, 1, 2]

                # Humidity
                base_humidity = [80, 75, 70, 65, 60, 65, 65, 65, 70, 75, 80, 80][month]
                humidity = min(95, max(30, base_humidity + np.random.randint(-15, 16)))

                # Wind speed
                wind_speed = np.random.gamma(shape=2, scale=2)

                # Determine weather condition
                if is_snow:
                    condition = 'snow'
                elif precipitation > 10:
                    condition = 'heavy_rain'
                elif precipitation > 0:
                    condition = 'light_rain'
                elif humidity > 90:
                    condition = 'fog'
                elif wind_speed > 10:
                    condition = 'windy'
                else:
                    condition = 'clear'

                return {
                    'temp_avg': round(float(temp_avg), 1),
                    'temp_max': round(float(temp_max), 1),
                    'temp_min': round(float(temp_min), 1),
                    'humidity': int(humidity),
                    'wind_speed': round(float(wind_speed), 1),
                    'precipitation': round(float(precipitation), 1),
                    'condition': condition,
                    'is_snow': 1 if is_snow else 0,
                    'source': 'model'
                }

            except Exception as e:
                print(f"Error using weather model: {e}")

        # Fallback to historical pattern simulation
        return get_historical_weather_pattern(lat, lon, date_str)

    except Exception as e:
        print(f"Error in model-based weather forecast: {e}")
        return get_historical_weather_pattern(lat, lon, date_str)


def get_historical_weather_pattern(lat, lon, date_str):
    """Generate weather based on historical patterns in the area"""
    try:
        # Get similar historical conditions
        similar_data = find_similar_conditions(lat, lon, date_str)

        # Parse the input date
        forecast_date = datetime.strptime(date_str, '%Y-%m-%d')
        month = forecast_date.month - 1  # 0-indexed

        # Seed random generator for consistent results
        seed = int(hash(f"{lat:.2f}_{lon:.2f}_{date_str}") % 10000000)
        np.random.seed(seed)

        # If we have historical weather data, use it as a basis
        if len(similar_data) > 0 and 'temp_avg' in similar_data.columns:
            # Get average weather values from similar conditions
            temp_avg = similar_data['temp_avg'].mean() if 'temp_avg' in similar_data.columns else None
            temp_min = similar_data['temp_min'].mean() if 'temp_min' in similar_data.columns else None
            temp_max = similar_data['temp_max'].mean() if 'temp_max' in similar_data.columns else None

            # Add some randomness
            if temp_avg is not None and not pd.isna(temp_avg):
                temp_avg += np.random.uniform(-2, 2)
            else:
                # Fallback temperature ranges by month
                temp_ranges = [
                    (-5, 3),  # Jan
                    (-3, 5),  # Feb
                    (0, 10),  # Mar
                    (5, 15),  # Apr
                    (10, 20),  # May
                    (15, 25),  # Jun
                    (17, 28),  # Jul
                    (16, 27),  # Aug
                    (12, 22),  # Sep
                    (7, 15),  # Oct
                    (2, 10),  # Nov
                    (-3, 5)  # Dec
                ]
                temp_min, temp_max = temp_ranges[month]
                temp_avg = (temp_min + temp_max) / 2

            # Get other weather parameters
            precipitation = similar_data[
                'rain_cumulative'].mean() if 'rain_cumulative' in similar_data.columns else None
            humidity = similar_data['humidity'].mean() if 'humidity' in similar_data.columns else None
            wind_speed = similar_data['wind_speed'].mean() if 'wind_speed' in similar_data.columns else None

            # Add randomness and handle missing values
            if precipitation is None or pd.isna(precipitation):
                precip_prob = [0.4, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4][month]
                precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0

            if humidity is None or pd.isna(humidity):
                base_humidity = [80, 75, 70, 65, 60, 65, 65, 65, 70, 75, 80, 80][month]
                humidity = min(95, max(30, base_humidity + np.random.randint(-15, 16)))

            if wind_speed is None or pd.isna(wind_speed):
                wind_speed = np.random.gamma(shape=2, scale=2)
        else:
            # No historical data, generate synthetic forecast
            # Temperature ranges by month
            temp_ranges = [
                (-5, 3),  # Jan
                (-3, 5),  # Feb
                (0, 10),  # Mar
                (5, 15),  # Apr
                (10, 20),  # May
                (15, 25),  # Jun
                (17, 28),  # Jul
                (16, 27),  # Aug
                (12, 22),  # Sep
                (7, 15),  # Oct
                (2, 10),  # Nov
                (-3, 5)  # Dec
            ]

            temp_min, temp_max = temp_ranges[month]
            temp_min += np.random.randint(-3, 4)
            temp_max += np.random.randint(-2, 4)
            temp_avg = (temp_min + temp_max) / 2 + np.random.uniform(-1, 1)

            # Precipitation probability by month
            precip_prob = [0.4, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2, 0.25, 0.3, 0.35, 0.4][month]
            precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0

            # Humidity
            base_humidity = [80, 75, 70, 65, 60, 65, 65, 65, 70, 75, 80, 80][month]
            humidity = min(95, max(30, base_humidity + np.random.randint(-15, 16)))

            # Wind speed
            wind_speed = np.random.gamma(shape=2, scale=2)

        # Determine weather condition based on parameters
        is_snow = precipitation > 0 and temp_avg < 2 and month in [11, 0, 1, 2]

        if is_snow:
            condition = 'snow'
        elif precipitation > 10:
            condition = 'heavy_rain'
        elif precipitation > 0:
            condition = 'light_rain'
        elif humidity > 90:
            condition = 'fog'
        elif wind_speed > 10:
            condition = 'windy'
        else:
            condition = 'clear'

        return {
            'temp_avg': round(float(temp_avg), 1),
            'temp_max': round(float(temp_max) if temp_max is not None and not pd.isna(temp_max) else temp_avg + 5, 1),
            'temp_min': round(float(temp_min) if temp_min is not None and not pd.isna(temp_min) else temp_avg - 5, 1),
            'humidity': int(float(humidity) if humidity is not None and not pd.isna(humidity) else 70),
            'wind_speed': round(float(wind_speed) if wind_speed is not None and not pd.isna(wind_speed) else 5, 1),
            'precipitation': round(
                float(precipitation) if precipitation is not None and not pd.isna(precipitation) else 0, 1),
            'condition': condition,
            'is_snow': 1 if is_snow else 0,
            'source': 'historical'
        }
    except Exception as e:
        print(f"Error in historical weather pattern: {e}")

        # Return default weather data
        return {
            'temp_avg': 15,
            'temp_max': 20,
            'temp_min': 10,
            'humidity': 70,
            'wind_speed': 5,
            'precipitation': 0,
            'condition': 'clear',
            'is_snow': 0,
            'source': 'default'
        }


def find_similar_conditions(lat, lon, date_str):
    """Find accidents with similar conditions to the input"""
    global merged_data, location_tree

    if merged_data is None or len(merged_data) == 0:
        return pd.DataFrame()

    try:
        # Convert input date
        input_date = datetime.strptime(date_str, '%Y-%m-%d')

        # Find the month and day of week
        month = input_date.month
        day_of_week = input_date.weekday()  # 0=Monday, 6=Sunday

        # Find nearby accidents using KDTree if available
        nearby_indices = []
        if location_tree is not None:
            # Find 50 nearest neighbors within 10km
            distances, indices = location_tree.query([[lat, lon]], k=50)

            # Filter to reasonable distance (roughly 10km)
            max_distance = 0.1  # Approximately 10km in degrees
            nearby_indices = [idx for i, idx in enumerate(indices[0]) if distances[0][i] < max_distance]

        # Fallback to manual search
        if not nearby_indices and merged_data is not None:
            # Fallback: manually find records within rough bounding box
            lat_min, lat_max = lat - 0.1, lat + 0.1
            lon_min, lon_max = lon - 0.1, lon + 0.1

            # Filter by bounding box if location columns exist
            if 'accident_lat' in merged_data.columns and 'accident_lon' in merged_data.columns:
                bbox_filter = (
                        (merged_data['accident_lat'] >= lat_min) &
                        (merged_data['accident_lat'] <= lat_max) &
                        (merged_data['accident_lon'] >= lon_min) &
                        (merged_data['accident_lon'] <= lon_max)
                )
                nearby_indices = merged_data[bbox_filter].index.tolist()[:50]

        # Get subset of data for nearby accidents
        similar_accidents = merged_data.iloc[nearby_indices].copy() if nearby_indices else merged_data.head(50).copy()

        # Try to filter by similar month and day of week if datetime columns exist
        for col in date_columns:
            dt_col = f"{col}_dt"
            if dt_col in similar_accidents.columns:
                # Add month and day of week columns
                similar_accidents[f"{col}_month"] = similar_accidents[dt_col].dt.month
                similar_accidents[f"{col}_day_of_week"] = similar_accidents[dt_col].dt.dayofweek

                # Weight records by similarity of month and day
                similar_accidents['time_similarity'] = (
                        (12 - abs(similar_accidents[f"{col}_month"] - month)) / 12 *
                        (7 - abs(similar_accidents[f"{col}_day_of_week"] - day_of_week)) / 7
                )

        return similar_accidents

    except Exception as e:
        print(f"Error finding similar conditions: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()


def calculate_risk_score(lat, lon, date_str, weather_data):
    """Calculate comprehensive risk score using ML model and rules"""
    try:
        # Get similar historical accidents
        similar_accidents = find_similar_conditions(lat, lon, date_str)

        # Parse date components
        risk_date = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_week = risk_date.weekday()  # 0=Monday, 6=Sunday
        month = risk_date.month - 1  # 0=January, 11=December
        year = risk_date.year

        # Try model-based prediction if available
        model_prediction = None
        model_confidence = 0

        if accident_model is not None:
            try:
                # Prepare feature data for prediction
                features = {
                    'lat': lat,
                    'lon': lon,
                    'month': month + 1,  # Convert back to 1-indexed
                    'day_of_week': day_of_week,
                    'year': year,
                    'temp_avg': weather_data['temp_avg'],
                    'temp_min': weather_data['temp_min'],
                    'temp_max': weather_data['temp_max'],
                    'precipitation': weather_data['precipitation'],
                    'wind_speed': weather_data['wind_speed'],
                    'humidity': weather_data['humidity'],
                    # Add categorical features
                    'road_condition': 'wet' if weather_data['precipitation'] > 0 else 'snow' if weather_data.get(
                        'is_snow', 0) else 'dry'
                }

                # Create a DataFrame
                X_pred = pd.DataFrame([features])

                # Make prediction
                if hasattr(accident_model, 'predict_proba'):
                    # For classifiers with probability
                    probabilities = accident_model.predict_proba(X_pred)
                    if probabilities.shape[1] > 1:
                        model_prediction = probabilities[0, 1]  # Probability of class 1 (accident with severity)
                        model_confidence = 0.7  # Trust model with 70% weight
                    else:
                        model_prediction = accident_model.predict(X_pred)[0]
                        model_confidence = 0.5  # Trust binary prediction with 50% weight
                else:
                    # For models without probability
                    model_prediction = accident_model.predict(X_pred)[0]
                    model_confidence = 0.5  # Trust binary prediction with 50% weight

                print(f"Model prediction: {model_prediction}, confidence: {model_confidence}")

            except Exception as e:
                print(f"Error in model prediction: {e}")
                print(traceback.format_exc())

        # Calculate rule-based risk factors

        # 1. Location-based risk (0-50 points)
        location_risk = 0
        if len(similar_accidents) > 0:
            # More accidents nearby = higher risk
            nearby_count = len(similar_accidents)
            location_risk = min(50, nearby_count)

            # Check for severity in similar accidents
            if 'accident_severity' in similar_accidents.columns:
                # Calculate percentage of severe accidents
                severe_pct = similar_accidents['accident_severity'].mean() * 100
                # Adjust location risk based on severity percentage
                location_risk = min(50, location_risk * (1 + severe_pct / 100))
        else:
            # No historical data, use a moderate risk
            location_risk = 25

        # Add some random variation with fixed seed for reproducibility
        seed = int(hash(f"{lat:.3f}_{lon:.3f}_{date_str}") % 10000000)
        np.random.seed(seed)
        location_risk += np.random.uniform(-5, 5)
        location_risk = max(1, min(50, location_risk))

        # 2. Weather risk factors (0-35 points)
        weather_risk = 0

        # Temperature risk
        if weather_data['temp_avg'] < -10:
            weather_risk += 15  # Very cold
        elif weather_data['temp_avg'] < 0:
            weather_risk += 10  # Below freezing
        elif weather_data['temp_avg'] > 30:
            weather_risk += 5  # Very hot

        # Precipitation risk
        if weather_data['precipitation'] > 15:
            weather_risk += 20  # Heavy rain
        elif weather_data['precipitation'] > 5:
            weather_risk += 12  # Moderate rain
        elif weather_data['precipitation'] > 0:
            weather_risk += 8  # Light rain

        # Snow risk (convert is_snow from int back to boolean if needed)
        is_snow = bool(weather_data.get('is_snow', 0))
        if is_snow:
            weather_risk += 25

        # Wind risk
        if weather_data['wind_speed'] > 15:
            weather_risk += 15  # Strong wind
        elif weather_data['wind_speed'] > 10:
            weather_risk += 10  # Moderate wind
        elif weather_data['wind_speed'] > 5:
            weather_risk += 5  # Light wind

        # Visibility risk
        if weather_data['condition'] == 'fog':
            weather_risk += 15

        # Cap weather risk
        weather_risk = min(35, weather_risk)

        # 3. Time-based risk (0-15 points)
        # Weekend risk
        weekend_risk = 10 if day_of_week >= 5 else 0

        # Month risk
        month_risk_map = [8, 7, 4, 2, 0, 2, 5, 5, 2, 4, 6, 9]  # Jan to Dec
        month_risk = month_risk_map[month]

        time_risk = min(15, weekend_risk + month_risk)

        # Combine rule-based risk with model prediction
        rule_based_risk = location_risk + time_risk + weather_risk

        # Calculate total risk using weighted average of model and rules
        total_risk = rule_based_risk
        if model_prediction is not None:
            # Convert model_prediction to 0-100 scale if needed
            if isinstance(model_prediction, (int, float)) and 0 <= model_prediction <= 1:
                model_risk = model_prediction * 100
            else:
                model_risk = 50  # Default for uncertain predictions

            # Weighted average
            total_risk = (model_confidence * model_risk) + ((1 - model_confidence) * rule_based_risk)

        # Ensure risk is within bounds
        total_risk = max(1, min(99, total_risk))

        # Determine risk category
        risk_category = 'Vysoké' if total_risk > 70 else 'Střední' if total_risk > 40 else 'Nízké'

        # Generate safety tips
        safety_tips = []

        # Always include a general tip
        general_tips = [
            "Vždy dodržujte bezpečnou vzdálenost od vozidla před vámi.",
            "Připoutejte se bezpečnostními pásy a zajistěte, aby tak učinili i všichni spolucestující.",
            "Dodržujte dopravní předpisy a povolenou rychlost.",
            "Věnujte řízení plnou pozornost a nepoužívejte mobilní telefon za jízdy."
        ]
        safety_tips.append(np.random.choice(general_tips))

        # Location-specific tip
        if location_risk > 30:
            safety_tips.append(
                "V této oblasti došlo v minulosti k vyššímu počtu nehod. Buďte obzvláště opatrní a pozorní.")

        # Weather-specific tips
        if weather_data['precipitation'] > 0:
            safety_tips.append("Při dešti zpomalte a zvětšete odstup od ostatních vozidel.")

        if is_snow:
            safety_tips.append("Na sněhu omezte rychlost, vyhněte se prudkému brzdění a řiďte plynule.")

        if weather_data['temp_avg'] < 0:
            safety_tips.append(
                "Při teplotách pod bodem mrazu počítejte s možností námrazy na vozovce, zejména na mostech a v lesních úsecích.")

        if weather_data['wind_speed'] > 10:
            safety_tips.append(
                "Při silném větru buďte opatrní především při předjíždění větších vozidel a na otevřených úsecích.")

        if weather_data['condition'] == 'fog':
            safety_tips.append(
                "V mlze používejte mlhová nebo potkávací světla a zpomalte na rychlost přiměřenou viditelnosti.")

        # Time-specific tips
        if weekend_risk > 0:
            safety_tips.append(
                "O víkendu je vyšší provoz a více řidičů s menšími zkušenostmi, buďte obzvláště opatrní.")

        # Return risk assessment with detailed factors
        return {
            'risk_score': round(total_risk, 1),
            'risk_category': risk_category,
            'safety_tips': safety_tips,
            'detailed_factors': {
                'location_risk': round(location_risk, 1),
                'time_risk': round(time_risk, 1),
                'weather_risk': round(weather_risk, 1),
                'model_prediction': round(model_prediction * 100, 1) if model_prediction is not None and isinstance(
                    model_prediction, float) else None,
                'model_confidence': round(model_confidence * 100, 1)
            }
        }

    except Exception as e:
        print(f"Error calculating risk score: {e}")
        print(traceback.format_exc())

        # Return default risk assessment
        return {
            'risk_score': 50,
            'risk_category': 'Střední',
            'safety_tips': [
                "Vždy dodržujte bezpečnou vzdálenost od vozidla před vámi.",
                "Došlo k chybě při výpočtu. Pro jistotu vždy dodržujte bezpečnost provozu."
            ],
            'detailed_factors': {
                'location_risk': 25,
                'time_risk': 10,
                'weather_risk': 15,
                'model_prediction': None,
                'model_confidence': 0
            }
        }


def get_accident_hotspots():
    """Generate accident hotspots from the merged data"""
    try:
        if merged_data is None or len(merged_data) == 0:
            return {"type": "FeatureCollection", "features": []}

        # Check if we have location data
        if 'accident_lat' not in merged_data.columns or 'accident_lon' not in merged_data.columns:
            return {"type": "FeatureCollection", "features": []}

        # Get unique accident locations (to avoid duplicates)
        locations = merged_data[['accident_lat', 'accident_lon']].dropna().drop_duplicates()

        # If we have severity information, use it
        if 'accident_severity' in merged_data.columns:
            # Merge severity data
            accident_counts = merged_data.groupby(['accident_lat', 'accident_lon']).size().reset_index(name='count')
            severity_data = merged_data.groupby(['accident_lat', 'accident_lon'])[
                'accident_severity'].mean().reset_index(name='severity')

            locations = accident_counts.merge(severity_data, on=['accident_lat', 'accident_lon'])
        else:
            # Just count accidents at each location
            locations = merged_data.groupby(['accident_lat', 'accident_lon']).size().reset_index(name='count')
            locations['severity'] = 0.5  # Default medium severity

        # If we have too many points, sample the most severe ones
        if len(locations) > 300:
            # Sort by severity and count
            locations['importance'] = locations['count'] * (1 + locations['severity'])
            locations = locations.sort_values('importance', ascending=False).head(300)

        # Convert to GeoJSON format
        features = []
        for _, row in locations.iterrows():
            # Determine intensity based on count and severity
            intensity = min(1.0, (row['count'] / 10) * (1 + row['severity']))

            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row['accident_lon']), float(row['accident_lat'])]
                },
                "properties": {
                    "type": "accident",
                    "count": int(row['count']),
                    "severity": float(row['severity']),
                    "intensity": float(intensity)
                }
            })

        return {
            "type": "FeatureCollection",
            "features": features
        }

    except Exception as e:
        print(f"Error generating accident hotspots: {e}")
        print(traceback.format_exc())
        return {"type": "FeatureCollection", "features": []}


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/results')
def results():
    """Show prediction results"""
    # Get data from session
    lat = session.get('lat', 49.1951)
    lng = session.get('lng', 16.6068)
    date_str = session.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Get the weather and risk data
    try:
        # Use real weather API for forecasts if available
        if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_API_KEY":
            weather = get_real_weather_forecast(lat, lng, date_str)
        else:
            weather = get_model_weather_forecast(lat, lng, date_str)

        risk = calculate_risk_score(lat, lng, date_str, weather)
        error = None
    except Exception as e:
        print(f"Error generating data for results: {e}")
        weather = get_model_weather_forecast(lat, lng, date_str)
        risk = calculate_risk_score(lat, lng, date_str, weather)
        error = str(e)

    # Clear session data
    session.pop('error', None)

    # Render results template
    return render_template('results.html',
                           lat=lat,
                           lng=lng,
                           date=date_str,
                           weather=weather,
                           risk=risk,
                           error=error,
                           api_key=OPENWEATHER_API_KEY if OPENWEATHER_API_KEY != "YOUR_API_KEY" else None)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and redirect to results"""
    try:
        # Get form data
        lat = float(request.form.get('lat', 49.1951))
        lng = float(request.form.get('lng', 16.6068))
        date_str = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))

        # Store in session - just store simple values, not complex objects
        session['lat'] = lat
        session['lng'] = lng
        session['date'] = date_str

        # Redirect to results page
        return redirect(url_for('results'))

    except Exception as e:
        print(f"Error in predict route: {e}")
        print(traceback.format_exc())
        session['error'] = str(e)
        return redirect(url_for('results'))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction requests"""
    try:
        # Get JSON data
        data = request.json
        lat = float(data.get('lat', 49.1951))
        lng = float(data.get('lng', 16.6068))
        date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))

        # Get predictions
        if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "YOUR_API_KEY":
            weather = get_real_weather_forecast(lat, lng, date_str)
        else:
            weather = get_model_weather_forecast(lat, lng, date_str)

        risk = calculate_risk_score(lat, lng, date_str, weather)

        # Return JSON response
        return jsonify({
            'success': True,
            'weather': weather,
            'risk': risk
        })

    except Exception as e:
        print(f"Error in API predict: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'weather': {
                'temp_avg': 15,
                'temp_max': 20,
                'temp_min': 10,
                'humidity': 70,
                'wind_speed': 5,
                'precipitation': 0,
                'condition': 'clear',
                'is_snow': 0
            },
            'risk': {
                'risk_score': 50,
                'risk_category': 'Střední',
                'safety_tips': ["Došlo k chybě při výpočtu. Pro jistotu vždy dodržujte bezpečnost provozu."]
            }
        })


@app.route('/api/hotspots')
def api_hotspots():
    """API endpoint for accident hotspots"""
    return jsonify(get_accident_hotspots())


if __name__ == '__main__':
    # Load data and initialize models before starting the app
    load_data()

    # Run the app
    app.run(debug=True)