import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from datetime import datetime
import warnings


def load_and_clean_data(intenzita_file, nehody_file, weather_file):
    """
    Load all three datasets and perform initial cleaning
    """
    # Load road traffic data
    print("Loading road traffic data...")
    roads_gdf = gpd.read_file(intenzita_file)
    print(f"Loaded {len(roads_gdf)} road segments")

    # Load accident data
    print("Loading accident data...")
    accidents_gdf = gpd.read_file(nehody_file)
    print(f"Loaded {len(accidents_gdf)} accidents")

    # Load weather data
    print("Loading weather data...")
    weather_df = pd.read_csv(weather_file, parse_dates=['date'])
    print(f"Loaded weather data for {len(weather_df)} days")

    # Ensure all accidents have Point geometry
    valid_accidents = accidents_gdf[accidents_gdf.geometry.type == 'Point']
    if len(valid_accidents) < len(accidents_gdf):
        print(f"Warning: {len(accidents_gdf) - len(valid_accidents)} accidents have invalid geometry")
        accidents_gdf = valid_accidents

    # Create a date field for accidents
    if 'datum' in accidents_gdf.columns:
        try:
            accidents_gdf['accident_date'] = pd.to_datetime(accidents_gdf['datum']).dt.date
        except:
            print("Could not parse 'datum' field")

    # Construct dates from component fields if needed
    if 'accident_date' not in accidents_gdf.columns and all(
            field in accidents_gdf.columns for field in ['rok', 'mesic', 'den']):
        try:
            # Convert to numeric if needed
            for field in ['rok', 'mesic', 'den']:
                if accidents_gdf[field].dtype == 'object':
                    accidents_gdf[field] = pd.to_numeric(accidents_gdf[field], errors='coerce')

            # Construct date
            accidents_gdf['accident_date'] = pd.to_datetime(
                accidents_gdf[['rok', 'mesic', 'den']].rename(
                    columns={'rok': 'year', 'mesic': 'month', 'den': 'day'}
                )
            ).dt.date
        except Exception as e:
            print(f"Error constructing dates: {e}")

    return roads_gdf, accidents_gdf, weather_df


def find_nearest_road(accident_point, roads_gdf):
    """
    Find the nearest road segment to an accident
    Returns the index of the nearest road and the distance
    """
    # Calculate distances
    distances = roads_gdf.geometry.apply(lambda x: accident_point.distance(x))

    # Find the index of the minimum distance
    nearest_idx = distances.idxmin()

    return nearest_idx, distances[nearest_idx]


def process_accidents(accidents_gdf, roads_gdf, weather_df, max_distance=None):
    """
    For each accident, find the nearest road and match with weather data
    """
    # Prepare result list
    merged_records = []

    # Counters for statistics
    total = len(accidents_gdf)
    processed = 0
    skipped_no_date = 0
    skipped_no_weather = 0
    skipped_distance = 0

    # Process each accident
    for idx, accident in accidents_gdf.iterrows():
        if idx % 100 == 0:
            print(f"Processing accident {idx + 1}/{total}")

        # Check if we have a date
        accident_date = None
        if 'accident_date' in accident and pd.notna(accident['accident_date']):
            accident_date = accident['accident_date']
        elif 'datum' in accident and pd.notna(accident['datum']):
            try:
                accident_date = pd.to_datetime(accident['datum']).date()
            except:
                pass

        # Skip if no date available
        if accident_date is None:
            skipped_no_date += 1
            continue

        # Find the nearest road segment
        try:
            nearest_road_idx, distance = find_nearest_road(accident.geometry, roads_gdf)
            nearest_road = roads_gdf.iloc[nearest_road_idx]
        except Exception as e:
            print(f"Error finding nearest road for accident {idx}: {e}")
            continue

        # Skip if the accident is too far from any road
        if max_distance is not None and distance > max_distance:
            skipped_distance += 1
            continue

        # Match with weather data
        weather_on_date = weather_df[weather_df['date'].dt.date == accident_date]

        # Skip if no weather data available
        if len(weather_on_date) == 0:
            skipped_no_weather += 1
            continue

        # Get the first weather record for that day
        weather = weather_on_date.iloc[0]

        # Extract accident year for traffic data
        accident_year = accident_date.year

        # Determine which traffic data to use (exact year or closest available)
        car_cols = [col for col in nearest_road.index if col.startswith('car_')]
        if car_cols:
            traffic_years = [int(col.split('_')[1]) for col in car_cols]
            closest_year = min(traffic_years, key=lambda y: abs(y - accident_year))
            traffic_year = closest_year
        else:
            traffic_year = None

        # Create merged record
        record = {
            # Basic accident info
            'accident_id': accident.get('id_nehody', accident.get('OBJECTID', idx)),
            'accident_date': accident_date,
            'accident_lat': accident.geometry.y,
            'accident_lon': accident.geometry.x,

            # Road information
            'road_id': nearest_road.get('id', None),
            'distance_to_road': distance,

            # Traffic data (if available for the accident year)
            'car_count': nearest_road.get(f'car_{traffic_year}', None) if traffic_year else None,
            'truck_count': nearest_road.get(f'truc_{traffic_year}', None) if traffic_year else None,

            # Weather data
            'temp_avg': weather.get('temp_avg', None),
            'temp_max': weather.get('temp_max', None),
            'temp_min': weather.get('temp_min', None),
            'wind_speed': weather.get('wind_speed', None),
            'pressure': weather.get('pressure', None),
            'humidity': weather.get('humidity', None),
            'rain_cumulative': weather.get('rain_cumulative', None),
            'snow_height': weather.get('snow_height', None),
            'sun': weather.get('sun', None),
        }

        # Add all traffic data for all years
        for year in range(2010, 2024):
            car_key = f'car_{year}'
            truck_key = f'truc_{year}'

            if car_key in nearest_road:
                record[car_key] = nearest_road[car_key]

            if truck_key in nearest_road:
                record[truck_key] = nearest_road[truck_key]

        # Add all other accident properties
        for key, value in accident.items():
            if key != 'geometry' and key not in record and pd.notna(value):
                record[f'acc_{key}'] = value

        # Add key road properties
        for key, value in nearest_road.items():
            if key not in ['geometry', 'id'] and key not in record and not key.startswith(
                    'car_') and not key.startswith('truc_') and pd.notna(value):
                record[f'road_{key}'] = value

        # Add record to results
        merged_records.append(record)
        processed += 1

    # Create DataFrame from all records
    result_df = pd.DataFrame(merged_records)

    # Print statistics
    print(f"\nProcessing complete:")
    print(f"  Total accidents: {total}")
    print(f"  Successfully processed: {processed}")
    print(f"  Skipped - no date: {skipped_no_date}")
    print(f"  Skipped - no weather data: {skipped_no_weather}")
    print(f"  Skipped - too distant from roads: {skipped_distance}")

    return result_df


def main():
    """Main function to run the data integration process"""
    # File paths
    intenzita_file = "../data/intenzita.geojson"
    nehody_file = "../data/nehody.geojson"
    weather_file = "weather_data.csv"
    output_file = "../accidentPredictor/merged_traffic_accident_data.csv"

    # Maximum allowable distance for linking accidents to roads (None for no limit)
    max_distance = None  # Can be set to a specific value if needed

    print("\n*** Traffic Accident Data Integration ***\n")

    # Suppress warnings
    warnings.filterwarnings('ignore')

    try:
        # Load and prepare data
        roads_gdf, accidents_gdf, weather_df = load_and_clean_data(intenzita_file, nehody_file, weather_file)

        # Process accidents to create merged dataset
        print("\nMatching accidents with roads and weather data...")
        merged_df = process_accidents(accidents_gdf, roads_gdf, weather_df, max_distance)

        # Save results
        print(f"\nSaving {len(merged_df)} records to {output_file}...")
        merged_df.to_csv(output_file, index=False)

        print("\nData integration completed successfully!")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Data integration failed.")


if __name__ == "__main__":
    main()