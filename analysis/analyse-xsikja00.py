import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, mapping
import numpy as np
import json
import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear


def load_geojson_files(accident_file, traffic_file):
    """Load and process GeoJSON files for accidents and traffic."""
    try:
        # Load GeoJSON files
        accidents_gdf = gpd.read_file(accident_file)
        traffic_gdf = gpd.read_file(traffic_file)

        print(f"Loaded {len(accidents_gdf)} accident records and {len(traffic_gdf)} traffic segments")
        return accidents_gdf, traffic_gdf

    except Exception as e:
        print(f"Error loading GeoJSON files: {e}")
        return None, None


def calculate_severity(accidents_gdf):
    """Calculate severity score for each accident."""
    # Convert severity column to float type first to avoid dtype warnings
    accidents_gdf['severity'] = 0.0  # Initialize as float

    # Calculate accident severity based on injuries and fatalities
    if 'lehce_zran_os' in accidents_gdf.columns and 'tezce_zran_os' in accidents_gdf.columns and 'usmrceno_os' in accidents_gdf.columns:
        accidents_gdf['severity'] = (
                accidents_gdf['usmrceno_os'].astype(float) * 10.0 +  # Fatal accidents (highest weight)
                accidents_gdf['tezce_zran_os'].astype(float) * 5.0 +  # Serious injuries
                accidents_gdf['lehce_zran_os'].astype(float) * 1.0  # Light injuries
        )
    else:
        # If columns not found, use a default severity
        accidents_gdf['severity'] = 1.0

    # Add minimum severity for property damage only accidents
    accidents_gdf.loc[accidents_gdf['severity'] == 0, 'severity'] = 0.1

    if 'hmotna_skoda_1' in accidents_gdf.columns:
        # Integrate property damage into severity (normalized)
        max_damage = accidents_gdf['hmotna_skoda_1'].max()
        if max_damage > 0:
            accidents_gdf['severity'] += accidents_gdf['hmotna_skoda_1'].astype(float) / max_damage

    return accidents_gdf


def calculate_traffic_metrics(traffic_gdf):
    """Calculate traffic metrics for each road segment."""
    # Initialize metrics columns as float to avoid dtype warnings
    traffic_gdf['traffic_volume'] = 0.0
    traffic_gdf['length_km'] = 0.0

    # Identify the traffic volume columns for cars and trucks
    car_cols = [col for col in traffic_gdf.columns if col.startswith('car_')]
    truck_cols = [col for col in traffic_gdf.columns if col.startswith('truc_')]

    # Calculate most recent traffic volume (2023)
    if 'car_2023' in traffic_gdf.columns and 'truc_2023' in traffic_gdf.columns:
        traffic_gdf['traffic_volume'] = traffic_gdf['car_2023'].astype(float) + traffic_gdf['truc_2023'].astype(float)
    elif car_cols and truck_cols:
        # Get the most recent year available
        latest_car_col = sorted(car_cols)[-1]
        latest_truck_col = sorted(truck_cols)[-1]
        traffic_gdf['traffic_volume'] = traffic_gdf[latest_car_col].astype(float) + traffic_gdf[
            latest_truck_col].astype(float)
    else:
        # Fallback if columns not found
        print("Warning: Traffic volume columns not found. Using default value.")
        traffic_gdf['traffic_volume'] = 1000.0

    # Project to a local CRS for Czech Republic (EPSG:5514 - S-JTSK / Krovak East North)
    # This ensures accurate length calculations
    traffic_gdf_projected = traffic_gdf.to_crs("EPSG:5514")

    # Calculate road segment length in kilometers
    traffic_gdf['length_km'] = traffic_gdf_projected.geometry.length / 1000.0  # Convert meters to km

    return traffic_gdf


def identify_dangerous_locations(accidents_gdf, traffic_gdf, buffer_distance=50):
    """Identify dangerous locations by associating accidents with road segments."""
    # Initialize accident metrics columns as float to avoid dtype warnings
    traffic_gdf['accident_count'] = 0.0
    traffic_gdf['accident_severity'] = 0.0

    # Project both datasets to the same projected CRS for Czech Republic (EPSG:5514)
    if accidents_gdf.crs != "EPSG:5514":
        accidents_gdf = accidents_gdf.to_crs("EPSG:5514")

    if traffic_gdf.crs != "EPSG:5514":
        traffic_gdf = traffic_gdf.to_crs("EPSG:5514")

    # Create buffered geometries for road segments to capture nearby accidents
    # Buffer distance in meters (now in a projected CRS)
    traffic_gdf['geometry_buffered'] = traffic_gdf.geometry.buffer(buffer_distance)

    # Associate accidents with road segments
    for idx, road in traffic_gdf.iterrows():
        # Find accidents within this road's buffer
        nearby_accidents = accidents_gdf[accidents_gdf.intersects(road['geometry_buffered'])]

        # Count accidents and sum severity
        accident_count = len(nearby_accidents)
        total_severity = nearby_accidents['severity'].sum() if 'severity' in nearby_accidents.columns else float(
            accident_count)

        # Store the results (as float values)
        traffic_gdf.at[idx, 'accident_count'] = float(accident_count)
        traffic_gdf.at[idx, 'accident_severity'] = float(total_severity)

    # Calculate danger metrics

    # 1. Accidents per kilometer
    traffic_gdf['accidents_per_km'] = traffic_gdf['accident_count'] / traffic_gdf['length_km']

    # 2. Accident severity per kilometer
    traffic_gdf['severity_per_km'] = traffic_gdf['accident_severity'] / traffic_gdf['length_km']

    # 3. Danger index normalized by traffic volume
    # Convert daily traffic to annual (×365) and to millions (÷1,000,000)
    annual_traffic_millions = traffic_gdf['traffic_volume'] * 365 / 1000000

    # Avoid division by zero
    annual_traffic_millions = annual_traffic_millions.replace(0, 0.001)

    # Calculate normalized danger index (severity per km per million vehicles)
    traffic_gdf['danger_index'] = traffic_gdf['severity_per_km'] / annual_traffic_millions

    # Fill NaN values that might result from calculations
    traffic_gdf = traffic_gdf.fillna(0)

    # Sort by danger index to identify most dangerous segments
    dangerous_roads = traffic_gdf.sort_values('danger_index', ascending=False)

    return dangerous_roads


def visualize_results(dangerous_roads, accidents_gdf, output_file='dangerous_roads.html'):
    """Create an interactive web visualization with OpenStreetMap background."""
    # Convert to WGS84 for web mapping
    if dangerous_roads.crs != "EPSG:4326":
        dangerous_roads = dangerous_roads.to_crs("EPSG:4326")

    if accidents_gdf.crs != "EPSG:4326":
        accidents_gdf = accidents_gdf.to_crs("EPSG:4326")

    # Calculate center of the map
    center_lat = (dangerous_roads.geometry.bounds['miny'].mean() +
                  dangerous_roads.geometry.bounds['maxy'].mean()) / 2
    center_lon = (dangerous_roads.geometry.bounds['minx'].mean() +
                  dangerous_roads.geometry.bounds['maxx'].mean()) / 2

    # Create a map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles='OpenStreetMap')

    # Add a colormap for danger index
    min_danger = dangerous_roads['danger_index'].min()
    max_danger = dangerous_roads['danger_index'].max()

    # Create a colormap
    colormap = linear.RdYlGn_09.scale(max_danger, min_danger)
    colormap.caption = 'Danger Index (Higher values are more dangerous)'
    m.add_child(colormap)

    # Add road segments with color based on danger index
    for idx, row in dangerous_roads.iterrows():
        # Get coordinates from the linestring
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            line_points = [[point[1], point[0]] for point in coords]  # Swap lat/lon for folium

            # Get road properties
            road_id = row.get('id', idx)
            danger_index = row['danger_index']
            accident_count = row['accident_count']
            road_length = row['length_km']

            # Choose color based on danger index (red for dangerous, green for safe)
            color = colormap(danger_index)

            # Create popup content
            popup_html = f"""
            <div style="font-family:sans-serif">
                <h4>Road Segment ID: {road_id}</h4>
                <p><b>Danger Index:</b> {danger_index:.4f}</p>
                <p><b>Accidents:</b> {accident_count}</p>
                <p><b>Length:</b> {road_length:.2f} km</p>
                <p><b>Traffic Volume:</b> {row['traffic_volume']} vehicles/day</p>
            </div>
            """

            # Add the line
            folium.PolyLine(
                line_points,
                color=color,
                weight=5,
                opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

    # Use marker clusters for accidents to improve performance
    marker_cluster = MarkerCluster().add_to(m)

    # Add accident points with information
    for idx, row in accidents_gdf.iterrows():
        # Get coordinates
        if isinstance(row.geometry, Point):
            lat, lon = row.geometry.y, row.geometry.x

            # Create popup content based on available columns
            popup_content = ["<div style='font-family:sans-serif'>"]

            # Add key information if available
            for col in ['nasledky', 'pricina', 'druh_vozidla', 'datum']:
                if col in row and pd.notna(row[col]):
                    popup_content.append(f"<p><b>{col}:</b> {row[col]}</p>")

            # Add severity information
            if 'severity' in row and pd.notna(row['severity']):
                popup_content.append(f"<p><b>Severity Score:</b> {row['severity']:.2f}</p>")

            popup_content.append("</div>")
            popup_html = "".join(popup_content)

            # Determine marker color based on severity if available
            if 'severity' in row and pd.notna(row['severity']):
                severity = row['severity']
                if severity > 5:
                    color = 'red'  # High severity
                elif severity > 1:
                    color = 'orange'  # Medium severity
                else:
                    color = 'blue'  # Low severity
            else:
                color = 'blue'

            # Add the marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(marker_cluster)

    # Highlight top 5 most dangerous segments
    top_dangerous = dangerous_roads.head(5)
    for idx, row in top_dangerous.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            line_points = [[point[1], point[0]] for point in coords]  # Swap lat/lon for folium

            road_id = row.get('id', idx)
            danger_index = row['danger_index']

            # Add the highlighted line
            folium.PolyLine(
                line_points,
                color='black',
                weight=8,
                opacity=0.9,
                popup=f"TOP DANGEROUS: Road ID {road_id}, Danger Index: {danger_index:.4f}"
            ).add_to(m)

    # Add a legend for top 5 dangerous segments
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px">
        <p><b>Most Dangerous Road Segments</b></p>
        <p><span style="color:black; font-weight:bold;">━━━</span> Top 5 Most Dangerous</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    m.save(output_file)
    print(f"Interactive web map saved as {output_file}")

    return m


def main():
    """Main function to run the analysis workflow."""
    # File paths
    accident_file = "data/nehody.geojson"
    traffic_file = "data/intenzita.geojson"

    # Load data
    accidents_gdf, traffic_gdf = load_geojson_files(accident_file, traffic_file)

    if accidents_gdf is None or traffic_gdf is None:
        print("Error: Could not process input data.")
        return

    # Process accident data
    accidents_gdf = calculate_severity(accidents_gdf)

    # Process traffic data
    traffic_gdf = calculate_traffic_metrics(traffic_gdf)

    # Identify dangerous locations
    dangerous_roads = identify_dangerous_locations(accidents_gdf, traffic_gdf)

    # Print summary of results
    print("\nTop 10 Most Dangerous Road Segments:")
    top_dangerous = dangerous_roads.head(10)
    for i, (idx, row) in enumerate(top_dangerous.iterrows(), 1):
        road_id = row.get('id', idx)
        print(f"{i}. Road ID: {road_id}")
        print(f"   Accidents: {row['accident_count']}")
        print(f"   Severity: {row['accident_severity']:.1f}")
        print(f"   Length: {row['length_km']:.2f} km")
        print(f"   Traffic volume: {row['traffic_volume']} vehicles/day")
        print(f"   Danger Index: {row['danger_index']:.4f}")
        print()

    # Visualize results with interactive web map
    map_obj = visualize_results(dangerous_roads, accidents_gdf, 'dangerous_roads.html')

    # Export results to CSV and GeoJSON
    # Remove geometry columns for CSV
    export_cols = [col for col in dangerous_roads.columns if col not in ['geometry', 'geometry_buffered']]
    dangerous_roads[export_cols].to_csv('dangerous_roads_analysis.csv', index=False)

    # Export with geometry as GeoJSON
    dangerous_roads_export = dangerous_roads.drop(columns=['geometry_buffered'])
    dangerous_roads_export.to_file('dangerous_roads_analysis.geojson', driver='GeoJSON')

    print("Analysis complete! Results saved as HTML web map, CSV and GeoJSON files.")

    return dangerous_roads


if __name__ == "__main__":
    dangerous_roads = main()