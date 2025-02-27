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


def simple_visualize_results(dangerous_roads, accidents_gdf, output_file='dangerous_roads.html'):
    """Create a simple interactive web visualization with OpenStreetMap background."""
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
                   tiles='OpenStreetMap', control_scale=True)

    # Create feature groups
    roads_layer = folium.FeatureGroup(name="Road Segments")
    danger_layer = folium.FeatureGroup(name="Top 5 Dangerous Roads")
    accident_layer = folium.FeatureGroup(name="Accidents")

    # Add layers to map first
    roads_layer.add_to(m)
    danger_layer.add_to(m)
    accident_layer.add_to(m)

    # Create a better distributed colormap for danger index
    # Find the danger index values at different percentiles
    danger_values = dangerous_roads['danger_index'].sort_values()

    # Get values at different percentiles for a better spread
    p20 = danger_values.quantile(0.2)
    p40 = danger_values.quantile(0.4)
    p60 = danger_values.quantile(0.6)
    p80 = danger_values.quantile(0.8)
    p90 = danger_values.quantile(0.9)
    p95 = danger_values.quantile(0.95)

    # Create a custom color function using these thresholds
    def get_color(danger_index):
        if danger_index <= p20:
            return '#1a9641'  # Dark green
        elif danger_index <= p40:
            return '#a6d96a'  # Light green
        elif danger_index <= p60:
            return '#ffffbf'  # Yellow
        elif danger_index <= p80:
            return '#fdae61'  # Orange
        elif danger_index <= p90:
            return '#d7191c'  # Red
        else:
            return '#7f0000'  # Dark red

    # Create a simple gradient colormap for the legend
    from branca.colormap import LinearColormap
    colormap = LinearColormap(
        colors=['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c', '#7f0000'],
        vmin=danger_values.min(),
        vmax=danger_values.max(),
        caption='Danger Index (percentile-based)'
    )
    m.add_child(colormap)

    # Add road segments with better color distribution
    for idx, row in dangerous_roads.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            line_points = [[point[1], point[0]] for point in coords]  # Swap lat/lon for folium

            road_id = row.get('id', idx)
            danger_index = row['danger_index']
            accident_count = row['accident_count']

            # Choose color based on percentile threshold
            color = get_color(danger_index)

            # Create more informative popup
            popup_html = f"""
            <div style="min-width:200px">
                <h4>Road ID: {road_id}</h4>
                <p><b>Danger Index:</b> {danger_index:.4f}</p>
                <p><b>Accidents:</b> {int(accident_count)}</p>
                <p><b>Length:</b> {row['length_km']:.2f} km</p>
                <p><b>Traffic:</b> {int(row['traffic_volume'])} veh/day</p>
                <p><b>Percentile:</b> {(dangerous_roads['danger_index'] <= danger_index).mean() * 100:.1f}%</p>
            </div>
            """

            folium.PolyLine(
                line_points,
                color=color,
                weight=4,
                opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Road {road_id} - Danger: {danger_index:.2f}"
            ).add_to(roads_layer)

    # Add top 5 dangerous roads
    top_dangerous = dangerous_roads.head(5)
    for idx, row in top_dangerous.iterrows():
        if isinstance(row.geometry, LineString):
            coords = list(row.geometry.coords)
            line_points = [[point[1], point[0]] for point in coords]

            road_id = row.get('id', idx)
            danger_index = row['danger_index']

            folium.PolyLine(
                line_points,
                color='black',
                weight=6,
                opacity=0.8,
                popup=f"Dangerous Road ID: {road_id}<br>Danger Index: {danger_index:.4f}"
            ).add_to(danger_layer)

    # Add accident markers
    accident_cluster = MarkerCluster().add_to(accident_layer)

    for idx, row in accidents_gdf.iterrows():
        if isinstance(row.geometry, Point):
            lat, lon = row.geometry.y, row.geometry.x

            # Simple popup
            popup_content = "Accident"

            if 'nasledky' in row and pd.notna(row['nasledky']):
                popup_content = f"{row['nasledky']}"

            # Determine marker color based on severity
            if 'severity' in row and pd.notna(row['severity']):
                severity = row['severity']
                if severity > 5:
                    icon = folium.Icon(color='red', icon='info-sign')
                elif severity > 1:
                    icon = folium.Icon(color='orange', icon='info-sign')
                else:
                    icon = folium.Icon(color='blue', icon='info-sign')
            else:
                icon = folium.Icon(color='blue', icon='info-sign')

            folium.Marker(
                location=[lat, lon],
                icon=icon,
                popup=popup_content
            ).add_to(accident_cluster)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add a better legend with color distribution
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 1px solid grey; border-radius: 5px">
        <p style="margin:5px 0 10px 0"><b>Danger Index Legend</b></p>
        <div style="margin-bottom:5px">
            <span style="display:inline-block; width:20px; height:10px; background-color:#1a9641; margin-right:5px"></span>
            Low Risk (0-20th percentile): &lt; {p20:.2f}
        </div>
        <div style="margin-bottom:5px">
            <span style="display:inline-block; width:20px; height:10px; background-color:#a6d96a; margin-right:5px"></span>
            Low-Medium Risk (20-40th): {p20:.2f} - {p40:.2f}
        </div>
        <div style="margin-bottom:5px">
            <span style="display:inline-block; width:20px; height:10px; background-color:#ffffbf; margin-right:5px"></span>
            Medium Risk (40-60th): {p40:.2f} - {p60:.2f}
        </div>
        <div style="margin-bottom:5px">
            <span style="display:inline-block; width:20px; height:10px; background-color:#fdae61; margin-right:5px"></span>
            Medium-High Risk (60-80th): {p60:.2f} - {p80:.2f}
        </div>
        <div style="margin-bottom:5px">
            <span style="display:inline-block; width:20px; height:10px; background-color:#d7191c; margin-right:5px"></span>
            High Risk (80-90th): {p80:.2f} - {p90:.2f}
        </div>
        <div style="margin-bottom:5px">
            <span style="display:inline-block; width:20px; height:10px; background-color:#7f0000; margin-right:5px"></span>
            Very High Risk (90-100th): &gt; {p90:.2f}
        </div>
        <div style="margin-top:10px">
            <span style="display:inline-block; width:20px; height:4px; background-color:black; margin-right:5px"></span>
            Top 5 Most Dangerous Roads
        </div>
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

    # Visualize results with a very simple web map
    map_obj = simple_visualize_results(dangerous_roads, accidents_gdf, 'dangerous_roads.html')

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