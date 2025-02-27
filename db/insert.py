import json
import psycopg2
from shapely.geometry import LineString
from psycopg2.extras import execute_values

from dotenv import load_dotenv
from os import getenv


load_dotenv()

# Database connection parameters
DB_PARAMS = {
    "dbname": getenv("DB_NAME", "test"),
    "user": getenv("DB_USER", "test"),
    "password": getenv("DB_PASSWORD", "test"),
    "host": getenv("DB_HOST", "localhost"),
    "port": getenv("DB_PORT", "5432"),
}


def create_table(conn):
    """Creates a table for storing the geojson data."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hustota_dopravy (
                id SERIAL PRIMARY KEY,
                feature_id TEXT NOT NULL,
                geometry GEOMETRY(LineString, 4326),
                car_2010 INTEGER,
                truc_2010 INTEGER,
                car_2011 INTEGER,
                truc_2011 INTEGER,
                car_2012 INTEGER,
                truc_2012 INTEGER,
                car_2013 INTEGER,
                truc_2013 INTEGER,
                car_2014 INTEGER,
                truc_2014 INTEGER,
                car_2015 INTEGER,
                truc_2015 INTEGER,
                car_2016 INTEGER,
                truc_2016 INTEGER,
                car_2017 INTEGER,
                truc_2017 INTEGER,
                car_2018 INTEGER,
                truc_2018 INTEGER,
                car_2019 INTEGER,
                truc_2019 INTEGER,
                car_2020 INTEGER,
                truc_2020 INTEGER,
                car_2021 INTEGER,
                truc_2021 INTEGER,
                car_2022 INTEGER,
                truc_2022 INTEGER,
                car_2023 INTEGER,
                truc_2023 INTEGER,
                datum_exportu TIMESTAMP,
                ObjectId INTEGER,
                GlobalID TEXT
            );
        """
        )
        conn.commit()
        print("Table created successfully.")


def load_geojson(file_path: str):
    """Loads geojson data from a file."""
    with open(file_path, "r") as file:
        geojson = json.load(file)
    return geojson


def insert_data(conn, data):
    """Inserts parsed geojson data into the table."""
    for geojson in data["features"]:
        feature_id = geojson["id"]
        properties = geojson["properties"]

        # Convert coordinates to PostGIS LineString format
        geom = LineString(geojson["geometry"]["coordinates"])
        geom_wkt = geom.wkt  # Convert to WKT format for PostGIS

        columns = ["feature_id", "geometry"] + list(properties.keys())
        values = [feature_id, geom_wkt] + list(properties.values())

        with conn.cursor() as cur:
            query = f"""
                INSERT INTO hustota_dopravy ({', '.join(columns)})
                VALUES ({', '.join(['%s'] * len(values))})
            """
            cur.execute(query, values)
            conn.commit()
            print("Data inserted successfully.")


def main():
    """Main function to handle database connection and data processing."""
    conn = psycopg2.connect(**DB_PARAMS)

    try:
        # create_table(conn)
        data = load_geojson("/home/mkoo7mk/Downloads/Telegram Desktop/intenzita_dopravy_pentlogramy_-2087072355465349330.geojson")
        insert_data(conn, data)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
