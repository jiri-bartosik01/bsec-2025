from datetime import datetime
import json
import psycopg2
from shapely.geometry import LineString
from psycopg2.extras import execute_values

from dotenv import load_dotenv
from os import getenv

from load_weather import insert_weather_data, read_xlsx


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


def create_dopravni_nehody_table(conn):
    """Creates a table for storing the geojson nehodove data."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dopravni_nehody (
                OBJECTID_1 BIGINT PRIMARY KEY,
                geometry GEOMETRY(Point, 4326),
                zuj TEXT,
                alkohol_vinik TEXT,
                hlavni_pricina TEXT,
                srazka TEXT,
                nasledky TEXT,
                pricina TEXT,
                stav_vozovky TEXT,
                povetrnostni_podm TEXT,
                rozhled TEXT,
                misto_nehody TEXT,
                druh_komun TEXT,
                pneumatiky TEXT,
                druh_pohonu TEXT,
                druh_vozidla TEXT,
                mestska_cast TEXT,
                pohlavi TEXT,
                alkohol TEXT,
                den_v_tydnu TEXT,
                mesic_t TEXT,
                katastr TEXT,
                chovani_chodce TEXT,
                stav_chodce TEXT,
                drogy_chodec TEXT,
                alkohol_chodec TEXT,
                osobni_prepravnik TEXT,
                reflexni_prvky TEXT,
                kategorie_chodce TEXT,
                nasledek TEXT,
                ozn_osoba TEXT,
                zavineni TEXT,
                viditelnost TEXT,
                situovani TEXT,
                osoba TEXT,
                stav_ridic TEXT,
                doba TEXT,
                smrt_po TEXT,
                lz BIGINT,
                TARGET_FID_1 BIGINT,
                Join_Count_1 BIGINT,
                OBJECTID BIGINT,
                Join_Count BIGINT,
                TARGET_FID BIGINT,
                den BIGINT,
                vek BIGINT,
                smrt_dny TEXT,
                rok_nar BIGINT,
                p48a BIGINT,
                p59d BIGINT,
                rok BIGINT,
                tz BIGINT,
                smrt BIGINT,
                lehce_zran_os BIGINT,
                tezce_zran_os BIGINT,
                usmrceno_os BIGINT,
                id_vozidla BIGINT,
                hodina BIGINT,
                ovlivneni_ridice BIGINT,
                cas BIGINT,
                mesic BIGINT,
                e BIGINT,
                d BIGINT,
                id_nehody BIGINT,
                datum TEXT,
                hmotna_skoda_1 BIGINT,
                skoda_vozidlo BIGINT,
                GlobalID TEXT
            );
        """
        )
        conn.commit()


def load_geojson(file_path: str):
    """Loads geojson data from a file."""
    with open(file_path, "r") as file:
        geojson = json.load(file)
    return geojson


def load_json(file_path: str):
    """Loads geojson data from a file."""
    with open(file_path, "r") as file:
        json_parsed = json.load(file)
    return json_parsed


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


def insert_dopravni_nehody_data(conn, data):
    """Inserts parsed geojson data into the table."""

    # Connect to the PostgreSQL database
    cur = conn.cursor()
    for json_data in data["features"]:

        # Extracting values from JSON
        geom = json_data["geometry"]
        lon, lat = geom["coordinates"]  # Extract coordinates

        properties = json_data["properties"]

        # Convert date format
        datum = datetime.strptime(properties["datum"], "%a, %d %b %Y %H:%M:%S GMT") if properties["datum"] else None

        # SQL INSERT statement
        sql = """
        INSERT INTO dopravni_nehody (
            objectid_1, zuj, alkohol_vinik, hlavni_pricina, srazka, nasledky, pricina, stav_vozovky,
            povetrnostni_podm, rozhled, misto_nehody, druh_komun, pneumatiky, druh_pohonu, druh_vozidla,
            mestska_cast, pohlavi, alkohol, den_v_tydnu, mesic_t, katastr, nasledek, ozn_osoba, zavineni,
            viditelnost, situovani, osoba, stav_ridic, doba, lz, den, vek, rok_nar, rok, tz, smrt, lehce_zran_os,
            tezce_zran_os, usmrceno_os, id_vozidla, hodina, ovlivneni_ridice, cas, mesic, e, d, id_nehody, datum,
            hmotna_skoda_1, skoda_vozidlo, globalid, geometry
        )
        VALUES %s
        """

        values = (
            properties["OBJECTID_1"],
            properties["zuj"],
            properties["alkohol_vinik"],
            properties["hlavni_pricina"],
            properties["srazka"],
            properties["nasledky"],
            properties["pricina"],
            properties["stav_vozovky"],
            properties["povetrnostni_podm"],
            properties["rozhled"],
            properties["misto_nehody"],
            properties["druh_komun"],
            properties["pneumatiky"],
            properties["druh_pohonu"],
            properties["druh_vozidla"],
            properties["mestska_cast"],
            properties["pohlavi"],
            properties["alkohol"],
            properties["den_v_tydnu"],
            properties["mesic_t"],
            properties["katastr"],
            properties["nasledek"],
            properties["ozn_osoba"],
            properties["zavineni"],
            properties["viditelnost"],
            properties["situovani"],
            properties["osoba"],
            properties["stav_ridic"],
            properties["doba"],
            properties["lz"],
            properties["den"],
            properties["vek"],
            properties["rok_nar"],
            properties["rok"],
            properties["tz"],
            properties["smrt"],
            properties["lehce_zran_os"],
            properties["tezce_zran_os"],
            properties["usmrceno_os"],
            properties["id_vozidla"],
            properties["hodina"],
            properties["ovlivneni_ridice"],
            properties["cas"],
            properties["mesic"],
            properties["e"],
            properties["d"],
            properties["id_nehody"],
            datum,
            properties["hmotna_skoda_1"],
            properties["skoda_vozidlo"],
            properties["GlobalID"],
            f"SRID=4326;POINT({lon} {lat})",  # PostGIS geometry format
        )

        # Execute the query
        execute_values(cur, sql, [values])

    # Commit and close
    conn.commit()
    cur.close()
    conn.close()
    print("Data successfully inserted into dopravni_nehody.")


def main():
    """Main function to handle database connection and data processing."""
    conn = psycopg2.connect(**DB_PARAMS)

    try:
        # create_table(conn)
        # data = load_geojson("/home/mkoo7mk/Downloads/Telegram Desktop/intenzita_dopravy_pentlogramy_-2087072355465349330.geojson")
        # insert_data(conn, data)
        # create_dopravni_nehody_table(conn)
        # data = load_json("/home/mkoo7mk/Downloads/Telegram Desktop/dopravni_nehody_4922215206159905331.geojson")
        # insert_dopravni_nehody_data(conn, data)
        data = read_xlsx("/home/mkoo7mk/Downloads/Telegram Desktop/B2BTUR01.xlsx")
        insert_weather_data(conn, data)  # type: ignore
    finally:
        conn.close()


if __name__ == "__main__":
    main()
