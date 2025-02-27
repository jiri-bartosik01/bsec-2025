import pandas as pd


def read_xlsx(file_path: str):
    xl_file = pd.ExcelFile(file_path)

    return {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names}


def insert_weather_data(conn, data: dict[str, pd.DataFrame]):
    """Inserts data into the database."""
    values: dict[tuple, dict[str, float]] = {}
    create_table_query = """
        CREATE TABLE IF NOT EXISTS weather (
            datum DATE PRIMARY KEY,
            """
    for sheet_name, df in data.items():
        if sheet_name == "geografie stanice":
            continue
        # Delete first row
        df = df.iloc[3:]
        table_name = sheet_name.replace(" ", "_").lower()
        create_table_query += f"{table_name} FLOAT,\n"
        days = df.columns
        rows = df.values
        for i, value in enumerate(rows):
            year = int(value[0])
            month = int(value[1])
            for j, val in enumerate(value[2:]):
                day = days[j + 1].split(": ")[1]
                if val == "" or str(val).lower() == "nan":
                    continue
                if (year, month, day) in values:
                    values[(year, month, day)][table_name] = float(val)
                else:
                    values[(year, month, day)] = {table_name: float(val)}
    create_table_query = create_table_query[:-2] + ");"
    with conn.cursor() as cur:
        cur.execute(create_table_query)
    # Create table with columns
    for (year, month, day), val_with_table_name in values.items():
        # Parse a date from year month and day so i can insert it into the database
        insert_values = [f"{year}-{month}-{day}"]
        for table_name, val in val_with_table_name.items():
            insert_values.append(val)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO weather VALUES ({', '.join(['%s' for _ in range(len(insert_values))])});",
                insert_values,
            )
    conn.commit()
    print("Data inserted successfully.")
