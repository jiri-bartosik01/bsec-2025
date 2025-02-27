import regex as re
import pandas as pd


def remove_fully_null_columns(csv_file: str, output_file: str):
    """
    Reads a CSV file, removes columns where all values are null, and saves a new CSV file.

    Args:
        csv_file (str): Path to the input CSV file.
        output_file (str): Path to save the new CSV file without fully null columns.

    Returns:
        None
    """
    df = pd.read_csv(csv_file)  # Load CSV into DataFrame
    if 'datum' not in df.columns:
        raise ValueError("The CSV file must contain a 'datum' column.")

    df = df.dropna(axis=1, how="all")  # Drop columns where all values are NaN
    df = df.dropna(axis=0, how="any")  # Drop rows where any value is NaN
    # Extract the year from the 'datum' column (assuming format 'YYYY-MM-DD' or similar)
    df['year'] = df['datum'].astype(str).str.extract(r'(\d{4})').astype(float)

    # Find all car_* and truc_* columns
    car_columns = {int(re.search(r'\d{4}', col).group()): col for col in df.columns if col.startswith('car_')}
    truc_columns = {int(re.search(r'\d{4}', col).group()): col for col in df.columns if col.startswith('truc_')}

    df['cars'] = df.apply(lambda row: row.get(f"car_{int(row['year'])}", None), axis=1)
    df['trucks'] = df.apply(lambda row: row.get(f"truc_{int(row['year'])}", None), axis=1)


    # Drop the temporary 'year' column
    df.drop(columns=['year'], inplace=True)
    df.drop(columns=list(car_columns.values()) + list(truc_columns.values()), inplace=True)

    df.to_csv(output_file, index=False)  # Save the cleaned CSV
    print(f"New CSV saved without null columns: {output_file}")


# Example usage
input_csv = "db/_WITH_closest_lines_AS_select_dn_hd_ST_X_dn_geometry_ST_Y_dn_geo_202502272021.csv"  # Replace with your actual CSV file
output_csv = "cleaned_data.csv"  # Output file name
remove_fully_null_columns(input_csv, output_csv)
