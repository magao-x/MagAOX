import csv
import pandas as pd
import argparse

def convert_csv_to_parquet(path_to_csv, save_to, parse_name="timestamp"):
    # 1) Read the CSV, parsing the timestamp column
    df = pd.read_csv(
        path_to_csv,
        parse_dates=[parse_name]       # make sure this matches your column name
    )
    df = df.set_index("timestamp")
    df.to_parquet(
    # "wind_lookup.parquet",
    save_to,
    engine="pyarrow",             # the default in recent pandas
    # compression="snappy"          # optional, gives a good size/speed tradeoff
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Take the output CSV from retrieve_wind_properties and convert it to a .parquet"
    )
    parser.add_argument(
        "path2csv", type=str,
        help="Path to the input wind CSV."
    )
    parser.add_argument(
        "--save-parquet-to", type=str, required=True,
        help="Path to save the parquet file."
    )
    args = parser.parse_args()
    convert_csv_to_parquet(args.path2csv, args.save_parquet_to)