import pandas as pd
import os

# ---------------------
def load_data(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERROR: File {file_path} does not exist")

    try:
        df = pd.read_csv(file_path)

        print(f"--- Data has successfully loaded: {file_path}. Shape: {df.shape} ---")
        return df

    except Exception as e:
        raise Exception(f"ERROR: Something went wrong while reading the data: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    path = os.path.join(current_dir, "..", "data", "raw", "train.csv")

    try:
        data = load_data(path)
        print(data.head())
    except Exception as error:
        print(error)