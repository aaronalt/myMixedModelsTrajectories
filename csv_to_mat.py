"""
csv_to_mat.py

Reads a CSV into a pandas DataFrame, optionally drops rows with missing
values, and saves the result as a .mat file for MATLAB.

Usage:
    python csv_to_mat.py input.csv output.mat
    python csv_to_mat.py input.csv output.mat --dropna
"""

import sys
import pandas as pd
import numpy as np
from scipy.io import savemat


def csv_to_mat(csv_path, mat_path, drop_na=False):
    df = pd.read_csv(csv_path)

    if drop_na:
        df = df.dropna().reset_index(drop=True)

    mat_dict = {}
    col_names = []

    for col in df.columns:
        safe = col.replace('-', '_').replace(' ', '_')
        col_names.append(safe)

        if pd.api.types.is_numeric_dtype(df[col]):
            mat_dict[safe] = df[col].values.astype(np.float64)
        else:
            mat_dict[safe] = np.array(df[col].astype(str).tolist(), dtype=object)

    mat_dict['col_names'] = np.array(col_names, dtype=object)

    savemat(mat_path, mat_dict)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols to {mat_path}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python csv_to_mat.py input.csv output.mat [--dropna]")
        sys.exit(1)

    csv_path = sys.argv[1]
    mat_path = sys.argv[2]
    drop_na = '--dropna' in sys.argv

    csv_to_mat(csv_path, mat_path, drop_na)
