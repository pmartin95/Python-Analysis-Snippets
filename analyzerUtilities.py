import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

# Constants
IDENTIFIER_PATTERN = r"m2_(-?\d+\.?\d*)beta(-?\d+\.?\d*)lambda(-?\d+\.?\d*)kappa(-?\d+\.?\d*)l(\d*)t(\d*)bc([ptc])\.dat"
GENERIC_PATTERN = rf"([a-zA-Z0-9]*){IDENTIFIER_PATTERN}"
PICKLE_PATH = "data.pkl"

plt.rcParams['text.usetex'] = True
plt.style.use('bmh')

def match_filename(filename):
    """
    Extracts information from filename using regex.

    :param filename: Name of the file to be processed.
    :return: Dictionary with matched groups or None if no match found.
    """
    match = re.match(GENERIC_PATTERN, filename)
    if match:
        obs_name, m2, beta, lam, kappa, l, t, bc = match.groups()
        return {
            'obs_name': obs_name,
            'm2': float(m2),
            'beta': float(beta),
            'lam': float(lam),
            'kappa': float(kappa),
            'l': int(l),
            't': int(t),
            'bc': bc
        }
    return None

def process_file(filename):
    """
    Processes the given file and converts it to a DataFrame with metadata from its filename.

    :param filename: Path to the data file to be processed.
    :return: DataFrame with observations and metadata or None if file is not valid.
    """
    metadata = match_filename(filename)
    if not metadata:
        return None

    df = pd.read_csv(filename, sep=" ", header=None)
    column_names = ['Time', 'Obs'] if df.shape[1] == 2 else ['Obs']

    if df.shape[1] > 2:
        raise ValueError("Data file has more than two columns")

    df = pd.DataFrame(df.values, columns=column_names)
    for key, value in metadata.items():
        df[key] = value

    return df

def load_data(import_from_pickled=True):
    """
    Loads data from pickled file if available; otherwise, processes all .dat files and pickles the result.

    :param import_from_pickled: Boolean indicating whether to load data from a pickle file.
    :return: DataFrame containing combined data from all processed files.
    """
    if import_from_pickled:
        try:
            return pd.read_pickle(PICKLE_PATH).reset_index(drop=True)
        except FileNotFoundError:
            print(f"No pickle file found at {PICKLE_PATH}. Falling back to processing files.")
    
    data_frames_list = [process_file(filename) for filename in glob.glob("*.dat")]
    data_frames_list = [df for df in data_frames_list if df is not None]

    if not data_frames_list:
        raise FileNotFoundError("No valid data files found.")

    df = pd.concat(data_frames_list, ignore_index=True)
    df.to_pickle(PICKLE_PATH)
    return df
