import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
from pathlib import Path
from os.path import exists
import os

plt.rcParams['text.usetex'] = True
plt.style.use('bmh')

identifierPattern = r"m2_(-?\d+\.?\d*)beta(-?\d+\.?\d*)lambda(-?\d+\.?\d*)kappa(-?\d+\.?\d*)l(\d*)t(\d*)bc([ptc])\.dat"
genericPattern = rf"([a-zA-Z0-9]*){identifierPattern}"

def match_filename(filename):
    match = re.match(genericPattern, filename)
    if match:
        obs_name, m2, beta, lam, kappa, l, t, bc = match.groups()
        return {
            'obs_name': obs_name,
            'm2': m2,
            'beta': beta,
            'lam': lam,
            'kappa': kappa,
            'l': l,
            't': t,
            'bc': bc
        }
    return None

def process_file(filename):
    d = match_filename(filename)
    if not d:
        return None

    df = pd.read_csv(filename, sep=" ", header=None)

    if df.shape[1] == 1:
        obs_col = np.array(df.iloc[:, 0].tolist())
        df = pd.DataFrame({'Obs': obs_col})
        df = df.explode(['Obs'], ignore_index=True)
    elif df.shape[1] == 2:
        time_col = np.array(df.iloc[:, 0].tolist())
        obs_col = np.array(df.iloc[:, 1].tolist())
        df = pd.DataFrame({'Time': time_col, 'Obs': obs_col})
        df = df.explode(['Obs', 'Time'], ignore_index=True)
    else:
        raise ValueError("Data has more than two columns")

    for key, value in d.items():
        df[key] = value
    return df

def load_data(import_from_pickled=True, pickle_path="data.pkl"):
    if not import_from_pickled:
        data_frames_list = [process_file(filename) for filename in glob.glob("*.dat")]
        data_frames_list = [df for df in data_frames_list if df is not None]

        df = pd.concat(data_frames_list, ignore_index=True)
        df.to_pickle(pickle_path)
    else:
        df = pd.read_pickle(pickle_path).reset_index(drop=True)

    return df