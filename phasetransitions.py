import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sqlite3
from scipy.optimize import curve_fit
import re
from pathlib import Path
from os.path import exists
import os
from jackknife import jackknife
from analyzerUtilities import *
from scipy.stats import kurtosis, skew,gaussian_kde
from scipy.odr import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.signal as signal
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

plt.rcParams['text.usetex'] = True
plt.style.use('bmh')

identifierPattern = r"m2_(-?\d+\.?\d*)beta(-?\d+\.?\d*)lambda(-?\d+\.?\d*)kappa(-?\d+\.?\d*)l(\d*)t(\d*)bc([ptc])\.dat"
genericPattern = rf"([a-zA-Z0-9]*){identifierPattern}"

def detect_symmetry_breaking(data, skewness_threshold=0.1, kurtosis_threshold=3, peak_prominence_ratio=0.2):
    # Calculate skewness and kurtosis
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)

    # Apply Kernel density estimation
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    y = kde(x)
    
    # Find peaks using prominence criterion
    peaks, properties = signal.find_peaks(y, prominence=(peak_prominence_ratio * y.max(), None))

    # Determine if it's bimodal
    is_bimodal = len(peaks) >= 2

    # Check if the data is symmetric
    is_symmetric = abs(data_skewness) < skewness_threshold

    # Check if the data has multiple peaks
    has_multiple_peaks = data_kurtosis > kurtosis_threshold

    # If the data is not symmetric or has multiple peaks, it's considered as symmetry breaking
    return not is_symmetric or has_multiple_peaks or is_bimodal


def generate_histogram(params):
    (beta, m2), group_data = params
    x = group_data['Obs']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylabel(r'$N$')
    if (float(beta) >= 2.0 and float(beta) <= 3.0 and len(x) < 5000):
        # write m2 and beta to file
        with open('m2_values.txt', 'a') as f:
            f.write(f'{m2}')
        with open('beta_values.txt', 'a') as f:
            f.write(f'{beta}')
        print(f'{beta} {m2}')
    if detect_symmetry_breaking(x):
        ax.set_title(r'$\beta = {}$ $m^2 = {}$ SB '.format(float(beta), float(m2)))
    else:
        ax.set_title(r'$\beta = {}$ $m^2 = {}$ '.format(float(beta), float(m2)))
    ax.hist(x)
    fig.savefig(f'polyakovBeta_{beta}m2_{m2}.png')
    plt.close(fig)

def find_phase_transition_points(df, beta_min=2, beta_max=3):
    phase_transitions = []

    # Group data by 'beta' values
    beta_groups = df.groupby('beta')

    for beta, beta_group in beta_groups:
        beta_float = float(beta)
        if beta_min <= beta_float <= beta_max:
            m2_values = beta_group['m2'].unique()
            sorted_m2_values = sorted(m2_values, key=float)

            transition_m2 = None
            last_non_symmetric_m2 = None
            last_symmetric_m2 = None
            found_transition = False

            # Iterate over sorted 'm2' values
            for m2 in sorted_m2_values:
                m2_group = beta_group[beta_group['m2'] == m2]
                data = m2_group['Obs']
                if detect_symmetry_breaking(data):
                    last_non_symmetric_m2 = m2
                else:
                    last_symmetric_m2 = m2
                if last_non_symmetric_m2 is not None and last_symmetric_m2 is not None:
                    transition_m2 = (float(m2) + float(last_non_symmetric_m2)) / 2
                    uncertainty = np.abs(float(m2) - float(last_non_symmetric_m2)) / 2
                    phase_transitions.append((beta, transition_m2, uncertainty))
                    break

    return phase_transitions

def plot_phase_diagram(phase_transitions):
    beta_values, m2_transition_values, uncertainties = zip(*phase_transitions)

    fig, ax = plt.subplots()
    ax.errorbar(beta_values, m2_transition_values, yerr=uncertainties, fmt='o', capsize=5,label='Center Symmetry Phase Transition')

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$m^2$')
    ax.set_title('Phase diagram: Center Symmetry')
    ax.legend()

    plt.savefig('phase_diagram_center_symmetry.png')
    plt.close()

def findPhaseTransition(lst):
    x = np.diff(np.array(lst))
    max_val = np.max(np.abs(x))
    max_idx = np.argmax(np.abs(x))
    if max_val > 0.1:
        return max_idx, max_val
    else:
        print("not enough of a change")
        return None, None

def find_phase_transition_higgs_square(df_higgs_square, beta_min=2.0, beta_max=3.0):
    phase_transitions = []
    beta_list = np.flip(df_higgs_square['beta'].unique())

    for beta in [x for x in beta_list if beta_min <= float(x) <= beta_max]:
        x_beta = []
        y_m2 = []
        y_unc = []

        for m2 in df_higgs_square[df_higgs_square['beta'] == beta]['m2'].unique():
            if m2 == '0.000000':
                continue

            x = df_higgs_square[(df_higgs_square['beta'] == beta) & (df_higgs_square['m2'] == m2)]['Obs']

            if len(x) < 20:
                continue

            ave, err = jackknife(x, 10, np.mean)
            x_beta.append(float(m2))
            y_m2.append(ave)
            y_unc.append(np.abs(err))

        x_transition, y_transition = findPhaseTransition(y_m2)
        if x_transition is not None:
            phase_transitions.append((beta, x_beta[x_transition], x_beta[x_transition + 1], y_unc[x_transition]))

    return phase_transitions

# The rest of the code remains the same.
def generate_new_beta_m2_values(phase_transitions, num_points=5, beta_spacing=0.1, m2_range_factor=1):
    new_beta_m2_values = []
    unique_beta_values = sorted(set(float(beta) for beta, _, _ in phase_transitions))

    # Generate new beta values
    new_beta_values = np.arange(unique_beta_values[0], unique_beta_values[-1], beta_spacing)

    # Interpolate m2 values for the new beta values
    beta_values, m2_values, uncertainties = zip(*phase_transitions)
    beta_values = [float(x) for x in beta_values]
    m2_values = [float(x) for x in m2_values]
    uncertainties = [float(x) for x in uncertainties]

    m2_interpolator = interp1d(beta_values, m2_values, kind='linear', fill_value='extrapolate')

    for beta_val in new_beta_values:
        m2_val = m2_interpolator(beta_val)
        m2_uncertainty = uncertainties[beta_values.index(min(beta_values, key=lambda x: abs(x - beta_val)))]

        m2_min = m2_val - m2_uncertainty * m2_range_factor
        m2_max = m2_val + m2_uncertainty * m2_range_factor

        m2_values_to_test = np.linspace(m2_min, m2_max, num_points)

        for m2_val_to_test in m2_values_to_test:
            new_beta_m2_values.append((beta_val, m2_val_to_test))

    return new_beta_m2_values

def write_values_to_files(beta_values, m2_values, beta_filename="beta_values.txt", m2_filename="m2_values.txt"):
    with open(beta_filename, 'w') as beta_file:
        for beta in beta_values:
            beta_file.write(str(round(beta, 5)) + '\n')

    with open(m2_filename, 'w') as m2_file:
        for m2 in m2_values:
            m2_file.write(str(m2) + '\n')

def phaseTransitionAnalyzeAndPlots():
    df = load_data(import_from_pickled=True)
    # Add any further analysis or plotting code here
    df_polyakov = df[(df['obs_name'] == 'polyakov') & (df['kappa'] =='1.000000')].reset_index(drop=True).drop(columns=['obs_name','Time','kappa'])
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    fig, ax = plt.subplots()
    # Plot phase diagram
    #center symmetry plot
    phase_transitions = find_phase_transition_points(df_polyakov)
    
    beta_values, m2_transition_values, uncertainties = zip(*phase_transitions)
    beta_values = [float(x) for x in beta_values]
    m2_transition_values = list(m2_transition_values)
    uncertainties = list(uncertainties)
    ax.errorbar(beta_values, m2_transition_values, yerr=uncertainties, fmt='o', capsize=10,markersize=8,label='Center Symmetry Phase Transition',color=CB_color_cycle[0])


    # higgs transition plot
    df_higgs_square = df[(df['obs_name'] == 'higgsSquare') & (df['kappa'] == '1.000000')].reset_index(drop=True).drop(columns=['obs_name', 'Time', 'kappa'])
    phase_transitions = find_phase_transition_higgs_square(df_higgs_square)

    beta_values, m2_start_values, m2_end_values, uncertainties = zip(*phase_transitions)
    beta_values = [float(x) for x in beta_values]
    uncertainties = list(uncertainties)
    m2_avg_values = [(start + end) / 2 for start, end in zip(m2_start_values, m2_end_values)]
    ax.errorbar(beta_values, m2_avg_values, yerr=uncertainties, fmt='o', capsize=10,markersize=8,label='Higgs Phase Transition',color=CB_color_cycle[1])


    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$m^2$')
    ax.set_xlim(1.9, 3.1)
    ax.set_title('Phase diagram')
    ax.legend()
    plt.savefig('phase_diagram_center_symmetry_and_higgs.png')
    plt.close()
    # print("starting analysis")
    grouped = df[df['kappa']  == '1.000000'].groupby(['beta', 'm2'])
    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     executor.map(generate_histogram, grouped)
    exit(1)
    for group in grouped:
        generate_histogram(group)