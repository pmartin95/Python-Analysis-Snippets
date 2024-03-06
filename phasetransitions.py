import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal
from scipy.stats import kurtosis, skew, gaussian_kde
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from jackknife import jackknife
from analyzerUtilities import load_data
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Set global matplotlib parameters
plt.rcParams['text.usetex'] = True
plt.style.use('bmh')

# Patterns for file identification
IDENTIFIER_PATTERN = r"m2_(-?\d+\.?\d*)beta(-?\d+\.?\d*)lambda(-?\d+\.?\d*)kappa(-?\d+\.?\d*)l(\d*)t(\d*)bc([ptc])\.dat"
GENERIC_PATTERN = rf"([a-zA-Z0-9]*){IDENTIFIER_PATTERN}"

def detect_symmetry_breaking(data, skewness_threshold=0.1, kurtosis_threshold=3, peak_prominence_ratio=0.2):
    """
    Detects symmetry breaking in data based on skewness, kurtosis, and peak prominence.
    Returns True if symmetry breaking conditions are met.
    """
    # Calculate skewness, kurtosis, and apply Kernel density estimation
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    y = kde(x)
    
    # Find peaks and determine bimodality
    peaks, _ = signal.find_peaks(y, prominence=(peak_prominence_ratio * y.max(), None))
    is_bimodal = len(peaks) >= 2
    is_symmetric = abs(data_skewness) < skewness_threshold
    has_multiple_peaks = data_kurtosis > kurtosis_threshold
    
    return not is_symmetric or has_multiple_peaks or is_bimodal


def generate_histogram(params):
    """
    Generates and saves a histogram for a given set of parameters.
    Writes `m2` and `beta` values to files if certain conditions are met.

    Parameters:
    - params: A tuple containing (beta, m2) and group_data as a DataFrame.
    """
    (beta, m2), group_data = params
    observations = group_data['Obs']

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylabel(r'$N$')
    
    # Condition to check if beta is within a specific range and the number of observations is less than 5000
    if 2.0 <= float(beta) <= 3.0 and len(observations) < 5000:
        # Write m2 and beta values to files
        with open('m2_values.txt', 'a') as f:
            f.write(f'{m2}\n')
        with open('beta_values.txt', 'a') as f:
            f.write(f'{beta}\n')
        print(f'beta: {beta}, m2: {m2}')
    
    # Set title based on whether symmetry breaking is detected
    title_suffix = "SB" if detect_symmetry_breaking(observations) else ""
    ax.set_title(rf'$\beta = {float(beta)}$ $m^2 = {float(m2)}$ {title_suffix}')
    ax.hist(observations)
    fig.savefig(f'polyakovBeta_{beta}m2_{m2}.png')
    plt.close(fig)

def find_phase_transition_points(dataframe, beta_min=2, beta_max=3):
    """
    Identifies phase transition points in the given dataset based on `beta` values.

    Parameters:
    - dataframe: The dataset containing observations.
    - beta_min: The minimum `beta` value to consider for phase transitions.
    - beta_max: The maximum `beta` value to consider for phase transitions.

    Returns:
    - A list of tuples, each containing (`beta`, transition `m2` value, and uncertainty).
    """
    phase_transitions = []
    beta_groups = dataframe.groupby('beta')

    for beta, group in beta_groups:
        beta_value = float(beta)
        if beta_min <= beta_value <= beta_max:
            m2_values = group['m2'].unique()
            sorted_m2 = sorted(m2_values, key=float)

            for m2_index in range(len(sorted_m2) - 1):
                current_m2 = sorted_m2[m2_index]
                next_m2 = sorted_m2[m2_index + 1]
                current_group = group[group['m2'] == current_m2]
                next_group = group[group['m2'] == next_m2]

                # Check for symmetry breaking in current and next m2 group
                if detect_symmetry_breaking(current_group['Obs']) != detect_symmetry_breaking(next_group['Obs']):
                    transition_m2 = (float(current_m2) + float(next_m2)) / 2
                    uncertainty = np.abs(float(current_m2) - float(next_m2)) / 2
                    phase_transitions.append((beta, transition_m2, uncertainty))
                    break

    return phase_transitions

def plot_phase_diagram(phase_transitions):
    """
    Plots the phase diagram using the provided phase transitions data.

    Parameters:
    - phase_transitions: A list of tuples containing (beta, m2 transition value, uncertainty).
    """
    beta_values, m2_transition_values, uncertainties = zip(*phase_transitions)

    fig, ax = plt.subplots()
    ax.errorbar(beta_values, m2_transition_values, yerr=uncertainties, fmt='o', capsize=5, label='Center Symmetry Phase Transition')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$m^2$')
    ax.set_title('Phase Diagram: Center Symmetry')
    ax.legend()
    plt.savefig('phase_diagram_center_symmetry.png')
    plt.close()


def find_phase_transition(lst):
    """
    Identifies the index and value of the maximum change in a list, indicating a phase transition.

    Parameters:
    - lst: List of numerical values representing observed data.

    Returns:
    - A tuple containing the index of the maximum change and the value of the change if significant; otherwise, None.
    """
    differences = np.diff(np.array(lst))
    max_change = np.max(np.abs(differences))
    max_index = np.argmax(np.abs(differences))
    
    if max_change > 0.1:
        return max_index, max_change
    else:
        return None, None


def find_phase_transition_higgs_square(dataframe, beta_min=2.0, beta_max=3.0):
    """
    Identifies phase transition points for the Higgs square observable within a specified beta range.

    Parameters:
    - dataframe: DataFrame containing beta, m2 values, and observations.
    - beta_min: The minimum beta value to consider.
    - beta_max: The maximum beta value to consider.

    Returns:
    - A list of tuples, each containing the beta value, the start and end m2 values indicating the phase transition, and the uncertainty.
    """
    phase_transitions = []
    beta_list = np.flip(dataframe['beta'].unique())

    for beta in [b for b in beta_list if beta_min <= float(b) <= beta_max]:
        m2_values, means, uncertainties = [], [], []

        for m2 in dataframe[dataframe['beta'] == beta]['m2'].unique():
            if m2 == '0.000000':
                continue

            observations = dataframe[(dataframe['beta'] == beta) & (dataframe['m2'] == m2)]['Obs']
            if len(observations) < 20:
                continue

            mean, uncertainty = jackknife(observations, 10, np.mean)
            m2_values.append(float(m2))
            means.append(mean)
            uncertainties.append(abs(uncertainty))

        transition_index, _ = find_phase_transition(means)
        if transition_index is not None:
            phase_transitions.append((beta, m2_values[transition_index], m2_values[transition_index + 1], uncertainties[transition_index]))

    return phase_transitions


def generate_new_beta_m2_values(phase_transitions, num_points=5, beta_spacing=0.1, m2_range_factor=1):
    """
    Generates new (beta, m2) values for interpolation within the range of observed phase transitions.

    Parameters:
    - phase_transitions: A list of tuples containing (beta, m2 transition value, uncertainty).
    - num_points: The number of m2 values to generate for each new beta.
    - beta_spacing: The spacing between each new beta value to be generated.
    - m2_range_factor: A factor to adjust the range around the interpolated m2 value based on uncertainty.

    Returns:
    - A list of tuples with new (beta, m2) values.
    """
    # Extract unique and sorted beta values from the phase transitions
    unique_beta_values = sorted(set(beta for beta, _, _ in phase_transitions))

    # Generate new beta values within the range of unique betas
    new_beta_values = np.arange(unique_beta_values[0], unique_beta_values[-1] + beta_spacing, beta_spacing)

    # Unpack and prepare phase transition data for interpolation
    beta_values, m2_values, uncertainties = zip(*phase_transitions)
    beta_values, m2_values, uncertainties = map(np.array, [beta_values, m2_values, uncertainties])

    # Create an interpolator for m2 values across beta values
    m2_interpolator = interp1d(beta_values, m2_values, kind='linear', fill_value='extrapolate')

    new_beta_m2_values = []

    for beta_val in new_beta_values:
        # Interpolate m2 value and determine its uncertainty at the current beta
        m2_val = m2_interpolator(beta_val)
        nearest_beta_index = np.argmin(np.abs(beta_values - beta_val))
        m2_uncertainty = uncertainties[nearest_beta_index]

        # Calculate the range of m2 values to test based on the uncertainty
        m2_min, m2_max = m2_val - m2_uncertainty * m2_range_factor, m2_val + m2_uncertainty * m2_range_factor
        m2_values_to_test = np.linspace(m2_min, m2_max, num_points)

        # Append new (beta, m2) pairs for testing
        new_beta_m2_values.extend([(beta_val, m2_test) for m2_test in m2_values_to_test])

    return new_beta_m2_values


def write_values_to_files(beta_values, m2_values, beta_filename="beta_values.txt", m2_filename="m2_values.txt"):
    with open(beta_filename, 'w') as beta_file:
        for beta in beta_values:
            beta_file.write(str(round(beta, 5)) + '\n')

    with open(m2_filename, 'w') as m2_file:
        for m2 in m2_values:
            m2_file.write(str(m2) + '\n')

def phase_transition_analyze_and_plots():
    """
    Analyzes phase transitions and plots the phase diagrams for both
    center symmetry and Higgs phase transitions.
    """
    # Load data and preprocess
    df = load_data(import_from_pickled=True)
    df_polyakov = df.query("obs_name == 'polyakov' & kappa == '1.000000'").reset_index(drop=True).drop(columns=['obs_name', 'Time', 'kappa'])
    df_higgs_square = df.query("obs_name == 'higgsSquare' & kappa == '1.000000'").reset_index(drop=True).drop(columns=['obs_name', 'Time', 'kappa'])

    # Color cycle for plots
    cb_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    
    # Initialize plot
    fig, ax = plt.subplots()

    # Analyze and plot Center Symmetry Phase Transition
    phase_transitions_center = find_phase_transition_points(df_polyakov)
    plot_phase_transition(ax, phase_transitions_center, label='Center Symmetry Phase Transition', color=cb_color_cycle[0])

    # Analyze and plot Higgs Phase Transition
    phase_transitions_higgs = find_phase_transition_higgs_square(df_higgs_square)
    plot_higgs_phase_transition(ax, phase_transitions_higgs, color=cb_color_cycle[1])

    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$m^2$')
    ax.set_xlim(1.9, 3.1)
    ax.set_title('Phase Diagram: Center Symmetry and Higgs')
    ax.legend()

    plt.savefig('phase_diagram_center_symmetry_and_higgs.png')
    plt.close()

def plot_phase_transition(ax, phase_transitions, label, color):
    """
    Plots phase transitions on a given axis.

    Parameters:
    - ax: The matplotlib axis to plot on.
    - phase_transitions: List of phase transition data.
    - label: Label for the plot legend.
    - color: Color of the plot.
    """
    beta_values, m2_transition_values, uncertainties = zip(*phase_transitions)
    beta_values = [float(beta) for beta in beta_values]
    ax.errorbar(beta_values, m2_transition_values, yerr=uncertainties, fmt='o', capsize=10, markersize=8, label=label, color=color)

def plot_higgs_phase_transition(ax, phase_transitions_higgs, color):
    """
    Special handling for plotting Higgs phase transitions due to additional data transformation.

    Parameters:
    - ax: The matplotlib axis to plot on.
    - phase_transitions_higgs: List of Higgs phase transition data.
    - color: Color of the plot.
    """
    beta_values, m2_start_values, m2_end_values, uncertainties = zip(*phase_transitions_higgs)
    beta_values = [float(beta) for beta in beta_values]
    m2_avg_values = [(start + end) / 2 for start, end in zip(m2_start_values, m2_end_values)]
    ax.errorbar(beta_values, m2_avg_values, yerr=uncertainties, fmt='o', capsize=10, markersize=8, label='Higgs Phase Transition', color=color)