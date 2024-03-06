import functools
import glob
import multiprocessing
import os
import re
from os.path import exists
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew, gaussian_kde
from scipy.odr import ODR, Model, Data, RealData
from jackknife import jackknife
from analyzerUtilities import load_data
from phasetransitions import calculate_transition

# Configuring Matplotlib
plt.rcParams['text.usetex'] = True
plt.style.use('bmh')

# Regular Expressions for File Identification
IDENTIFIER_PATTERN = r"m2_(-?\d+\.?\d*)beta(-?\d+\.?\d*)lambda(-?\d+\.?\d*)kappa(-?\d+\.?\d*)l(\d*)t(\d*)bc([ptc])\.dat"
GENERIC_PATTERN = rf"([a-zA-Z0-9]*){IDENTIFIER_PATTERN}"

def graph_pure_and_higgs(df_higgs, m2, betas_pure, string_tension_values_pure, string_tension_errors_pure):
    """
    Generates and saves a graph comparing pure and Higgs string tensions.
    
    Parameters:
    - df_higgs: DataFrame containing Higgs data.
    - m2: mass squared value for Higgs data filtering.
    - betas_pure: List of beta values for pure data.
    - string_tension_values_pure: List of string tension values for pure data.
    - string_tension_errors_pure: List of string tension errors for pure data.
    """
    beta_values_higgs = sorted(df_higgs['beta'].unique())
    betas_higgs, string_tension_values_higgs, string_tension_errors_higgs = [], [], []
    for beta in beta_values_higgs:
        obs_values = df_higgs[(df_higgs['beta'] == beta) & (df_higgs['m2'] == m2)]['Obs']
        jackknife_mean, jackknife_std = jackknife(obs_values, 10, np.mean)
        if jackknife_mean > 0 and jackknife_std / jackknife_mean < 0.1:
            betas_higgs.append(float(beta) / 4)
            string_tension_values_higgs.append(-np.log(jackknife_mean))
            string_tension_errors_higgs.append(jackknife_std / jackknife_mean)

    plt.figure(figsize=(5, 7.5))
    plt.errorbar(betas_pure, string_tension_values_pure, yerr=string_tension_errors_pure, fmt='o', capsize=2, markersize=2, label=r'Pure $SU(2)$')
    plt.errorbar(betas_higgs, string_tension_values_higgs, yerr=string_tension_errors_higgs, fmt='o', capsize=2, markersize=2, label=r'$m^2={}$'.format(m2.rstrip('0').rstrip('.')))
    plt.yscale('log')
    plt.yticks([0.1, 1.0, 10.0], [r"0.1", r"1.0", r"10"])
    plt.xticks([0.0, 0.25, 0.50, 0.75])
    plt.xlabel(r'$1/g^2$')
    plt.ylabel(r'$\chi(1,1)$')
    plt.legend()
    plt.savefig(f"string_tension{m2}.pdf")
    plt.close()

def graph_pure_and_higgs_parallel(df_higgs, m2_values, betas_pure, string_tension_values_pure, string_tension_errors_pure):
    """
    Parallelizes the graphing of pure and Higgs string tensions.
    
    Parameters:
    - df_higgs: DataFrame containing Higgs data.
    - m2_values: List of mass squared values for Higgs data filtering.
    - betas_pure: List of beta values for pure data.
    - string_tension_values_pure: List of string tension values for pure data.
    - string_tension_errors_pure: List of string tension errors for pure data.
    """
    with multiprocessing.Pool() as pool:
        partial_graph = functools.partial(graph_pure_and_higgs, df_higgs=df_higgs, betas_pure=betas_pure, string_tension_values_pure=string_tension_values_pure, string_tension_errors_pure=string_tension_errors_pure)
        pool.starmap(partial_graph, [(m2,) for m2 in m2_values])

def main():
    df =load_data(import_from_pickled=True)
    df_pure = df[(df['kappa'] == '0.000000') & (df['obs_name'] =='rect1x1' ) ].drop(columns=['Time','kappa']).reset_index(drop=True)
    df_higgs = df[(df['kappa'] == '1.000000') & (df['obs_name'] =='rect1x1' ) ].drop(columns=['Time','kappa']).reset_index(drop=True)
    beta_values_pure = sorted(df_pure['beta'].unique())
    
    m2_values_higgs = sorted(df_higgs['m2'].unique())
    for m2 in m2_values_higgs:
        for beta in df_higgs[df_higgs["m2"] == m2]['beta'].unique():
            if df_higgs[(df_higgs["m2"] == m2)&(df_higgs["beta"] == beta)].shape[0] < 100:
                m2_values_higgs.remove(m2)
                break

    betas_pure = []
    string_tension_values_pure = []
    string_tension_errors_pure = []

    for beta in beta_values_pure:
        obs_values = df_pure[df_pure['beta'] == beta]['Obs']
        jackknife_mean, jackknife_std = jackknife(obs_values, 10 , np.mean)
        if jackknife_mean > 0:
            betas_pure.append(float(beta)/4)
            string_tension_values_pure.append(-np.log(jackknife_mean))
            string_tension_errors_pure.append(jackknife_std/jackknife_mean)

    graphPureAndHiggs_parallel(df_higgs, m2_values_higgs, betas_pure, string_tension_values_pure, string_tension_errors_pure)
        

if __name__ == "__main__":
    main()