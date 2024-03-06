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
from phasetransitions import *
from scipy.stats import kurtosis, skew,gaussian_kde
from scipy.odr import *
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.signal as signal
import multiprocessing
import functools
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

plt.rcParams['text.usetex'] = True
plt.style.use('bmh')

identifierPattern = r"m2_(-?\d+\.?\d*)beta(-?\d+\.?\d*)lambda(-?\d+\.?\d*)kappa(-?\d+\.?\d*)l(\d*)t(\d*)bc([ptc])\.dat"
genericPattern = rf"([a-zA-Z0-9]*){identifierPattern}"


def graphPureAndHiggs(df_higgs,m2,betas_pure,string_tension_values_pure,string_tension_errors_pure):
    beta_values_higgs = sorted(df_higgs['beta'].unique())
    betas_higgs = []
    string_tension_values_higgs = []
    string_tension_errors_higgs = []
    for beta in beta_values_higgs:
        obs_values = df_higgs[(df_higgs['beta'] == beta)&(df_higgs['m2'] == m2)]['Obs']
        jackknife_mean, jackknife_std = jackknife(obs_values, 10 , np.mean)
        if jackknife_mean:
            if jackknife_mean > 0 and jackknife_std/jackknife_mean < .1:
                betas_higgs.append(float(beta)/4)
                string_tension_values_higgs.append(-np.log(jackknife_mean))
                string_tension_errors_higgs.append(jackknife_std/jackknife_mean)

    plt.figure(figsize=(5,7.5))
    plt.errorbar(betas_pure, string_tension_values_pure, yerr=string_tension_errors_pure,fmt='o', capsize=2,markersize=2,label=r'Pure $SU(2)$')
    plt.errorbar(betas_higgs, string_tension_values_higgs, yerr=string_tension_errors_higgs, fmt='o', capsize=2, markersize=2, label=r'$m^2={}$'.format(m2.rstrip('0').rstrip('.')))
    plt.yscale('log')
    plt.yticks([0.1,1.0, 10.0,],[r"0.1",r"1.0", r"10"])
    plt.xticks([0.0,0.25,0.50,0.75])
    plt.xlabel(r'$1/g^2$')
    plt.ylabel(r'$\chi(1,1)$')
    plt.legend()
    plt.savefig(r"string_tension{}.pdf".format(m2))
    plt.close()

def graphPureAndHiggs_parallel(df_higgs, m2_values, betas_pure, string_tension_values_pure, string_tension_errors_pure):
    pool = multiprocessing.Pool()
    partial_graphPureAndHiggs = functools.partial(graphPureAndHiggs, betas_pure=betas_pure, string_tension_values_pure=string_tension_values_pure, string_tension_errors_pure=string_tension_errors_pure)
    pool.starmap(partial_graphPureAndHiggs, zip([df_higgs]*len(m2_values), m2_values))
    pool.close()
    pool.join()


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