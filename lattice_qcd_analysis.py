import glob
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.odr import Model, RealData, ODR
from scipy.stats import kurtosis, skew
from jackknife import jackknife
from analyzerUtilities import match_filename
from fitFunctions import *

# Configure Matplotlib for LaTeX integration and style
plt.rcParams['text.usetex'] = True
plt.style.use('bmh')


def transposeListOfList(lst):
    return [list(x) for x in zip(*lst)]

def specialFitting(lst):
    x = transposeListOfList(lst)
    T = np.arange(0,10,1)
    aves = []
    for i in range(10):
        ave = np.mean(x[i])
        aves.append(ave)
    # remove t and x if they are less than zero
    aves = np.array(aves)
    mask = aves > 0
    aves = aves[mask]
    T = T[mask]
    try:
        coeffs = np.polyfit(T,np.log(aves),1)
    except np.linalg.LinAlgError:
        print('Error: Singular matrix encountered during polyfit')
        return np.nan
    except np.RankWarning as e:
        print(f"Warning: {e}")
        return np.nan
    except Exception as e:
        print(f"Error: {e}")
        return np.nan

    return -coeffs[1]

# higgs mass as a function of m^2
df_higgsMass = df[(df['obs_name'] == 'higgsCorr') & (df['kappa'] =='1.000000')].reset_index(drop=True).drop(columns=['obs_name','kappa'])
for beta in df_higgsMass['beta'].unique():
    fig,ax = plt.subplots(1,1,figsize=(8,6))
    ax.set_title(r'Higgs mass $\beta = {}$'.format(beta))
    ax.set_xlabel(r'$m^2$')
    ax.set_ylabel(r'$M_{Higgs}$')
    workingm2 = []
    M = []
    Merr = []
    for m2 in df_higgsMass[df_higgsMass['beta'] == beta]['m2'].unique().tolist():
        times = df_higgsMass[df_higgsMass['beta'] == beta]['Time'].unique().tolist()
        Y = []
        for t in times:
            y =  df_higgsMass[(df_higgsMass['beta'] == beta) & (df_higgsMass['Time'] == t) & (df_higgsMass['m2'] == m2)]['Obs'].tolist()
            if(len(y) < 20):
                times.remove(t)
                continue
            Y.append(y)
        if len(Y) == 0:
            continue
        ave,err = jackknife(transposeListOfList(Y),10,specialFitting)
        if ave == np.nan or ave == np.inf or ave == -np.inf or ave == 0.0  or ave == None :
            continue
        if err == np.nan or err == np.inf or err == -np.inf or err == 0.0  or err == None :
            continue
        M.append(ave)
        Merr.append(err)
        workingm2.append(float(m2))
    if M == []:
        continue
    plt.errorbar(workingm2,M,yerr=Merr,fmt='o')
    plt.savefig(f'higgsMassBeta_{beta}.png')
    plt.close()



def findPhaseTransition(lst):
    x = np.diff(np.array(lst))
    max_val = np.max(np.abs(x))
    max_idx = np.argmax(np.abs(x))
    if max_val > 0.1:
        return max_idx,max_val
    else:
        print("not enough of a change")
        return None,None
    
open('beta_values.txt','w').close()
open('m2_values.txt','w').close()
x_beta = []
y_m2 = []
y_unc = []
# Go through beta values, order by m2, search list for phase transition
df_higgsSquare = df[(df['obs_name'] == 'higgsSquare') & (df['kappa'] =='1.000000')].reset_index(drop=True).drop(columns=['obs_name','Time','kappa'])
betaList = np.flip(df_higgsSquare['beta'].unique())
for beta in [x for x in betaList if float(x) <= 3.0 and float(x) >= 2.0]:
    X = []
    Y = []
    Yerr = []
    for m2 in df_higgsSquare[df_higgsSquare['beta'] == beta]['m2'].unique():
        if m2 == '0.000000':
            continue
        x = df_higgsSquare[(df_higgsSquare['beta'] == beta) & (df_higgsSquare['m2'] ==m2)]['Obs']
        if(len(x) < 20):
            continue
        ave,err = jackknife(x,10,np.mean)
        X.append(float(m2))
        Y.append(ave)
        Yerr.append(np.abs(err))
    x,y = findPhaseTransition(Y)
    if x == None:
        continue
    else:
        print(f'Phase transition bewteen m^2 = {X[x]} and m^2 = {X[x+1]} for beta = {beta}')
        if float(beta) <= 3.0 and float(beta) >= 2.0:
            m2range = np.linspace(X[x],X[x+1],5)
            # for m2 in m2range:
            #     beta_file = open("beta_values.txt","a")
            #     beta_file.write(f"{beta}\n")
            #     beta_file.close()
            #     m2_file = open("m2_values.txt","a")
            #     m2_file.write(f"{m2}\n")
            #     m2_file.close()
        x_beta.append(float(beta))
        y_m2.append((X[x] + X[x+1])/2.0)
        y_unc.append(np.abs(X[x] - X[x+1])/2.0)

fig,ax = plt.subplots(1,1,figsize=(8,6))
ax.set_title(r'Phase transition')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$m^2$')
ax.errorbar(x_beta,y_m2,yerr=y_unc,fmt='o')

# Fit to linear function
p0 = [1.0,1.0]
lin_model = Model(fitFuncLinear)
dataForFit = RealData(x_beta,y_m2,sy=y_unc)
fit = ODR(dataForFit,lin_model,beta0=p0)
out = fit.run()
out.pprint()
x_fit = np.linspace(2.0,3.0,100)
y_fit = fitFuncLinear(out.beta,x_fit)
# ax.plot(x_fit,y_fit)

# Fit to sigmoidal function
p0,_ = curve_fit(curveFitSigmoidal,x_beta,y_m2)
sig_model = Model(fitSigmoidal)
dataForFit = RealData(x_beta,y_m2,sy=y_unc)
fit = ODR(dataForFit,sig_model,beta0=p0,maxit=1000)
out = fit.run()
out.pprint()
x_fit = np.linspace(2.0,3.0,100)
y_fit = fitSigmoidal(out.beta,x_fit)
ax.plot(x_fit,y_fit)


plt.savefig(f'phaseTransition.png')
print(betaList)
tempBetas = np.sort(np.array([float(x) for x in betaList if float(x) <= 3.0 and float(x) >= 2.0])) 
print(tempBetas)
betasToAnalyse = np.array([])
for i in range(len(tempBetas)-1):
    betasToAnalyse = np.concatenate((betasToAnalyse,np.arange(tempBetas[i],tempBetas[i+1],0.1)[1:]))
    print(betasToAnalyse)

for beta in np.sort(betasToAnalyse):
    print(beta)
    m2_mid = fitSigmoidal(out.beta,beta)
    m2_range = np.linspace(m2_mid-0.2,m2_mid+0.2,5)
    for m2 in m2_range:
        beta_file = open("beta_values.txt","a")
        beta_file.write(f"{beta}\n")
        beta_file.close()
        m2_file = open("m2_values.txt","a")
        m2_file.write(f"{m2}\n")
        m2_file.close()






#Polyakov histogram for different m^2 and beta
df_polyakov = df[(df['obs_name'] == 'polyakov') & (df['kappa'] =='1.000000')].reset_index(drop=True).drop(columns=['obs_name','Time','kappa'])
for beta in df_polyakov['beta'].unique():
    for m2 in df_polyakov[df_polyakov['beta'] == beta]['m2'].unique():
        fig,ax = plt.subplots(1,1,figsize=(8,6))

        ax.set_ylabel(r'$N$')
        x = df_polyakov[(df_polyakov['beta'] == beta) & (df_polyakov['m2'] == m2) ]['Obs']
        ax.set_title(r'$\beta = {}$ $m^2 = {}$ $\mu = {}$ $\sigma^2 = {}$ sk = {} k = {}'.format(float(beta),float(m2),np.round(np.mean(x),2),np.round(np.var(x),2),np.round(kurtosis(x),2),np.round(skew(x),2)))  
        ax.hist(x)
        plt.savefig(f'polyakovBeta_{beta}m2_{m2}.png')
        plt.close()