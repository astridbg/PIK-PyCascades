import os
import sys
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from EWS_functions import *
from scipy.stats import kendalltau

def even_number(a):
    if (len(a) % 2) == 0:
        return a
    else:
        return a[:-1]
        print ("The length of the series is an odd number")

def delta(k, N):
    if k == 0 or k == N/2:
        return 1
    else:
        return 0

def discrete_Fourier(a):
    N = len(a)
    k_list = np.arange(0, N/2+1)
    a_k = []
    for k in k_list:
        s = 0
        for j in range(N):
            s += a[j] * np.exp(2*np.pi*1j*j*k/N)
        a_k.append( (2-delta(k,N))/N * s )
    return a_k

def random_phase(a_k):
    
    N_k = len(a_k)
    r_k = []
    r_k.append(0)
    for k in range(1,N_k-1):
        theta = np.random.uniform(0,2*np.pi)
        r_k.append(abs(a_k[k]) * np.exp(1j*theta))
    theta = np.random.uniform(0,2*np.pi)
    r_k.append( 2**0.5 * abs(a_k[N_k-1]) * np.cos(theta) )
    return r_k

def inverse_discrete_Fourier(r_k, N):
    
    N_k = len(r_k)
    r = []

    for j in range(N):
        s = 0
        for k in range(N_k):
            s += r_k[k] * np.exp (-2*np.pi*1j*j*k/N)
        r.append(s.real)
    
    return r

network = "1.0_1.0"
empirical_values = "0137"
folder = "../results/feedbacks/network_"+network+"/"+empirical_values
n = 4 # number of tipping elements
N = 10 # number of surrogate series

bandwidth = [20, 20, 20, 20]
step_size = 10

subfolders = np.sort(glob.glob(folder + "/*"))
print(subfolders)

# Extract temperature rates and coupling strengths (should be equivalent for all ensemble runs)
trates = []
strengths = []
datafiles = np.sort(glob.glob(subfolders[1] + "/states_*.json"))
for f in datafiles:
        trate = re.split("_",re.split("Trate",f)[-1])[0]
        strength = re.split("_",re.split("d",f)[-1])[0]
        trates.append(trate)
        strengths.append(strength)
elemnms = ["GIS", "THC", "WAIS", "AMAZ"] 
# Create empty dataframe for each tipping element
tuples = list(zip(*[np.sort(trates),strengths]))
index = pd.MultiIndex.from_tuples(tuples, names=["trate", "strength"])
tau_ac = pd.DataFrame(index=index, columns=elemnms)
pv_ac = pd.DataFrame(index=index, columns=elemnms)
tau_var = pd.DataFrame(index=index, columns=elemnms)
pv_var = pd.DataFrame(index=index, columns=elemnms)

for subfolder in subfolders:
    # Start saving structure
    subname = re.split(folder+"/", subfolder)[-1]

    try:
        os.stat("postprocessed/{}".format(subname))
    except:
        os.mkdir("postprocessed/{}".format(subname))

    datafiles = np.sort(glob.glob(subfolder + "/states_*.json"))
    for f in datafiles:
        # Get temperature rate and coupling strength
        trate = re.split("_",re.split("Trate",f)[-1])[0]
        strength = re.split("_",re.split("d",f)[-1])[0]
        # Get temperature increase starting point
        start_point = int(re.split("_",re.split("tstart",f)[-1])[0])
    
        #Open dataset 
        data = json.load(open(f))
        print("Data opened")
    
        for elem in range(n):
            
            # Select detrending window
            detrend_window = int((len(data[elem])-start_point)/2)
        
            # Take residual and choose starting point
            states = np.array(even_number(data[elem][start_point-detrend_window:]))
            A = states - gaussian_filter1d(states, bandwidth[elem])
            autocorr = calc_autocorrelation(A, step_size, detrend_window)
            variance = calc_variance(A, step_size, detrend_window)

            tau_autocorr = kendalltau(autocorr, np.arange(len(autocorr)))[0]
            tau_variance = kendalltau(variance, np.arange(len(autocorr)))[0]

            p_ac = kendall_tau_test(autocorr, N, tau_autocorr)
            p_var = kendall_tau_test(variance, N, tau_variance)
            
            tau_ac.loc[(trate, strength), elemnms[elem]] = tau_autocorr
            pv_ac.loc[(trate, strength), elemnms[elem]] = p_ac
            tau_var.loc[(trate, strength), elemnms[elem]] = tau_variance
            pv_var.loc[(trate, strength), elemnms[elem]] = p_var

            # Plotting
            # plt.figure()
            # plt.hist(tau_autocorr, bins=5)
            # plt.axvline(o_tau_autocorr, c='r',linestyle='--', label='Original')
            # plt.xlabel(r"Kendall $\tau$ correlation")
            # plt.ylabel("Frequency")
            # plt.legend(loc='best')
            # plt.show()
    
    tau_ac.to_csv("postprocessed/"+subname+"/tau_ac.csv")
    pv_ac.to_csv("postprocessed/"+subname+"/pvalue_ac.csv")
    tau_var.to_csv("postprocessed/"+subname+"/tau_var.csv")
    pv_var.to_csv("postprocessed/"+subname+"/pvalue_var.csv")
    

