import os
import sys
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 

from scipy.ndimage import gaussian_filter1d
from EWS_functions import *
from scipy.stats import kendalltau
#measure time
start = time.time()

network = "1.0_1.0"
empirical_values = "0137"
folder = "../results/feedbacks/network_"+network+"/"+empirical_values
subfolders = np.sort(glob.glob(folder + "/0*"))

n = 4 # number of tipping elements
N = 1000 # number of surrogate series

step_size = 5
bandwidths = pd.read_csv("../results/sensitivity/network_1.0_1.0/0137/bandwidths.csv", index_col=0)

# Extract temperature rates and coupling strengths (should be equivalent for all ensemble runs)
trates = []
strengths = []
datafiles = np.sort(glob.glob(subfolders[1] + "/states_*.json"))
for f in datafiles:
        trate = np.round(float(re.split("_",re.split("Trate",f)[-1])[0]),5)
        strength = float(re.split("_",re.split("d",f)[-1])[0])
        trates.append(trate)
        strengths.append(strength)
elemnms = ["GIS", "THC", "WAIS", "AMAZ"] 

# Create empty dataframe for saving data
tuples = list(zip(*[np.sort(trates),strengths]))
index = pd.MultiIndex.from_tuples(tuples, names=["trate", "strength"])
tau_ac = pd.DataFrame(index=index, columns=elemnms)
pv_ac = pd.DataFrame(index=index, columns=elemnms)
tau_var = pd.DataFrame(index=index, columns=elemnms)
pv_var = pd.DataFrame(index=index, columns=elemnms)

for subfolder in [subfolders[1]]:
    # Start saving structure
    subname = re.split(folder+"/", subfolder)[-1]
    
    try:
        os.stat("postprocessed")
    except:
        os.mkdir("postprocessed")
    try:
        os.stat("postprocessed/{}".format(subname))
    except:
        os.mkdir("postprocessed/{}".format(subname))

    datafiles = np.sort(glob.glob(subfolder + "/states_*.json"))
    for f in datafiles:
        # Get temperature rate and coupling strength
        trate = np.round(float(re.split("_",re.split("Trate",f)[-1])[0]),5)
        strength = float(re.split("_",re.split("d",f)[-1])[0])
        # Get temperature increase starting point
        start_point = int(re.split("_",re.split("tstart",f)[-1])[0])
    
        #Open dataset 
        data = json.load(open(f))
        print("Data opened")
    
        for elem in range(n):
            
            # Select detrending window
            detrend_window = max(int((len(data[elem])-start_point)/2),50)
            # Select filtering bandwidth
            bandwidth = bandwidths[elemnms[elem]].loc[trate]

            # Take residual and choose starting point
            states = np.array(even_number(data[elem][start_point-detrend_window:]))
            A = states - gaussian_filter1d(states, bandwidth)
            
            # Calculate autocorrelation and variance of original series
            autocorr = calc_autocorrelation(A, step_size, detrend_window)
            variance = calc_variance(A, step_size, detrend_window)
            tau_autocorr = kendalltau(autocorr, np.arange(len(autocorr)))[0]
            tau_variance = kendalltau(variance, np.arange(len(autocorr)))[0]
            
            # Find the signifiance of the kendall taus by generating surrogates
            p_ac = kendall_tau_test_Boers(autocorr, N, tau_autocorr)
            p_var = kendall_tau_test_Boers(variance, N, tau_variance)

            tau_ac.loc[(trate, strength), elemnms[elem]] = tau_autocorr
            pv_ac.loc[(trate, strength), elemnms[elem]] = p_ac
            tau_var.loc[(trate, strength), elemnms[elem]] = tau_variance
            pv_var.loc[(trate, strength), elemnms[elem]] = p_var

    tau_ac.to_csv("postprocessed/"+subname+"/tau_ac.csv")
    pv_ac.to_csv("postprocessed/"+subname+"/pvalue_ac.csv")
    tau_var.to_csv("postprocessed/"+subname+"/tau_var.csv")
    pv_var.to_csv("postprocessed/"+subname+"/pvalue_var.csv")
    
end = time.time()
print("Time elapsed until Finish: {}s".format(end - start))

