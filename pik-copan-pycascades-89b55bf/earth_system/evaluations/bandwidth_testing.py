import os
import sys
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)

from scipy.ndimage import gaussian_filter1d
from EWS_functions import *
from scipy.stats import kendalltau

################################
# Selection of data
################################
network = "1.0_1.0"
empirical_values = "0137"
folder = "../results/feedbacks/network_"+network+"/"+empirical_values
subfolder = folder+"/002"
#################################
# Specify tipping elements
#################################
n = 4 # number of tipping elements
colors = ['c','b','k','g']
labels = ["GIS", "THC", "WAIS", "AMAZ"]
#################################
# Early Warning Signal parameters
#################################
step_size = 5
# Bandwidths range for each individual tipping element
gis_bw = np.arange(10, 110,10)
thc_bw = np.arange(5,60,10)
wais_bw = np.arange(10,110,10)
amaz_bw = [2, 5] + list(np.arange(10,40,10))
bws = [gis_bw, thc_bw, wais_bw, amaz_bw]
#################################

# Extract temperature rates and coupling strengths (should be equivalent for all ensemble runs)
trates = []
datafiles = np.sort(glob.glob(subfolder + "/states_*.json"))
for f in datafiles:
        trate = re.split("_",re.split("Trate",f)[-1])[0]
        trates.append(np.round(float(trate),5))
trates = np.sort(list(set(trates)))
df_bws = pd.DataFrame(index=trates, columns=labels)

# Start saving structure
subname = re.split(folder+"/", subfolder)[-1]
try:
    os.stat("../results/sensitivity/network_{}/{}/{}".format(network, empirical_values,subname))
except:
    os.mkdir("../results/sensitivity/network_{}/{}/{}".format(network, empirical_values,subname))

datafiles = np.sort(glob.glob(subfolder + "/states_*.json"))
for f in datafiles:
    # Get temperature rate and coupling strength
    trate = float(re.split("_",re.split("Trate",f)[-1])[0])
    strength = float(re.split("_",re.split("d",f)[-1])[0])
    if strength != 0:
        continue
    # Get temperature increase starting point
    start_point = int(re.split("_",re.split("tstart",f)[-1])[0])
    
    #Open dataset 
    data = json.load(open(f))
    print("Data opened")
        
    for elem in range(n):
        # Select detrending window
        detrend_window = max(int((len(data[elem])-start_point)/2), 50)
        
        # Take residual and choose starting point
        states = np.array(even_number(data[elem][start_point-detrend_window:]))
        bandwidths = bws[elem]
        tau_autocorr = np.zeros(len(bandwidths))
        for i in range(len(bandwidths)):
            bandwidth = bandwidths[i]
            A = states - gaussian_filter1d(states, bandwidth)
            autocorr = calc_autocorrelation(A, step_size, detrend_window)
            tau_autocorr[i] = kendalltau(autocorr, np.arange(len(autocorr)))[0]
        
        bandwidth = bandwidths[np.argmax(tau_autocorr)]
        df_bws[labels[elem]].loc[trate] = bandwidth
        fig, [ax1, ax2, ax3] = plt.subplots(3,1)
        fig.suptitle(labels[elem]+", Temperature rate: {}".format(float(trate)))
        
        # Plot Kendall tau's variation with bandwidth
        ax1.scatter(bandwidths, tau_autocorr, c=colors[elem])
        ax1.set_xlabel("Bandwidths")
        ax1.set_ylabel(r"Kendall $\tau$ (autocorrelation)")
        
        # Plot states
        ax2.plot(np.arange(len(states))+start_point-detrend_window, states, c=colors[elem])
        ax2.plot(np.arange(len(states))+start_point-detrend_window, gaussian_filter1d(states,bandwidth), c=colors[elem])

        # Plot residual
        residual = states - gaussian_filter1d(states, bandwidth)
        ax3.grid(True)
        ax3.plot(np.arange(len(states))+start_point-detrend_window, residual, c=colors[elem])
        ax3.set_ylabel("Residual")

        fig.tight_layout()
        fig.savefig("../results/sensitivity/network_{}/{}/{}/bw_elem{}_Trate{}_d{:.2f}.png".format(network, empirical_values, subname, elem, trate, strength))
        plt.clf()
        plt.close()

df_bws.to_csv("../results/sensitivity/network_{}/{}/{}/bandwidths.csv".format(network, empirical_values, subname))
