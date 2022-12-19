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

# Start saving structure
subname = re.split(folder+"/", subfolder)[-1]
try:
    os.stat("../results/sensitivity/network_{}/{}/{}".format(network, empirical_values,subname))
except:
    os.mkdir("../results/sensitivity/network_{}/{}/{}".format(network, empirical_values,subname))

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
bandwidths = pd.read_csv("../results/sensitivity/network_{}/{}/{}".format(network, empirical_values,subname)+"/bandwidths.csv",index_col=0)
#################################

datafiles = np.sort(glob.glob(subfolder + "/states_*.json"))
for f in datafiles:
    # Get temperature rate and coupling strength
    trate = np.round(float(re.split("_",re.split("Trate",f)[-1])[0]),5)
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
        # Get bandwidth
        bandwidth = bandwidths[labels[elem]].loc[trate]

        # Take residual and choose starting point
        states = np.array(even_number(data[elem][start_point-detrend_window:]))
        A = states - gaussian_filter1d(states, bandwidth)
        autocorr = calc_autocorrelation(A, step_size, detrend_window)
        variance = calc_variance(A, step_size, detrend_window)
        tau_autocorr = kendalltau(autocorr, np.arange(len(autocorr)))[0]
        tau_variance = kendalltau(variance, np.arange(len(variance)))[0]
        
        fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, figsize=(8,8), sharex=True)
        fig.suptitle(labels[elem]+", Window={}".format(detrend_window)+", Bandwidth={}".format(bandwidth))
        # Plot time window of states

        ax1.grid(True)
        ax1.plot(np.arange(len(states))+start_point-detrend_window, states, c=colors[elem])
        ax1.plot(np.arange(len(states))+start_point-detrend_window, gaussian_filter1d(states, bandwidth), c="gray")
        ax1.set_ylabel("System state [a.u.]")


        # Plot residual
        ax2.grid(True)
        ax2.plot(np.arange(len(states))+start_point-detrend_window, A, c=colors[elem])
        ax2.set_ylabel("System residual")


        # Plot autocorrelation
        ax3.grid(True)
        ax3.scatter(np.arange(0,len(autocorr)*step_size,step_size)+start_point, autocorr, label=labels[elem], c=colors[elem], s=2)
        ax3.set_ylabel("Autocorrelation")
        ax3.text(0.15, 0.1, "Kendall tau: {}".format(np.round(tau_autocorr, 2)),
                        horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

        # Plot variance
        ax4.grid(True)
        ax4.scatter(np.arange(0,len(variance)*step_size,step_size)+start_point, variance, label=labels[elem], c=colors[elem], s=2)
        ax4.set_xlabel("Time [yr]")
        ax4.set_ylabel(r"Variance")
        ax4.text(0.15, 0.1, "Kendall tau: {}".format(np.round(tau_variance, 2)),
                        horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

        fig.tight_layout()
        fig.savefig("../results/sensitivity/network_{}/{}/{}/EWS_elem{}_Trate{}_d{:.2f}.png".format(network, empirical_values, subname, elem, float(trate), float(strength)))
        plt.clf()
        plt.close()
