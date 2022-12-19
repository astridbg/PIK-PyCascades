import os
import sys
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
import seaborn as sns
sns.set(font_scale=1.)

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
# import bandwidths found from testing
bandwidths = pd.read_csv("../results/sensitivity/network_1.0_1.0/0137/bandwidths.csv", index_col=0)

elemnms = ["GIS", "THC", "WAIS", "AMAZ"]

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
    
    # choose which datafile(s) (temprate + coupling strength) to consider 
    datafiles = [datafiles[-2]]
    # choose which tipping element(s) to consider
    elements = [3]

    for f in datafiles:
        # Get temperature rate and coupling strength
        trate = np.round(float(re.split("_",re.split("Trate",f)[-1])[0]),5)
        strength = float(re.split("_",re.split("d",f)[-1])[0])
        # Get temperature increase starting point
        start_point = int(re.split("_",re.split("tstart",f)[-1])[0])
    
        #Open dataset 
        data = json.load(open(f))
        print("Data opened")
    
        for elem in elements:
            
            # Select detrending window
            detrend_window = max(int((len(data[elem])-start_point)/2),50)
            bandwidth = bandwidths[elemnms[elem]].loc[trate]

            # Take residual and choose starting point
            states = np.array(even_number(data[elem][start_point-detrend_window:]))
            A = states - gaussian_filter1d(states, bandwidth)
            autocorr = calc_autocorrelation(A, step_size, detrend_window)
            variance = calc_variance(A, step_size, detrend_window)

            tau_autocorr = kendalltau(autocorr, np.arange(len(autocorr)))[0]
            tau_variance = kendalltau(variance, np.arange(len(autocorr)))[0]

            p_ac1, stat_ac1 = kendall_tau_test_Boers(autocorr, N, tau_autocorr)
            p_var1, stat_var1 = kendall_tau_test_Boers(variance, N, tau_variance)
            p_ac2, p_var2, stat_ac2, stat_var2 = kendall_tau_test_Bolt(A, N, tau_autocorr, tau_variance, step_size, detrend_window)

            fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,5), sharey=True)
            fig.suptitle(elemnms[elem]+" autocorrelation")

            ax1.set_title("Boers et.al. (2021)")
            ax1.hist(stat_ac1,50)
            ax1.axvline(tau_autocorr,c='r')
            ax1.set_ylabel("Frequency")
            ax1.set_xlabel(r"Kendall $\tau$ correlation")
            ax1.text(0.2, 0.95, "P value: {}".format(np.round(p_ac1, 2)),
                        horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            ax2.set_title("van der Bolt et.al. (2021)")
            ax2.hist(stat_ac2,50)
            ax2.axvline(tau_autocorr,c='r')
            ax2.set_xlabel(r"Kendall $\tau$ correlation")
            ax2.text(0.2, 0.95, "P value: {}".format(np.round(p_ac2, 2)),
                        horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            
            fig.savefig("../results/sensitivity/network_1.0_1.0/0137/"+subname+"/method_comparison_AC_{}_Trate{}_d{}.png".format(elemnms[elem], trate, strength))
            plt.clf()
            plt.close()

            fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,5), sharey=True)
            fig.suptitle(elemnms[elem]+" variance")

            ax1.set_title("Boers et.al. (2021)")
            ax1.hist(stat_var1,50)
            ax1.axvline(tau_variance,c='r')
            ax1.set_ylabel("Frequency")
            ax1.set_xlabel(r"Kendall $\tau$ correlation")
            ax1.text(0.2, 0.95, "P value: {}".format(np.round(p_var1, 2)),
                        horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            ax2.set_title("van der Bolt et.al. (2021)")
            ax2.hist(stat_var2,50)
            ax2.axvline(tau_variance,c='r')
            ax2.set_xlabel(r"Kendall $\tau$ correlation")
            ax2.text(0.2, 0.95, "P value: {}".format(np.round(p_var2, 2)),
                        horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

            fig.savefig("../results/sensitivity/network_1.0_1.0/0137/"+subname+"/method_comparison_var_{}_Trate{}_d{}.png".format(elemnms[elem], trate, strength))
            plt.clf()
            plt.close()

end = time.time()
print("Time elapsed until Finish: {}s".format(end - start))

