import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
import re
import glob
import pandas as pd

# read files in one folder to get temprates and coupling strengths
trates = []
strengths = []

folder = "../results/feedbacks/network_1.0_1.0/0071"
subfolders = np.array(np.sort(glob.glob(folder + "/Tau_*.txt")))
for kk in subfolders:
    trate = re.split("_",re.split("Trate",kk)[-1])[0]
    strength = re.split("_",re.split("d",kk)[-1])[0]
    trates.append(trate)
    strengths.append(strength)

# create dataframes for each tipping element with temprates and coupling strength as indices

tuples = list(zip(*[trates,strengths]))
index = pd.MultiIndex.from_tuples(list(zip(*[trates,strengths])), names=["trate", "strength"])
gis = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
thc = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
wais = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
amaz = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
dfs = [gis, thc, wais, amaz]


# read all files
networks = np.array(np.sort(glob.glob("../results/feedbacks/*")))

for trate in set(trates):
    for strength in set(strengths):
        tau_AC = []
        tau_var = []
        for network in [networks[8]]:
            output = []
            final = []
            folders = np.array(np.sort(glob.glob(network + "/0*"))) #do not collect special folders which start with a "-"
            for folder in [folders[6]]:
                fnames = np.array(np.sort(glob.glob(folder + "/Tau*")))
                fname = folder + "/Tau_Tend3.5_Trate{}_d{}_n0.005.txt".format(trate, strength)
                # requires only one noise level and end temperature to be used
                file = np.loadtxt(fname)
                for elem in range(4):
                    if file[0][elem] == 0:
                        file[1][elem] = np.nan
                        file[2][elem] = np.nan
                tau_AC.append(file[1])
                tau_var.append(file[2])
        tau_AC = np.array(tau_AC)
        tau_var = np.array(tau_var)
        for elem in range(4):
            df = dfs[elem]
            df.loc[(trate, strength), "Tau_AC_mean"] = np.nanmean(tau_AC.T[elem])
            df.loc[(trate, strength), "Tau_AC_std"] = np.nanstd(tau_AC.T[elem])
            df.loc[(trate, strength), "Tau_var_mean"] = np.nanmean(tau_var.T[elem])
            df.loc[(trate, strength), "Tau_var_std"] = np.nanstd(tau_var.T[elem])

gis.to_csv("data_reshaped/tau_gis.csv")
thc.to_csv("data_reshaped/tau_thc.csv")
wais.to_csv("data_reshaped/tau_wais.csv")
amaz.to_csv("data_reshaped/tau_amaz.csv")
