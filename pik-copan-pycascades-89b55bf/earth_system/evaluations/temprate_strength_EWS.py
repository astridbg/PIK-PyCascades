import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
import re
import glob
import pandas as pd

init_folder = "0111"

# read files in one folder to get temprates and coupling strengths
trates = []
strengths = []

folder = "../results/feedbacks/network_1.0_1.0/"+init_folder
subfolders = np.array(np.sort(glob.glob(folder + "/Tau_*.txt")))
for kk in subfolders:
    trate = re.split("_",re.split("Trate",kk)[-1])[0]
    strength = re.split("_",re.split("d",kk)[-1])[0]
    trates.append(trate)
    strengths.append(strength)

# create dataframes for each tipping element with temprates and coupling strength as indices

tuples = list(zip(*[trates,strengths]))
index = pd.MultiIndex.from_tuples(list(zip(*[trates,strengths])), names=["trate", "strength"])
gis_tip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
thc_tip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
wais_tip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
amaz_tip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
gis_notip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
thc_notip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
wais_notip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])
amaz_notip = pd.DataFrame(index=index, columns=["Tau_AC_mean", "Tau_AC_std", "Tau_var_mean", "Tau_var_std"])

dfs_tip = [gis_tip, thc_tip, wais_tip, amaz_tip]
dfs_notip = [gis_notip, thc_notip, wais_notip, amaz_notip]

# read all files
networks = np.array(np.sort(glob.glob("../results/feedbacks/*")))

for trate in set(trates):
    for strength in set(strengths):
        tau_AC_tip = []
        tau_var_tip = []
        tau_AC_notip = []
        tau_var_notip = []
        for network in [networks[8]]:
            output = []
            final = []
            folders = np.array(np.sort(glob.glob(network + "/0*"))) #do not collect special folders which start with a "-"
            for folder in [folders[-1]]:
                fnames = np.array(np.sort(glob.glob(folder + "/Tau*")))
                fname = folder + "/Tau_Tend4.0_Trate{}_d{}_n0.005.txt".format(trate, strength)
                # requires only one noise level and end temperature to be used
                data = np.loadtxt(fname)
                tip = data[0]
                tip_tau_AC = list(data[1])
                notip_tau_AC = list(data[1])
                tip_tau_var = list(data[2])
                notip_tau_var = list(data[2])
                for elem in range(4):
                    if tip[elem] == 0:
                        tip_tau_AC[elem] = np.nan
                        tip_tau_var[elem] = np.nan
                    else:
                        notip_tau_AC[elem] = np.nan
                        notip_tau_var[elem] = np.nan
                tau_AC_tip.append(tip_tau_AC)
                tau_var_tip.append(tip_tau_var)
                tau_AC_notip.append(notip_tau_AC)
                tau_var_notip.append(notip_tau_var)
        tau_AC_tip = np.array(tau_AC_tip)
        tau_var_tip = np.array(tau_var_tip)
        tau_AC_notip = np.array(tau_AC_notip)
        tau_var_notip = np.array(tau_var_notip)        
        for elem in range(4):
            df_tip = dfs_tip[elem]
            df_notip = dfs_notip[elem]
            df_tip.loc[(trate, strength), "Tau_AC_mean"] = np.nanmean(tau_AC_tip.T[elem])
            df_tip.loc[(trate, strength), "Tau_AC_std"] = np.nanstd(tau_AC_tip.T[elem])
            df_tip.loc[(trate, strength), "Tau_var_mean"] = np.nanmean(tau_var_tip.T[elem])
            df_tip.loc[(trate, strength), "Tau_var_std"] = np.nanstd(tau_var_tip.T[elem])
            df_notip.loc[(trate, strength), "Tau_AC_mean"] = np.nanmean(tau_AC_notip.T[elem])
            df_notip.loc[(trate, strength), "Tau_AC_std"] = np.nanstd(tau_AC_notip.T[elem])
            df_notip.loc[(trate, strength), "Tau_var_mean"] = np.nanmean(tau_var_notip.T[elem])
            df_notip.loc[(trate, strength), "Tau_var_std"] = np.nanstd(tau_var_notip.T[elem])

gis_tip.to_csv("data_reshaped/tau_gis_tip_"+init_folder+".csv")
thc_tip.to_csv("data_reshaped/tau_thc_tip_"+init_folder+".csv")
wais_tip.to_csv("data_reshaped/tau_wais_tip_"+init_folder+".csv")
amaz_tip.to_csv("data_reshaped/tau_amaz_tip_"+init_folder+".csv")
gis_notip.to_csv("data_reshaped/tau_gis_notip_"+init_folder+".csv")
thc_notip.to_csv("data_reshaped/tau_thc_notip_"+init_folder+".csv")
wais_notip.to_csv("data_reshaped/tau_wais_notip_"+init_folder+".csv")
amaz_notip.to_csv("data_reshaped/tau_amaz_notip_"+init_folder+".csv")
