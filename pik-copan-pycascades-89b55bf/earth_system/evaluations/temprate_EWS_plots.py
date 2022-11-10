import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
import pandas as pd
import os

init_folder = "0111"

try:
    os.stat("plots/{}".format(init_folder))
except:
    os.mkdir("plots/{}".format(init_folder))

# read kendall tau value data
gis_tip = pd.read_csv("data_reshaped/tau_gis_tip_"+init_folder+".csv", index_col = [0, 1])
thc_tip = pd.read_csv("data_reshaped/tau_thc_tip_"+init_folder+".csv", index_col = [0, 1])
wais_tip = pd.read_csv("data_reshaped/tau_wais_tip_"+init_folder+".csv", index_col = [0, 1])
amaz_tip = pd.read_csv("data_reshaped/tau_amaz_tip_"+init_folder+".csv", index_col = [0, 1])
gis_notip = pd.read_csv("data_reshaped/tau_gis_notip_"+init_folder+".csv", index_col = [0, 1])
thc_notip = pd.read_csv("data_reshaped/tau_thc_notip_"+init_folder+".csv", index_col = [0, 1])
wais_notip = pd.read_csv("data_reshaped/tau_wais_notip_"+init_folder+".csv", index_col = [0, 1])
amaz_notip = pd.read_csv("data_reshaped/tau_amaz_notip_"+init_folder+".csv", index_col = [0, 1])

# collect data for all elements in one list
dfs_tip = [gis_tip, thc_tip, wais_tip, amaz_tip]
dfs_notip = [gis_notip, thc_notip, wais_notip, amaz_notip]

title = ["GIS", "THC", "WAIS", "AMAZ"]

# plot for two coupling strengths

strength1 = 0.0
strength2 = 0.25
trates = np.array(gis_tip.loc[(slice(None),strength1),"Tau_AC_mean"].index.get_level_values("trate"))
inv_trates = []
for i in range(len(trates)):
    inv_trates.append(r"1$^{\circ}C$/"+"{}yr".format(int(1./trates[i])))

for elem in range(4):
    df_tip = dfs_tip[elem]
    df_notip = dfs_notip[elem]

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9,5), sharey=True)
    fig.suptitle("Autocorrelation "+title[elem])

    ax1.scatter(trates, df_tip.loc[(slice(None),strength1),"Tau_AC_mean"].values, c="r")
    ax1.scatter(trates, df_notip.loc[(slice(None),strength1),"Tau_AC_mean"].values, c="k")
    ax1.set_xscale('log')
    ax1.set_xticks(trates)
    ax1.set_xticklabels(inv_trates, rotation='vertical')
    ax1.set_ylabel(r"Kendall's $\tau$ correlation")
    ax1.set_ylim(-1,1)
    ax1.set_title("Coupling strength: {}".format(strength1))

    
    ax2.scatter(trates, df_tip.loc[(slice(None),strength2),"Tau_AC_mean"].values, c="r", label="Tipping")
    ax2.scatter(trates, df_notip.loc[(slice(None),strength2),"Tau_AC_mean"].values, c="k", label="No tipping")
    ax2.set_xscale('log')
    ax2.set_xticks(trates)
    ax2.set_xticklabels(inv_trates, rotation='vertical')
    ax2.set_title("Coupling strength: {}".format(strength2))
    ax2.legend(loc="lower left", scatterpoints=1)
    fig.tight_layout()
    fig.savefig("plots/"+init_folder+"/EWS_AC_{}".format(title[elem]))
    plt.clf()

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9,5), sharey=True)
    fig.suptitle("Variance "+title[elem])

    ax1.scatter(trates, df_tip.loc[(slice(None),strength1),"Tau_var_mean"].values, c="r")
    ax1.scatter(trates, df_notip.loc[(slice(None),strength1),"Tau_var_mean"].values, c="k")
    ax1.set_xscale('log')
    ax1.set_xticks(trates)
    ax1.set_xticklabels(inv_trates, rotation='vertical')
    ax1.set_ylabel(r"Kendall's $\tau$ correlation")
    ax1.set_ylim(-1,1)
    ax1.set_title("Coupling strength: {}".format(strength1))


    ax2.scatter(trates, df_tip.loc[(slice(None),strength2),"Tau_var_mean"].values, c="r", label="Tipping")
    ax2.scatter(trates, df_notip.loc[(slice(None),strength2),"Tau_var_mean"].values, c="k", label="No tipping")
    ax2.set_xscale('log')
    ax2.set_xticks(trates)
    ax2.set_xticklabels(inv_trates, rotation='vertical')
    ax2.legend(loc="lower left", scatterpoints=1)
    ax2.set_title("Coupling strength: {}".format(strength2))
    
    fig.tight_layout()
    fig.savefig("plots/"+init_folder+"/EWS_var_{}.png".format(title[elem]))

# plotting scatter plot with colormap for all coupling strengths
def add_colorbar(mappable,label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax,label=label)
    plt.sca(last_axes)
    return cbar

# extract arrays of temperature rates and couplings strengths
trates_full = np.array(gis_tip.index.get_level_values("trate"))
strengths_full = np.array(gis_tip.index.get_level_values("strength"))

for elem in range(4):
    df = dfs_tip[elem]

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9,5), sharey=True)
    fig.suptitle(title[elem])

    im1 = ax1.scatter(trates_full, strengths_full, c=df["Tau_AC_mean"], cmap='viridis', vmin=-1, vmax=1, s = plt.rcParams['lines.markersize'] ** 2 + df["Tau_AC_std"])
    ax1.set_xscale('log')
    ax1.set_xticks(trates)
    ax1.set_xticklabels(inv_trates, rotation='vertical')
    ax1.set_title("Autocorrelation")
    ax1.set_ylabel("Coupling strength")
    #ax1.set_xlabel(r"Temperature rate $^{\circ}$C/yr", loc='right')

    im2 = ax2.scatter(trates_full, strengths_full, c=df["Tau_var_mean"], cmap='viridis', vmin=-1, vmax=1, s = plt.rcParams['lines.markersize'] ** 2 + df["Tau_var_std"])
    ax2.set_xscale('log')
    ax2.set_xticks(trates)
    ax2.set_xticklabels(inv_trates, rotation='vertical')
    add_colorbar(im2, r"Kendall's $\tau$ correlation")
    ax2.set_title("Variance")
    
    fig.tight_layout()
    fig.savefig("plots/"+init_folder+"/EWS_temprate_{}.png".format(title[elem]))


"""
    pivoted = gis["Tau_AC_mean"].reset_index().pivot_table(index='strength',
                                                       columns='trate', 
                                                       dropna=False)
    arr = np.array(pivoted.values[:, :])
    print(arr)

    X, Y = np.meshgrid(trates, strengths)
    plt.contourf(X, Y, arr)

    plt.show()
"""
