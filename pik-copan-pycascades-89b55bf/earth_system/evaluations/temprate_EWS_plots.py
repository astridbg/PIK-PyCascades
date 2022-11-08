import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
import pandas as pd


# read kendall tau value data
gis = pd.read_csv("data_reshaped/tau_gis.csv", index_col = [0, 1])
thc = pd.read_csv("data_reshaped/tau_thc.csv", index_col = [0, 1])
wais = pd.read_csv("data_reshaped/tau_wais.csv", index_col = [0, 1])
amaz = pd.read_csv("data_reshaped/tau_amaz.csv", index_col = [0, 1])

# collect data for all elements in one list
dfs = [gis, thc, wais, amaz]

# extract arrays of temperature rates and couplings strengths
#trates = np.array(list(set(list(gis.index.get_level_values("trate")))))
#strengths = np.array(list(set(list(gis.index.get_level_values("strength")))))
trates = np.array(gis.index.get_level_values("trate"))
strengths = np.array(gis.index.get_level_values("strength"))

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

title = ["GIS", "THC", "WAIS", "AMAZ"]

# plotting scatter plot with colormap for all coupling strengths
for elem in range(1):
    df = dfs[elem]

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9,5), sharey=True)
    fig.suptitle(title[elem])

    im1 = ax1.scatter(trates, strengths, c=df["Tau_AC_mean"], cmap='viridis', vmin=-1, vmax=1, s = plt.rcParams['lines.markersize'] ** 2 + df["Tau_AC_std"])
    ax1.set_xscale('log')
    ax1.set_title("Autocorrelation")
    ax1.set_ylabel("Coupling strength")
    #ax1.set_xlabel(r"Temperature rate $^{\circ}$C/yr", loc='right')

    im2 = ax2.scatter(trates, strengths, c=df["Tau_var_mean"], cmap='viridis', vmin=-1, vmax=1, s = plt.rcParams['lines.markersize'] ** 2 + df["Tau_var_std"])
    ax2.set_xscale('log')
    add_colorbar(im2, r"Kendall's $\tau$ correlation")
    fig.supxlabel(r"Temperature rate $^{\circ}$C/yr", fontsize=12)
    ax2.set_title("Variance")

    plt.show()

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
