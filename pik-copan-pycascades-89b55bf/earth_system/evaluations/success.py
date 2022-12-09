import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)

alpha= 0.1
folders = np.sort(glob.glob("postprocessed/*"))
N = len(folders)

df = pd.read_csv(folders[1]+"/pvalue_ac.csv", index_col=[0,1])
trates = np.array(df.index.get_level_values("trate").unique())
strengths = np.array(df.index.get_level_values("strength").unique())
inv_trates = []
for i in range(len(trates)):
    inv_trates.append(r"1$^{\circ}C$/"+"{}yr".format(int(1./trates[i])))

elemnms = ["GIS", "THC", "WAIS", "AMAZ"]

indicators = ["ac", "var"]
titles = ["Autocorrelation", "Variance"]

for i in range(len(indicators)):

    fig, axes = plt.subplots(2, 2,figsize=(8,5),sharex=True)

    for elem in range(4):
        pvalue = np.empty((len(trates), len(strengths), N))
        
        for k in range(N):
            folder = folders[k]
            pv = pd.read_csv(folder+"/pvalue_"+indicators[i]+".csv", index_col=[0,1], usecols=["trate", "strength", elemnms[elem]])
            pv = pv.values.reshape((len(trates), len(strengths)))
            pvalue[:,:,k] = pv

        s_pv = pvalue <= alpha
        s_pv = np.sum(s_pv, axis=2)/N
        
        ax = axes.flat[elem]
        im = ax.imshow(s_pv.T, cmap=plt.cm.get_cmap('viridis', 10), vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(trates)))
        ax.set_yticks(np.arange(len(strengths)))
        ax.set_xticklabels(inv_trates,rotation='vertical')
        ax.set_yticklabels(strengths)
        ax.invert_yaxis()
        ax.set_title(elemnms[elem])

        # Create white grid
        ax.grid(False)
        ax.set_xticks(np.arange(len(trates)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(strengths)+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
    
    fig.suptitle(titles[i])
    fig.subplots_adjust(bottom=0.35)
    cbar_ax = fig.add_axes([0.25, 0.1, 0.49,0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
    cbar.ax.set_xlabel("Success rate")
    fig.savefig("plots/success_"+indicators[i]+".png")
    plt.clf()
    plt.close()
