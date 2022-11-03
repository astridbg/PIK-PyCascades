import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
import re
import glob


# read all files
networks = np.array(np.sort(glob.glob("../results/feedbacks/*")))


for network in networks:
    print(network)
    output = []
    final = []
    net_splitter = re.split("network", network)[-1] #used for the saving structure
    folders = np.array(np.sort(glob.glob(network + "/0*"))) #do not collect special folders which start with a "-"
    for folder in folders:
        subfolders = np.array(np.sort(glob.glob(folder + "/Tau_*.txt")))
        for kk in subfolders:
            Trate = re.split("_",re.split("Trate", kk)[-1])[0]
            print(Trate)
            file = np.loadtxt(kk)
            print(file)
            #final.append([file[6], file[7], file[8], file[9], 1])
    final = np.array(final)


    #output.append([
    #    np.sum(final.T[0]), np.sum(final.T[1]), np.sum(final.T[2]), np.sum(final.T[3]), np.sum(final.T[4])
    #    ])
    #output = np.array(output)
    #print(output)

print("Finish")
