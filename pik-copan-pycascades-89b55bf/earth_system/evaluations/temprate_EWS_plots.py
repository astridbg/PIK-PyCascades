import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
import pandas as pd

gis = pd.read_csv("data_reshaped/tau_gis.csv", index_col = [0, 1])
thc = pd.read_csv("data_reshaped/tau_thc.csv", index_col = [0, 1])
wais = pd.read_csv("data_reshaped/tau_wais.csv", index_col = [0, 1])
amaz = pd.read_csv("data_reshaped/tau_amaz.csv", index_col = [0, 1])
dfs = [gis, thc, wais, amaz]
print(gis)

trates = np.array(list(set(list(gis.index.get_level_values("trate")))))
strengths = np.array(list(set(list(gis.index.get_level_values("strength")))))

print(trates)
print(strengths)

pivoted = gis["Tau_AC_mean"].reset_index().pivot_table(index='strength',
                                       columns='trate', dropna=False)
arr = np.array(pivoted.values[:, :])
print(arr)

X, Y = np.meshgrid(trates, strengths)
plt.contourf(X, Y, arr)

plt.show()

