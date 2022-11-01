import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)

def CO2_to_GMT_IPCC90(CO2):

    CO2_pi = 280
    alpha = 5.35
    f = 2.02

    GMT = alpha*np.log(CO2/CO2_pi)/f

    return GMT

def CO2_to_GMT_Shi92(CO2):
    
    CO2_pi = 280
    alpha = 4.841
    beta = 0.0906
    f = 2.02 # Conversion factor from radiative forcing to temperature increase

    GMT = (alpha*np.log( CO2 / CO2_pi ) + beta*(np.sqrt(CO2)-np.sqrt(CO2_pi)))/f

    return GMT

def g(C):
    
    return np.log(1 + 1.2*C + 0.005*C**2 + 1.4*10**(-6)*C**3)

def CO2_to_GMT_WMO99(CO2):
    
    CO2_pi = 280
    alpha = 3.35
    f = 2.02 # Conversion factor from radiative forcing to temperature increase
    
    GMT = alpha*(g(CO2) - g(CO2_pi))/f

    return GMT

path = "/home/astridbg/Documents/traineeship/gmd-13-3571-2020-supplement/"
fname = path+"SUPPLEMENT_DataTables_Meinshausen_6May2020.xlsx"


historical = pd.read_excel(fname, sheet_name='T2 - History Year 1750 to 2014',
                     skiprows=11, index_col=0, usecols=[0,1])

ssp1_26 = pd.read_excel(fname, sheet_name='T4 -  SSP1-2.6 ',
                     skiprows=11, index_col=0, usecols=[0,1])
hist_ssp1_26 = pd.concat([historical, ssp1_26])
hist_ssp1_26 = hist_ssp1_26.rename({'Unnamed: 1': "CO2 ppm World"},axis='columns')
hist_ssp1_26['GMT'] = CO2_to_GMT_Shi92(hist_ssp1_26['CO2 ppm World'])
hist_ssp1_26.to_csv(path+'hist_ssp1-26.csv')

ssp2_45 = pd.read_excel(fname, sheet_name='T5 - SSP2-4.5 ',
                     skiprows=11, index_col=0, usecols=[0,1])
hist_ssp2_45 = pd.concat([historical, ssp2_45])
hist_ssp2_45 = hist_ssp2_45.rename({'Unnamed: 1': "CO2 ppm World"},axis='columns')
hist_ssp2_45['GMT'] = CO2_to_GMT_Shi92(hist_ssp2_45['CO2 ppm World'])
hist_ssp2_45.to_csv(path+'hist_ssp2-45.csv')
print(hist_ssp2_45)

# Visualize temperature trajectory

data = pd.read_csv(path+'hist_ssp2-45.csv', index_col=0)

fig, ax1 = plt.subplots(1,1)
ax2 = ax1.twinx()
ax1.plot(data.index, data['CO2 ppm World'], c='black')
ax1.set_ylabel('CO2 ppm World')
ax1.yaxis.label.set_color('black')
ax2.plot(data.index, data['GMT'], label="Shi (1992)",c='orange')
ax2.set_ylabel(r'$\Delta$GMT')
ax2.yaxis.label.set_color('orange')
ax1.grid(False)
ax2.grid(True)
plt.show()

