from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.25)
from pyDOE import * #function name >>> lhs



#Tipping limits, see Schellnhuber, et al., 2016:
limits_gis  = [0.8, 3.0]  #0.8-3.0 (central: 1.5)     old values: [0.8, 3.2] here in brackets: values from new review of D.A. McKay (publicly available yet since January 2022)
limits_thc  = [1.4, 8.0]  #1.4-8.0 (central: 4.0)     old values: [3.5, 6.0]
limits_wais = [1.0, 3.0]  #1.0-3.0 (central: 1.5)      old values: [0.8, 5.5]
limits_amaz = [2.0, 6.0]  #2.0-6.0 (central: 3.5)      old values: [3.5, 4.5]
limits_nino = [3.0, 6.0]  #3.0-6.0 (central: uncleear) old values: [3.5, 7.0]


###################################################
# Probability fraction (PF) increases with tipping from A-->B (see gray boxes in Kriegler et al., 2009)
#TO GIS
pf_wais_to_gis = [0.1, 0.2]  # from PF = [1., 2.]
pf_thc_to_gis = [0.1, 1.]    # from PF = [0.1, 1.]
# TO THC
pf_gis_to_thc = [0.1, 1.]    # from PF = [1., 10.]
pf_nino_to_thc = [0.1, 0.2]  # changed from [0.5, 2.0] due to Lenton & Williams, 2013. It seems that only a stabilizing effect makes sense
                             # from PF = [0.5, 1.]
# unclear link
pf_wais_to_thc = [0.1, 0.3]  # from PF = [0.3, 3.]
# TO WAIS
pf_nino_to_wais = [0.1, 0.5] # from PF = [1., 5.]
pf_thc_to_wais = [0.1, 0.15] # from PF = [1., 1.5]
pf_gis_to_wais = [0.1, 1.0]  # from PF = [1., 10.]
# TO NINO
pf_thc_to_nino = [0.1, 0.2]     # from PF = [1., 2.]
# unclear link
pf_amaz_to_nino = [0.1, 0.15]   # from PF = [0.8, 1.5]
# TO AMAZ
pf_nino_to_amaz = [0.1, 1.]     # from PF = [1., 10.]
# unclear link
pf_thc_to_amaz = [0.1, 0.4]  # from PF = [1.0, 4.0], changed from +-Link to + Link due to C. Ciemers results

###################################################
#Time scale of tipping for the tipping elements (taken from the literature review of DA. McKay)
tau_gis  = [1000, 15000]         #1000-15000(central: 10.000)      old values: [1000, 15000] 
tau_thc  = [15, 300]             #15-120 (central: 50)             old values: [15, 300]     
tau_wais = [500, 13000]          #500-13000 (central: 2000)        old values: [1000, 13000] 
tau_nino = [25, 200]             #unclear (around 100)             old values: [25, 200]     
tau_amaz = [50, 200]             #50-200 (central: 100)            old values: [50, 200]     


#Social parameters
cares = np.array([0,1,8])
tau_solid = 37
taus = [tau_solid]#np.array([tau_solid/2.2, tau_solid, tau_solid*2.2])

"""
Latin hypercube sampling
Note: These points need a rescaling according to the uncertainty ranges
This can be done by: x_new = lower_lim + (upper_lim - lower_lim) * u[0;1), where u[0;1) = Latin-HC
"""
points = np.array(lhs(22, samples=100)) #give dimensions and sample size, here shown for a Latin hypercube; (unfortunately not space filling and not orthogonal)

#rescaling function from latin hypercube
def latin_function(limits, rand):
    resc_rand = limits[0] + (limits[1] - limits[0]) * rand
    return resc_rand


#MAIN
array_limits = []
sh_file = []
for i in range(0, len(points)):
    print(i)

    #TIPPING RANGES
    rand_gis = latin_function(limits_gis, points[i][0])
    rand_thc = latin_function(limits_thc, points[i][1])
    rand_wais = latin_function(limits_wais, points[i][2])
    rand_amaz = latin_function(limits_amaz, points[i][3])
    rand_nino = latin_function(limits_nino, points[i][4])
        

    # PROBABILITY FRACTIONS
    rand_wais_to_gis = latin_function(pf_wais_to_gis, points[i][5])
    rand_thc_to_gis = latin_function(pf_thc_to_gis, points[i][6])
    rand_gis_to_thc = latin_function(pf_gis_to_thc, points[i][7])
    rand_nino_to_thc = latin_function(pf_nino_to_thc, points[i][8])
    rand_wais_to_thc = latin_function(pf_wais_to_thc, points[i][9])
    rand_nino_to_wais = latin_function(pf_nino_to_wais, points[i][10])
    rand_thc_to_wais = latin_function(pf_thc_to_wais, points[i][11])
    rand_gis_to_wais = latin_function(pf_gis_to_wais, points[i][12])
    rand_thc_to_nino = latin_function(pf_thc_to_nino, points[i][13])
    rand_amaz_to_nino = latin_function(pf_amaz_to_nino, points[i][14])
    rand_nino_to_amaz = latin_function(pf_nino_to_amaz, points[i][15])
    rand_thc_to_amaz = latin_function(pf_thc_to_amaz, points[i][16])


    #FEEDBACKS
    rand_tau_gis = latin_function(tau_gis, points[i][17])
    rand_tau_thc = latin_function(tau_thc, points[i][18])
    rand_tau_wais = latin_function(tau_wais, points[i][19])
    rand_tau_amaz = latin_function(tau_amaz, points[i][20])
    rand_tau_nino = latin_function(tau_nino, points[i][21])

    for care in cares:
        for tau in taus:
            array_limits.append([rand_gis, rand_thc, rand_wais, rand_amaz, rand_nino,
                                 rand_wais_to_gis, rand_thc_to_gis, rand_gis_to_thc, rand_nino_to_thc,
                                 rand_wais_to_thc, rand_nino_to_wais, rand_thc_to_wais, rand_gis_to_wais,
                                 rand_thc_to_nino, rand_amaz_to_nino, rand_nino_to_amaz, rand_thc_to_amaz,
                                 rand_tau_gis, rand_tau_thc, rand_tau_wais, rand_tau_nino, rand_tau_amaz, 
                                 care, tau])


            sh_file.append(["python /p/projects/dominoes/laeoc/pycascades/earth_sys/MAIN_cluster_earth_system_complete_no_enso.py $SLURM_NTASKS {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                                     rand_gis, rand_thc, rand_wais, rand_amaz, rand_nino,
                                     rand_wais_to_gis, rand_thc_to_gis, rand_gis_to_thc, rand_nino_to_thc,
                                     rand_wais_to_thc, rand_nino_to_wais, rand_thc_to_wais, rand_gis_to_wais,
                                     rand_thc_to_nino, rand_amaz_to_nino, rand_nino_to_amaz, rand_thc_to_amaz,
                                     rand_tau_gis, rand_tau_thc, rand_tau_wais, rand_tau_nino, rand_tau_amaz,
                                     care, tau,
                                     str(i).zfill(4))]) #zfill necessary to construct enough folders for monte carlo runs


array_limits = np.array(array_limits)
np.savetxt("latin_prob.txt", array_limits, delimiter=" ")


#Create .sh file to run on the cluster
sh_file = np.array(sh_file)
np.savetxt("latin_sh_file.txt", sh_file, delimiter=" ", fmt="%s")




#tipping ranges and plots
gis = array_limits.T[0]
thc = array_limits.T[1]
wais = array_limits.T[2]
amaz = array_limits.T[3]
nino = array_limits.T[4]


plt.grid(True)
plt.hist(gis, 24, facecolor='c', alpha=0.5, label="GIS")
plt.hist(thc, 25, facecolor='b', alpha=0.5, label="THC")
plt.hist(wais, 47, facecolor='k', alpha=0.5, label="WAIS")
plt.hist(amaz, 10, facecolor='g', alpha=0.5, label="AMAZ")
plt.hist(nino, 35, facecolor='r', alpha=0.5, label="NINO")
plt.legend(loc='best')
plt.xlabel("Tipping range [°C]")
plt.ylabel("N [#]")
plt.tight_layout()
plt.savefig("latin_prob_TR.png")
plt.savefig("latin_prob_TR.pdf")
#plt.show()
plt.clf()
plt.close()


#coupling strength
wais_to_gis = array_limits.T[5]
thc_to_gis = array_limits.T[6]
gis_to_thc = array_limits.T[7]

nino_to_thc = array_limits.T[8]
wais_to_thc = array_limits.T[9]
nino_to_wais = array_limits.T[10]

thc_to_wais = array_limits.T[11]
gis_to_wais = array_limits.T[12]
thc_to_nino = array_limits.T[13]

amaz_to_nino = array_limits.T[14]
nino_to_amaz = array_limits.T[15]
thc_to_amaz_pos = array_limits.T[16]


plt.grid(True)
plt.hist(wais_to_gis, 10, facecolor='c', alpha=0.5, label="wais_to_gis")
plt.hist(thc_to_gis, 100, facecolor='b', alpha=0.5, label="thc_to_gis")
plt.hist(gis_to_thc, 100, facecolor='k', alpha=0.5, label="gis_to_thc")
plt.hist(nino_to_thc, 10, facecolor='g', alpha=0.5, label="nino_to_thc")
plt.hist(wais_to_thc, 30, facecolor='r', alpha=0.5, label="wais_to_thc")
plt.hist(nino_to_wais, 50, facecolor='#FF7000', alpha=0.5, label="nino_to_wais")
plt.hist(thc_to_wais, 5, facecolor='#2D9575', alpha=0.5, label="thc_to_wais")
plt.hist(gis_to_wais, 100, facecolor='#8E58C3', alpha=0.5, label="gis_to_wais")
plt.hist(thc_to_nino, 10, facecolor='#FF00F7', alpha=0.5, label="thc_to_nino")
plt.hist(amaz_to_nino, 5, facecolor='#DAF7A6', alpha=0.5, label="amaz_to_nino")
plt.hist(nino_to_amaz, 100, facecolor='#FFC300', alpha=0.5, label="nino_to_amaz")
plt.hist(thc_to_amaz_pos, 40, facecolor='#FF5733', alpha=0.5, label="thc_to_amaz")
plt.legend(loc='best')
plt.xlabel("Probability fraction [a.u.]")
plt.ylabel("N [#]")
plt.tight_layout()
plt.savefig("latin_prob_PF.png")
plt.savefig("latin_prob_PF.pdf")
#plt.show()
plt.clf()
plt.close()


#feedbacks
rand_tau_gis = array_limits.T[17]
rand_tau_thc = array_limits.T[18]
rand_tau_wais = array_limits.T[19]
rand_tau_nino = array_limits.T[20]
rand_tau_amaz = array_limits.T[21]


plt.grid(True)
plt.hist(rand_tau_gis,  140, facecolor='c', alpha=0.5, label="tipping_time_gis")
plt.hist(rand_tau_thc,  16, facecolor='b', alpha=0.5, label="tipping_time_thc")
plt.hist(rand_tau_wais, 120, facecolor='k', alpha=0.5, label="tipping_time_wais")
plt.hist(rand_tau_amaz, 14, facecolor='g', alpha=0.5, label="tipping_time_amaz")
plt.hist(rand_tau_nino, 14, facecolor='r', alpha=0.5, label="tipping_time_nino")
plt.legend(loc='best')
plt.xlabel("Tipping time [yr]")
plt.ylabel("N [#]")
plt.tight_layout()
plt.savefig("latin_prob_tau.png")
plt.savefig("latin_prob_tau.pdf")
#plt.show()
plt.clf()
plt.close()

print("Finish")