# Add modules directory to path
import os
import sys
import re

sys.path.append('')

# global imports
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import time
import glob
from PyPDF2 import PdfFileMerger
from netCDF4 import Dataset


# private imports from sys.path
from core.evolve import evolve

#private imports for earth system
from earth_sys.timing_no_enso import timing
from earth_sys.functions_earth_system_no_enso import global_functions
from earth_sys.earth_no_enso import earth_system

#for cluster computations
os.chdir("/p/projects/dominoes/nicowun/conceptual_tipping/uniform_distribution/overshoot_study")

#measure time
#start = time.time()
#############################GLOBAL SWITCHES#########################################
time_scale = True            # time scale of tipping is incorporated
plus_minus_include = True    # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
######################################################################
duration = 50000 #actual real simulation years



#Names to create the respective directories
namefile = "no"
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
coupling_strength = np.linspace(0.0, 1.0, 11, endpoint=True)
#temperature input (forced with generated overshoot inputs)
GMT_files = np.sort(glob.glob("temp_input/timeseries_final/*.txt"))


"""
#make ensemble a bit smaller for the moment
GMT_files_1 = np.sort(glob.glob("temp_input/timeseries_final/*100.txt"))
GMT_files_2 = np.sort(glob.glob("temp_input/timeseries_final/*500.txt"))
GMT_files_3 = np.sort(glob.glob("temp_input/timeseries_final/*1000.txt"))
GMT_files = np.concatenate((np.concatenate((GMT_files_1, GMT_files_2)), GMT_files_3))
"""


########################Declaration of variables from passed values#######################
#Must sort out first and second value since this is the actual file and the number of nodes used
sys_var = np.array(sys.argv[2:], dtype=str) #low sample -3, intermediate sample: -2, high sample: -1


#TEST SYTEM
#print("USING TEST SYSTEM")
#sys_var = np.array([1.6, 4.75, 3.25, 4.0, 4.0, 0.2, 1.0, 1.0, 0.2, 0.3, 0.5, 0.15, 1.0, 0.2, 0.15, 1.0, 0.4, 4000, 150, 4000, 50, 100, 2400])
#####################################################################


#Tipping ranges from distribution
limits_gis, limits_thc, limits_wais, limits_amaz, limits_nino = float(sys_var[0]), float(sys_var[1]), float(sys_var[2]), float(sys_var[3]), float(sys_var[4])

#Probability fractions
# TO GIS
pf_wais_to_gis, pf_thc_to_gis = float(sys_var[5]), float(sys_var[6])
# TO THC
pf_gis_to_thc, pf_nino_to_thc, pf_wais_to_thc = float(sys_var[7]), float(sys_var[8]), float(sys_var[9])
# TO WAIS
pf_nino_to_wais, pf_thc_to_wais, pf_gis_to_wais = float(sys_var[10]), float(sys_var[11]), float(sys_var[12])
# TO NINO
pf_thc_to_nino, pf_amaz_to_nino = float(sys_var[13]), float(sys_var[14])
# TO AMAZ
pf_nino_to_amaz, pf_thc_to_amaz = float(sys_var[15]), float(sys_var[16])

#tipping time scales
tau_gis, tau_thc, tau_wais, tau_nino, tau_amaz = float(sys_var[17]), float(sys_var[18]), float(sys_var[19]), float(sys_var[20]), float(sys_var[21])


#Time scale
"""
All tipping times are computed ion comparison to the Amazon rainforest tipping time. As this is variable now, this affects the results to a (very) level
"""
if time_scale == True:
    print("compute calibration timescale")
    #function call for absolute timing and time conversion
    time_props = timing(tau_gis, tau_thc, tau_wais, tau_amaz, tau_nino)
    gis_time, thc_time, wais_time, nino_time, amaz_time = time_props.timescales()
    conv_fac_gis = time_props.conversion()
else:
    #no time scales included
    gis_time = thc_time = wais_time = nino_time = amaz_time = 1.0
    conv_fac_gis = 1.0

#include uncertain "+-" links:
if plus_minus_include == True:
    plus_minus_links = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=3)))

    #in the NO_ENSO case (i.e., the second link must be 0.0)
    plus_minus_data = []
    for pm in plus_minus_links:
        if pm[1] == 0.0:
            plus_minus_data.append(pm)
    plus_minus_links = np.array(plus_minus_data)

else:
    plus_minus_links = [np.array([1., 1., 1.])]


#directories for the Monte Carlo simulation
mc_dir = int(sys_var[-1])

#plus_minus_links = [plus_minus_links[0]]
#plus_minus_links = [plus_minus_links[1]]
#plus_minus_links = [plus_minus_links[2]]
#plus_minus_links = [plus_minus_links[3]]
#plus_minus_links = [plus_minus_links[4]]
#plus_minus_links = [plus_minus_links[5]]
#plus_minus_links = [plus_minus_links[6]]
#plus_minus_links = [plus_minus_links[7]]
#plus_minus_links = [plus_minus_links[8]]



#plus_minus_links = plus_minus_links[6:8]
#print(plus_minus_links)


################################# MAIN #################################
#Create Earth System
earth_system = earth_system(gis_time, thc_time, wais_time, nino_time, amaz_time,
                            limits_gis, limits_thc, limits_wais, limits_nino, limits_amaz,
                            pf_wais_to_gis, pf_thc_to_gis, pf_gis_to_thc, pf_nino_to_thc,
                            pf_wais_to_thc, pf_gis_to_wais, pf_thc_to_wais, pf_nino_to_wais,
                            pf_thc_to_nino, pf_amaz_to_nino, pf_nino_to_amaz, pf_thc_to_amaz)

################################# MAIN LOOP #################################
for kk in plus_minus_links:
    print("Wais to Thc:{}".format(kk[0]))
    print("Amaz to Nino:{}".format(kk[1]))
    print("Thc to Amaz:{}".format(kk[2]))
    try:
        os.stat("{}".format(long_save_name))
    except:
        os.makedirs("{}".format(long_save_name))

    try:
        os.stat("{}/{}_feedbacks".format(long_save_name, namefile))
    except:
        os.mkdir("{}/{}_feedbacks".format(long_save_name, namefile))

    try:
        os.stat("{}/{}_feedbacks/network_{}_{}_{}".format(long_save_name, namefile, kk[0], kk[1], kk[2]))
    except:
        os.mkdir("{}/{}_feedbacks/network_{}_{}_{}".format(long_save_name, namefile, kk[0], kk[1], kk[2]))

    try:
        os.stat("{}/{}_feedbacks/network_{}_{}_{}/{}".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4) ))
    except:
        os.mkdir("{}/{}_feedbacks/network_{}_{}_{}/{}".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4) ))

    #save starting conditions
    np.savetxt("{}/{}_feedbacks/network_{}_{}_{}/{}/empirical_values.txt".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4)), sys_var, delimiter=" ", fmt="%s")

    for strength in coupling_strength:
        print("Coupling strength: {}".format(strength))

        for GMT_file in GMT_files:
            print(GMT_file)
            parts = re.split("_|Tlim|Tpeak|tconv|.txt", GMT_file)
            T_lim  = int(parts[-6])
            T_peak = int(parts[-4])
            t_conv = int(parts[-2])
            
            GMT = np.loadtxt(GMT_file).T[-1]

            print("T_lim: {}°C".format(T_lim))
            print("T_peak: {}°C".format(T_peak))
            print("t_conv: {}yrs".format(t_conv))


            output = []
            for t in range(0, int(duration)):
                if os.path.isfile("{}/{}_feedbacks/network_{}_{}_{}/{}/feedbacks_Tlim{}_Tpeak{}_tconv{}_{:.2f}.txt".format(long_save_name, 
                    namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength)) == True:
                    print("File already computed")
                    break
                #print(t)
                #For feedback computations
                effective_GMT = GMT[t]

                #get back the network of the Earth system
                net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1], kk[2])

                # initialize state
                if t == 0:
                    initial_state = [-1, -1, -1, -1] #initial state
                else:
                    initial_state = [ev.get_timeseries()[1][-1, 0], ev.get_timeseries()[1][-1, 1], ev.get_timeseries()[1][-1, 2], ev.get_timeseries()[1][-1, 3]]
                ev = evolve(net, initial_state)
                # plotter.network(net)

                # Timestep to integration; it is also possible to run integration until equilibrium
                timestep = 0.1

                #t_end given in years; also possible to use equilibrate method
                t_end = 1.0/conv_fac_gis #simulation length in "real" years
                ev.integrate(timestep, t_end)


                #saving structure
                output.append([t,
                               ev.get_timeseries()[1][-1, 0],
                               ev.get_timeseries()[1][-1, 1],
                               ev.get_timeseries()[1][-1, 2],
                               ev.get_timeseries()[1][-1, 3],
                               net.get_number_tipped(ev.get_timeseries()[1][-1]),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[0]].count(True),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[1]].count(True),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[2]].count(True),
                               [net.get_tip_states(ev.get_timeseries()[1][-1])[3]].count(True)
                               ])



            #necessary for break condition
            if len(output) != 0:
                #saving structure
                data = np.array(output)
                np.savetxt("{}/{}_feedbacks/network_{}_{}_{}/{}/feedbacks_Tlim{}_Tpeak{}_tconv{}_{:.2f}.txt".format(long_save_name, 
                    namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength), data[-1])
                time = data.T[0]
                state_gis = data.T[1]
                state_thc = data.T[2]
                state_wais = data.T[3]
                state_amaz = data.T[4]

                #plotting structure
                fig = plt.figure()
                plt.grid(True)
                plt.title("Coupling strength: {}\n  Wais to Thc:{}  Amaz to Nino:{} Thc to Amaz:{} \n Tlim={}°C Tpeak={}°C tconv={}yr".format(
                    np.round(strength, 2), kk[0], kk[1], kk[2], T_lim, T_peak, t_conv))
                plt.plot(time, state_gis, label="GIS", color='c')
                plt.plot(time, state_thc, label="THC", color='b')
                plt.plot(time, state_wais, label="WAIS", color='k')
                plt.plot(time, state_amaz, label="AMAZ", color='g')
                plt.xlabel("Time [yr]")
                plt.ylabel("system feature f [a.u.]")
                plt.legend(loc='best')  # , ncol=5)
                plt.tight_layout()
                plt.savefig("{}/{}_feedbacks/network_{}_{}_{}/{}/feedbacks_Tlim{}_Tpeak{}_tconv{}_{:.2f}.pdf".format(long_save_name, namefile, 
                    kk[0], kk[1], kk[2], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength))
                #plt.show()
                plt.clf()
                plt.close()



    # it is necessary to limit the amount of saved files
    # --> compose one pdf file for each network setting and remove the other time-files
    current_dir = os.getcwd()
    os.chdir("{}/{}_feedbacks/network_{}_{}_{}/{}/".format(long_save_name, namefile, kk[0], kk[1], kk[2], str(mc_dir).zfill(4)))
    pdfs = np.array(np.sort(glob.glob("feedbacks_*.pdf"), axis=0))
    if len(pdfs) != 0.:
        merger = PdfFileMerger()
        for pdf in pdfs:
            merger.append(pdf)
        os.system("rm feedbacks_*.pdf")
        merger.write("feedbacks_complete.pdf")
        print("Complete PDFs merged")
    os.chdir(current_dir)

print("Finish")
#end = time.time()
#print("Time elapsed until Finish: {}s".format(end - start))