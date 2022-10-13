# Add modules directory to path
import os
import sys

sys.path.append('')
sys.path.append('../modules/gen')
sys.path.append('../modules/core')

# global imports
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.)
import itertools
import time
import glob
from PyPDF2 import PdfFileMerger


# private imports from sys.path
#from evolve import evolve # astridg commented out
from evolve_sde import evolve # astridg edit
from EWS_functions import calc_autocorrelation, autocorrelation

#private imports for earth system
from earth_sys.timing import timing
from earth_sys.functions_earth_system import global_functions
from earth_sys.earth import earth_system

#measure time
#start = time.time()

###MAIN

# astridg added from another program
#############################GLOBAL SWITCHES#########################################
time_scale = True               # time scale of tipping is incorporated
plus_minus_include = True       # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
ews_calculate = True            # early warning signals are calculated
#####################################################################
duration = 10000.               # actual real simulation years
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
#strength = 0.25 # astridg commented out
#coupling_strength = np.linspace(0.0, 1.0, 11, endpoint=True)
coupling_strength = np.array([0.25])
#drive global mean temperature (GMT) above pre-industrial
#GMT = 2.0 # astridg commented out
GMTs = [np.linspace(0.0,0.0 , int(duration))]
#GMTs = np.fill(int(duration))
##########Characteristics of noise###################################
# Define sigma for random processes
noise = 0.01                    # noise level (can be changed; from Laeo Crnkovic-Rubsamen: 0.01)
n = 4                           # number of investigated tipping elements
sigma = np.diag([1]*n)*noise    # diagonal uncorrelated noise
# put sigma to "None" to exclude noise

##########Variables for Early Warning Signal analysis################
min_point = 100
detrend_window = 5000
step_size = 10

########################Declaration of variables from passed values#######################
sys_var = np.array(sys.argv[2:], dtype=float)
print(sys_var)

#Tipping ranges from distribution
limits_gis, limits_thc, limits_wais, limits_amaz = sys_var[0], sys_var[1], sys_var[2], sys_var[3]

#Probability fractions
# TO GIS
pf_wais_to_gis, pf_thc_to_gis = sys_var[4], sys_var[5]
# TO THC
pf_gis_to_thc, pf_wais_to_thc = sys_var[6], sys_var[7]
# TO WAIS
pf_thc_to_wais, pf_gis_to_wais = sys_var[8], sys_var[9]
# TO AMAZ
pf_thc_to_amaz = sys_var[10]


#astridg commented out
#Time scale
#print("Compute calibration timescale")
#function call for absolute timing and time conversion
#time_props = timing()
#gis_time, thc_time, wais_time, amaz_time = time_props.timescales()
#conv_fac_gis = time_props.conversion()

# Time scale
"""
All tipping times are computed ion comparison to the Amazon rainforest tipping time. As this is variable now, this affects the results to a (very) level
"""
if time_scale == True:
    print("Compute calibration timescale")
    #function call for absolute timing and time conversion
    #time_props = timing(tau_gis, tau_thc, tau_wais, tau_amaz)
    time_props = timing()
    gis_time, thc_time, wais_time, amaz_time = time_props.timescales()
    conv_fac_gis = time_props.conversion()
else:
    #no time scales included
    gis_time = thc_time = wais_time = amaz_time = 1.0
    conv_fac_gis = 1.0

# astridg commented out
#plus_minus_links = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)))
#directories for the Monte Carlo simulation
#mc_dir = int(sys_var[-1])

# Include uncertain "+-" links:
if plus_minus_include == True:
    plus_minus_links = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)))
else:
    plus_minus_links = [np.array([1., 1., 1.])]

#directories for the Monte Carlo simulation
mc_dir = int(sys_var[-1])

################################# MAIN #################################
#Create Earth System
earth_system = earth_system(gis_time, thc_time, wais_time, amaz_time,
                            limits_gis, limits_thc, limits_wais, limits_amaz,
                            pf_wais_to_gis, pf_thc_to_gis, pf_gis_to_thc,
                            pf_wais_to_thc, pf_gis_to_wais, pf_thc_to_wais, pf_thc_to_amaz)

################################# MAIN LOOP #################################
#for kk in plus_minus_links: # astridg commented out for fast checking 
for kk in [plus_minus_links[0]]:
    print("Wais to Thc:{}".format(kk[0]))
    print("Thc to Amaz:{}".format(kk[1]))
    
    try:
        os.stat("{}/feedbacks".format(long_save_name))
    except:
        os.mkdir("{}/feedbacks".format(long_save_name))

    try:
        os.stat("{}/feedbacks/network_{}_{}".format(long_save_name, kk[0], kk[1]))
    except:
        os.mkdir("{}/feedbacks/network_{}_{}".format(long_save_name, kk[0], kk[1]))

    try:
        os.stat("{}/feedbacks/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    except:
        os.mkdir("{}/feedbacks/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    
    #save starting conditions
    np.savetxt("{}/feedbacks/network_{}_{}/{}/empirical_values.txt".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4)), sys_var)

    for strength in coupling_strength:

        for GMT in GMTs:
            
            output = []
            times = []
            states_GIS = []; autocorr_GIS = []; ann_autocorr_GIS = []
            states_THC = []; autocorr_THC = []; ann_autocorr_THC = []
            states_WAIS = []; autocorr_WAIS = []; ann_autocorr_WAIS = []
            states_AMAZ = []; autocorr_AMAZ = []; ann_autocorr_AMAZ = []
            last_point = 0

            for t in range(0, int(duration)):
                #print("t: ", t)
                effective_GMT = GMT[t]

                #get back the network of the Earth system
                net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

                # initialize state
                if t == 0:
                    initial_state = [-1, -1, -1, -1]
                else:
                    initial_state = [ev.get_timeseries()[1][-1, 0], ev.get_timeseries()[1][-1, 1], ev.get_timeseries()[1][-1, 2], ev.get_timeseries()[1][-1, 3]]
                ev = evolve(net, initial_state)
                # plotter.network(net)

                # Timestep to integration; it is also possible to run integration until equilibrium
                timestep = 0.1
                #t_end given in years; also possible to use equilibrate method
                t_end = 1.0/conv_fac_gis # simulation length in "real" years 
                
                ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                
                if ews_calculate == True:
                    
                    # save all states for ews analysis
                    # if t !=0, do not count the first state, as it is initialized from the previous year
                    if t == 0:
                        states_GIS = np.append(states_GIS, ev.get_timeseries()[1][:, 0])
                        states_THC = np.append(states_THC, ev.get_timeseries()[1][:, 1])
                        states_WAIS = np.append(states_WAIS, ev.get_timeseries()[1][:, 2])
                        states_AMAZ = np.append(states_AMAZ, ev.get_timeseries()[1][:, 3])
                    else:
                        states_GIS = np.append(states_GIS, ev.get_timeseries()[1][1:, 0])
                        states_THC = np.append(states_THC, ev.get_timeseries()[1][1:, 1])
                        states_WAIS = np.append(states_WAIS, ev.get_timeseries()[1][1:, 2])
                        states_AMAZ = np.append(states_AMAZ, ev.get_timeseries()[1][1:, 3])
                    
                    if len(states_GIS) > max(min_point+detrend_window, last_point+detrend_window):
                        
                        autocorr_GIS, last_point0, ann_mean_GIS = calc_autocorrelation(states_GIS, last_point, autocorr_GIS,
                                                                        detrend_window, min_point, step_size)
                        autocorr_THC, last_point0, ann_mean_THC = calc_autocorrelation(states_THC, last_point, autocorr_THC,
                                                                        detrend_window, min_point, step_size)
                        autocorr_WAIS, last_point0, ann_mean_WAIS = calc_autocorrelation(states_WAIS, last_point, autocorr_WAIS,
                                                                        detrend_window, min_point, step_size)
                        autocorr_AMAZ, last_point0, ann_mean_AMAZ = calc_autocorrelation(states_AMAZ, last_point, autocorr_AMAZ,
                                                                        detrend_window, min_point, step_size)
                        last_point = last_point0

                        ann_autocorr_GIS = np.append(ann_autocorr_GIS, ann_mean_GIS)
                        ann_autocorr_THC = np.append(ann_autocorr_THC, ann_mean_THC)
                        ann_autocorr_WAIS = np.append(ann_autocorr_WAIS, ann_mean_WAIS)
                        ann_autocorr_AMAZ = np.append(ann_autocorr_AMAZ, ann_mean_AMAZ)

                        # here I want to get the mean autocorrelation value each year
                        # and also the slope
                    else:
                        ann_autocorr_GIS = np.append(ann_autocorr_GIS, np.nan)
                        ann_autocorr_THC = np.append(ann_autocorr_THC, np.nan)
                        ann_autocorr_WAIS = np.append(ann_autocorr_WAIS, np.nan)
                        ann_autocorr_AMAZ = np.append(ann_autocorr_AMAZ, np.nan)

 
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

            #iprint(mean_autocorr_GIS)
            #print(len(mean_autocorr_GIS))
            #necessary for break condition
            if len(output) != 0:
                #saving structure
                data = np.array(output)
                np.savetxt("{}/feedbacks/network_{}_{}/{}/feedbacks_{:.2f}.txt".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4), strength), data)
                time = data.T[0]
                state_gis = data.T[1]
                state_thc = data.T[2]
                state_wais = data.T[3]
                state_amaz = data.T[4]

                #plotting structure
                fig = plt.figure()
                plt.grid(True)
                plt.title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                plt.plot(time, state_gis, label="GIS", color='c')
                plt.plot(time, state_thc, label="THC", color='b')
                plt.plot(time, state_wais, label="WAIS", color='k')
                plt.plot(time, state_amaz, label="AMAZ", color='g')
                plt.xlabel("Time [yr]")
                plt.ylabel("system feature f [a.u.]")
                plt.legend(loc='best')  # , ncol=5)
                plt.tight_layout()
                plt.savefig("{}/feedbacks/network_{}_{}/{}/time_d{:.2f}.pdf".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4), np.round(strength, 2)))
                #plt.show()
                plt.clf()
                plt.close()

                fig, ax1 = plt.subplots(1,1)
                ax2 = ax1.twinx()
                ax1.grid(True)
                ax2.grid(False)
                fig.suptitle("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                ax1.scatter(time, ann_autocorr_GIS, label="GIS", color='c', s=2)
                ax1.scatter(time, ann_autocorr_THC, label="THC", color='b', s=2)
                ax1.scatter(time, ann_autocorr_WAIS, label="WAIS", color='k', s=2)
                ax1.scatter(time, ann_autocorr_AMAZ, label="AMAZ", color='g', s=2)
                ax2.plot(time, GMT, color='r')
                ax1.set_xlabel("Time [yr]")
                ax1.set_ylabel("Autocorrelation")
                ax2.set_ylabel(r"$\Delta$GMT")
                ax1.legend(loc='best')  # , ncol=5)
                fig.tight_layout()
                fig.savefig("{}/feedbacks/network_{}_{}/{}/AC_time_d{:.2f}.pdf".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4), np.round(strength, 2)))
                #plt.show()
                fig.clf()
                plt.close()

    # it is necessary to limit the amount of saved files
    # --> compose one pdf file for each network setting and remove the other time-files
    current_dir = os.getcwd()
    os.chdir("{}/feedbacks/network_{}_{}/{}/".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4)))
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

