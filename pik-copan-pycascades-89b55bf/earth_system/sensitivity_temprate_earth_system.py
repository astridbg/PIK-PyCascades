# Add modules directory to path
import os
import sys
import re

sys.path.append('')
sys.path.append('../modules/gen')
sys.path.append('../modules/core')

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
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau

# private imports from sys.path
#from evolve import evolve # astridg commented out
from evolve_sde import evolve # astridg edit
from EWS_functions_temprate import autocorrelation, calc_autocorrelation, calc_variance

#private imports for earth system
from earth_sys.timing import timing
from earth_sys.functions_earth_system import global_functions
from earth_sys.earth import earth_system

#measure time
#start = time.time()

#############################GLOBAL SWITCHES#########################################
time_scale = True               # time scale of tipping is incorporated
plus_minus_include = False      # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
stochasticity = True            # gaussian noise is included in the tipping element evolution  
scaled_noise = False            # noise levels for each tipping element is scaled by the respective timescale    
ews_calculate = True            # early warning signals are calculated; requires stochasticity == True
#####################################################################
duration = 50000
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
coupling_strength = np.array([0])
#drive global mean temperature (GMT) above pre-industrial
T_end = 5.                             # final GMT increase [deg] 
T_rates = [1./10000]                   # GMT increase rate [deg/yr]
GMT_arrays = []
for T_rate in T_rates:
    increase_length = int(T_end/T_rate)
    increase_arr = np.linspace(0, T_end, increase_length)
    buffer_arr = np.zeros(duration-increase_length)
    GMT_arr = np.concatenate((buffer_arr, increase_arr))
    GMT_arrays.append(GMT_arr)

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

#------------------------------------------

#directories for the Monte Carlo simulation
mc_dir = int(sys_var[-1])

###################### Initialize switches ##########################

# Time scale
"""
All tipping times are computed ion comparison to the Amazon rainforest tipping time.
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

# Include uncertain "+-" links:
if plus_minus_include == True:
    plus_minus_links = np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)))
else:
    plus_minus_links = [np.array([1., 1., 1.])]

# Include stochasticity in the tipping elements
if stochasticity == True:
    noise = 0.005   # noise level (very changeable; from Laeo Crnkovic-Rubsamen: 0.01)
    n = 4           # number of investigated tipping elements
    if scaled_noise == True:
        sigma = np.diag([1./gis_time, 1./thc_time, 1./wais_time, 1./amaz_time])*noise # diagonal uncorrelated noise
    else:
        sigma = np.diag([1]*n)*noise # diagonal uncorrelated noise
else:
    sigma = None
    if ews_calculate == True:
        ews_calculate = False
        print("Early warning signals are not analysed as there is no stochasticity in the tipping elements")

# Set parameters for Early Warning Signal analysis
if ews_calculate == True:
    step_size = 10                                  # the number of states between each EWS calculation
    bandwidth = np.arange(10, 250, 10)              # the bandwidth for filtering timeseries
    tip_threshold = -1./np.sqrt(3)                  # the analytical state value for element to reach tipping    

#######################INTEGRATION PARAMETERS########################
# Timestep to integration; it is also possible to run integration until equilibrium
timestep = 0.1
#t_end in each integration loop given in years; also possible to use equilibrate method
t_end = 1.0/conv_fac_gis # simulation length in "real" years

################################# MAIN #################################
#Create Earth System
earth_system = earth_system(gis_time, thc_time, wais_time, amaz_time,
                            limits_gis, limits_thc, limits_wais, limits_amaz,
                            pf_wais_to_gis, pf_thc_to_gis, pf_gis_to_thc,
                            pf_wais_to_thc, pf_gis_to_wais, pf_thc_to_wais, pf_thc_to_amaz)

################################# MAIN LOOP #################################
for kk in plus_minus_links:
    print("Wais to Thc:{}".format(kk[0]))
    print("Thc to Amaz:{}".format(kk[1]))
    
    try:
        os.stat("{}/sensitivity".format(long_save_name))
    except:
        os.mkdir("{}/sensitivity".format(long_save_name))

    try:
        os.stat("{}/sensitivity/network_{}_{}".format(long_save_name, kk[0], kk[1]))
    except:
        os.mkdir("{}/sensitivity/network_{}_{}".format(long_save_name, kk[0], kk[1]))

    try:
        os.stat("{}/sensitivity/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    except:
        os.mkdir("{}/sensitivity/network_{}_{}/{}".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4) ))
    
    #save starting conditions
    np.savetxt("{}/sensitivity/network_{}_{}/{}/empirical_values.txt".format(long_save_name, kk[0], kk[1], str(mc_dir).zfill(4)), sys_var)

    for strength in coupling_strength:
        print("Coupling strength: {}".format(strength))
        
        for T_ind in range(len(T_rates)):
            
            T_rate = T_rates[T_ind]
            GMT = GMT_arrays[T_ind]
            duration = len(GMT)
            print("Temperature rate: {}°C/yr".format(T_rate))
            print("Length of simulation: {}yr".format(duration))
            t = 0               # Start time
            effective_GMT = GMT[t]   # Start temperature
            
            output = []

            # initialize intermediate storage arrays for EWS analysis
            states = np.empty((n, duration)) 
            Tau_autocorr = np.empty((n, len(bandwidth)))
            Tau_variance = np.empty((n, len(bandwidth)))
            state_tipped = [False, False, False, False]
            start_point = [0, 0, 0, 0]
            tip_t = [np.nan, np.nan, np.nan, np.nan]
            detrend_window = [np.nan, np.nan, np.nan, np.nan]

            # set up the network of the Earth System
            net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

            # initialize state
            initial_state = [-1, -1, -1, -1]
            ev = evolve(net, initial_state)
            
            # evolve
            ev.integrate( timestep, t_end, initial_state, sigma=sigma)
            
            for elem in range(n):
                # store states intermediately for analysis
                states[elem, t] = ev.get_timeseries()[1][-1, elem]

            # saving structure
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

            for t in range(1, duration):
                
                # increase GMT by set rate
                effective_GMT = GMT[t]
                print(t)

                # get back the network of the Earth system
                net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

                # initialize state
                initial_state = [ev.get_timeseries()[1][-1, 0], ev.get_timeseries()[1][-1, 1], ev.get_timeseries()[1][-1, 2], ev.get_timeseries()[1][-1, 3]]
                ev = evolve(net, initial_state)
                
                # evolve
                ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                
                for elem in range(n):

                    states[elem, t] = ev.get_timeseries()[1][-1, elem]
                    
                    # check if the element is in a tipped state
                    if state_tipped[elem] == False and states[elem, t] >= tip_threshold + noise:
                        state_tipped[elem] = True
                        tip_t[elem] = t
                        # set the detrending window to half the tipping length
                        detrend_window[elem] = int(t/2)
                        
                        # calculate autocorrelation and variance for the points in the fixed time window
                        for jj in range(len(bandwidth)):
                            autocorr_elem = calc_autocorrelation(states[elem,:t], step_size, detrend_window[elem], bandwidth[jj])
                            variance_elem = calc_variance(states[elem,:t], step_size, detrend_window[elem], bandwidth[jj])

                            # calculate Kendall Tau correlation of autocorrelation and variance
                            Tau_autocorr[elem, jj], p_value = kendalltau(autocorr_elem, np.arange(len(autocorr_elem)))
                            Tau_variance[elem, jj], p_value = kendalltau(variance_elem, np.arange(len(variance_elem)))

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
                np.savetxt("{}/sensitivity/network_{}_{}/{}/feedbacks_Tend{}_Trate{}_d{:.2f}_n{}.txt".format(long_save_name, 
			        kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise), data)
                np.savetxt("{}/sensitivity/network_{}_{}/{}/Tau_AC_Tend{}_Trate{}_d{:.2f}_n{}.txt".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise), Tau_autocorr)
                np.savetxt("{}/sensitivity/network_{}_{}/{}/Tau_var_Tend{}_Trate{}_d{:.2f}_n{}.txt".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise), Tau_variance)
                # Plotting structure
                time = data.T[0]
                colors = ['c','b','k','g']
                labels = ['GIS', 'THC', 'WAIS', 'AMAZ']

                fig = plt.figure()
                plt.grid(True)
                plt.title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                for elem in range(n):
                    plt.plot(time, data.T[elem+1], label=labels[elem], color=colors[elem])
                plt.xlabel("Time [yr]")
                plt.ylabel("System feature f [a.u.]")
                plt.legend(loc='best')
                ax2 = plt.gca().twinx()
                ax2.plot(time, GMT, color='r')
                ax2.grid(False)
                ax2.set_ylabel("$\Delta$GMT")
                ax2.yaxis.label.set_color('r')
                plt.tight_layout()
                fig.savefig("{}/sensitivity/network_{}_{}/{}/feedbacks_Tend{}_Trate{}_d{:.2f}_n{}.png".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise))
                plt.clf()
                plt.close()

                    
                fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10,5), sharey=True)
                fig.suptitle("Coupling strength: {}, Wais to Thc:{} Thc to Amaz:{}, Noise level:{}".format(np.round(strength, 2), kk[0], kk[1], noise))

                # Plot autocorrelation

                ax1.grid(True)
                for elem in range(n):
                    ax1.plot(bandwidth, Tau_autocorr[elem, :], c=colors[elem])
                    ax1.scatter(bandwidth, Tau_autocorr[elem, :], c=colors[elem], label=labels[elem])
                ax1.set_ylabel(r"Kendall $\tau$ correlation")
                ax1.set_ylim(-1, 1)
                ax1.set_xlabel("Filtering bandwidth")
                ax1.set_title("Autocorrelation")

                # Plot residual
                ax2.grid(True)
                for elem in range(n):
                    ax2.plot(bandwidth, Tau_variance[elem,:], c=colors[elem])
                    ax2.scatter(bandwidth, Tau_variance[elem,:], c=colors[elem], label=labels[elem])
                ax2.set_xlabel("Filtering bandwidth")
                ax2.legend(loc="best")
                ax2.set_title("Variance")

                    
                fig.tight_layout()
                fig.savefig("{}/sensitivity/network_{}_{}/{}/bandwidth_Tend{}_Trate{}_d{:.2f}_n{}.png".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise))
                plt.clf()
                plt.close()

"""
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
"""
print("Finish")
#end = time.time()
#print("Time elapsed until Finish: {}s".format(end - start))

