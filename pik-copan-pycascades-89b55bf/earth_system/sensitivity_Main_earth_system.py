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
from EWS_functions import autocorrelation, calc_autocorrelation, calc_variance

#private imports for earth system
from earth_sys.timing import timing
from earth_sys.functions_earth_system import global_functions
from earth_sys.earth import earth_system

temp_path = "/p/projects/dominoes/nicowun/conceptual_tipping/uniform_distribution/overshoot_study/temp_input/timeseries_final/"


#measure time
#start = time.time()

#############################GLOBAL SWITCHES#########################################
time_scale = True               # time scale of tipping is incorporated
plus_minus_include = True       # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
stochasticity = True            # gaussian noise is included in the tipping element evolution  
scaled_noise = True             # noise levels for each tipping element is scaled by the respective timescale    
ews_calculate = True            # early warning signals are calculated; requires stochasticity == True
#####################################################################
duration = 50000.               # actual real simulation years
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
#coupling_strength = np.linspace(0.0, 1.0, 11, endpoint=True)
coupling_strength = np.array([0.0])
#drive global mean temperature (GMT) above pre-industrial
#GMT_files = np.sort(glob.glob(temp_path+"*.txt"))
#GMT_files = [temp_path+"Tlim20_Tpeak35_tconv1000.txt"]
GMT_min = 0.0
GMT_max = 2.0
GMT_array = np.concatenate((np.concatenate((
                            np.linspace(GMT_min, GMT_min, 5000), 
                            np.linspace(GMT_min, GMT_max, 40000))), 
                            np.linspace(GMT_max, GMT_max, 5000)))
GMTs = [GMT_array]
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
All tipping times are computed in comparison to the Amazon rainforest tipping time. 
"""
if time_scale == True:
    print("Compute calibration timescale")
    #function call for absolute timing and time conversion
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
    noise = 0.5 # noise level (can be changed; from Laeo Crnkovic-Rubsamen: 0.01)
    if scaled_noise == True:
        sigma = np.diag([1./gis_time, 1./thc_time, 1./wais_time, 1./amaz_time])*noise # diagonal uncorrelated noise
    else:
        n = 4 # number of investigated tipping elements
        sigma = np.diag([1]*n)*noise # diagonal uncorrelated noise
else:
    sigma = None
    ews_calculate = False
    print("Early warning signals are not analysed as there is no stochasticity in the tipping elements")

# Set parameters for Early Warning Signal analysis
if ews_calculate == True:
    min_point = 0                                   # minimum number of states after which to start calculating EWS
    detrend_windows = np.arange(10000, 18000, 1000) # the length of the detrending window
    step_size = 10                                  # the number of states between each EWS calculation
    bws = np.arange(200,320, 20)                    # the bandwidth for filtering timeseries

    tip_t = np.abs(GMT_array-limits_gis).argmin() # find t where GIS reaches tipping element
    #tip_t = np.abs(GMT_array-limits_thc).argmin()
    #tip_t = np.abs(GMT_array-limits_wais).argmin()
    #tip_t = np.abs(GMT_array-limits_amaz).argmin()

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
        print("Coupling strength: {}".format(strength))
        
        """
        for GMT_file in GMT_files:
            #print(GMT_file)
            parts = re.split("_|Tlim|Tpeak|tconv|.txt", GMT_file)
            T_lim  = int(parts[-6])
            T_peak = int(parts[-4])
            t_conv = int(parts[-2])

            GMT = np.loadtxt(GMT_file).T[-1]
        """
        for GMT in GMTs:
            
            T_lim = GMT_min
            T_peak = GMT_max
            t_conv = 0

            print("T_lim: {}°C".format(T_lim))
            print("T_peak: {}°C".format(T_peak))
            print("t_conv: {}yrs".format(t_conv))

            output = []
            
            states_GIS = []; 
            states_THC = []; 
            states_WAIS = []; 
            states_AMAZ = []; 

            for t in range(0, int(duration)):
                
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

                ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                
                if ews_calculate == True:
                    
                    # save all states for ews analysis (relevant if t_end > timestep)
                    # if t !=0, do not count the initial state, which is the last state of the previous year
                    if t == 0:
                        states_GIS = np.append(states_GIS, ev.get_timeseries()[1][:, 0])
                        #states_THC = np.append(states_THC, ev.get_timeseries()[1][:, 1])
                        #states_WAIS = np.append(states_WAIS, ev.get_timeseries()[1][:, 2])
                        #states_AMAZ = np.append(states_AMAZ, ev.get_timeseries()[1][:, 3])
                    else:
                        states_GIS = np.append(states_GIS, ev.get_timeseries()[1][1:, 0])
                        #states_THC = np.append(states_THC, ev.get_timeseries()[1][1:, 1])
                        #states_WAIS = np.append(states_WAIS, ev.get_timeseries()[1][1:, 2])
                        #states_AMAZ = np.append(states_AMAZ, ev.get_timeseries()[1][1:, 3])
                    
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
            
            # Test sensitivity of autocorrelation and variance to detrending windows and filtering bandwiths 
            tau_GIS_autocorr = np.empty((len(detrend_windows), len(bws)))
            #tau_THC_autocorr = np.empty((len(detrend_windows), len(bws)))
            #tau_WAIS_autocorr = np.empty((len(detrend_windows), len(bws)))
            #tau_AMAZ_autocorr = np.empty((len(detrend_windows), len(bws)))
            
            for i in range(len(detrend_windows)):
                
                detrend_window = int(detrend_windows[i])
                print("Detrending window: ", detrend_window)
                
                for j in range(len(bws)):

                    bw = int(bws[j])
                    print("Filtering bandwidth: ", bw)
                    
                    last_point = 0
                    autocorr_GIS = []; variance_GIS = []; 
                    autocorr_THC = []; variance_THC = []; 
                    autocorr_WAIS = []; variance_WAIS = []; 
                    autocorr_AMAZ = []; variance_AMAZ = []; 

                    # Calculate autocorrelation and variance
                    # If number of states != number of years, tip_t must be modified
                    # There also needs to be one specific tip_t for each tipping element

                    autocorr_GIS, last_point0, ann_AC_GIS = calc_autocorrelation(states_GIS[:tip_t], last_point, autocorr_GIS,
                                                                        detrend_window, min_point, step_size, bw)
                    #autocorr_THC, last_point0, ann_AC_THC = calc_autocorrelation(states_THC, last_point, autocorr_THC,
                    #                                                detrend_window, min_point, step_size, bw)
                    #autocorr_WAIS, last_point0, ann_AC_WAIS = calc_autocorrelation(states_WAIS, last_point, autocorr_WAIS,
                    #                                                detrend_window, min_point, step_size, bw)
                    #autocorr_AMAZ, last_point0, ann_AC_AMAZ = calc_autocorrelation(states_AMAZ, last_point, autocorr_AMAZ,
                    #                                                detrend_window, min_point, step_size, bw)

                    #variance_GIS, last_point0, ann_var_GIS = calc_variance(states_GIS, last_point, variance_GIS,
                    #                                                detrend_window, min_point, step_size, bw)
                    #variance_THC, last_point0, ann_var_THC = calc_variance(states_THC, last_point, variance_THC,
                    #                                                detrend_window, min_point, step_size, bw)
                    #variance_WAIS, last_point0, ann_var_WAIS = calc_variance(states_WAIS, last_point, variance_WAIS,
                    #                                                detrend_window, min_point, step_size, bw)
                    #variance_AMAZ, last_point0, ann_var_AMAZ = calc_variance(states_AMAZ, last_point, variance_AMAZ,
                    #                                                detrend_window, min_point, step_size, bw)

                    # Calculate Kendall tau correlation for autocorrelation and variance

                    Tau_AC, p_value = kendalltau(autocorr_GIS, np.arange(len(autocorr_GIS)))
                    #Tau_var, p_value = kendalltau(variance_GIS, np.arange(len(variance_GIS)))
                    tau_GIS_autocorr[i, j] = Tau_AC
                    #tau_GIS_variance = np.append(tau_GIS_variance, Tau_var)
            
                    #Tau_AC, p_value = kendalltau(autocorr_THC, np.arange(len(autocorr_THC)))
                    #Tau_var, p_value = kendalltau(variance_THC, np.arange(len(variance_THC)))
                    #tau_THC_autocorr[i, j] = Tau_AC
                    #tau_THC_variance = np.append(tau_THC_variance, Tau_var)

                    #Tau_AC, p_value = kendalltau(autocorr_WAIS, np.arange(len(autocorr_WAIS)))
                    #Tau_var, p_value = kendalltau(variance_WAIS, np.arange(len(variance_WAIS)))
                    #tau_WAIS_autocorr[i, j] = Tau_AC
                    #tau_WAIS_variance = np.append(tau_WAIS_variance, Tau_var)

                    #Tau_AC, p_value = kendalltau(autocorr_AMAZ, np.arange(len(autocorr_AMAZ)))
                    #Tau_var, p_value = kendalltau(variance_AMAZ, np.arange(len(variance_AMAZ)))
                    #tau_AMAZ_autocorr[i, j] = Tau_AC
                    #tau_AMAZ_variance = np.append(tau_AMAZ_variance, Tau_var)


            #necessary for break condition
            if len(output) != 0:
                #saving structure
                data = np.array(output)
                np.savetxt("{}/feedbacks/network_{}_{}/{}/feedbacks_Tlim{}_Tpeak{}_tconv{}_{:.2f}.txt".format(long_save_name, 
			        kk[0], kk[1], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength), data)
                time = data.T[0]
                state_gis = data.T[1]
                state_thc = data.T[2]
                state_wais = data.T[3]
                state_amaz = data.T[4]
                """
                #plotting structure
                fig = plt.figure()
                plt.grid(True)
                plt.title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                #plt.plot(time, state_gis, label="GIS", color='c')
                plt.plot(time, gaussian_filter1d(state_gis, bw), color = 'grey')
                #plt.plot(time, state_thc, label="THC", color='b')
                #plt.plot(time, state_wais, label="WAIS", color='k')
                #plt.plot(time, state_amaz, label="AMAZ", color='g')
                #plt.plot(time, gaussian_filter1d(state_amaz, bw), color = 'r')
                plt.axvline(tip_t, color='k', linestyle='--')
                plt.xlabel("Time [yr]")
                plt.ylabel("system feature f [a.u.]")
                plt.legend(loc='best')  # , ncol=5)
                ax2 = plt.gca().twinx()
                ax2.plot(time, GMT[:int(duration)], color='r')
                ax2.grid(False)
                ax2.set_ylabel("$\Delta$GMT")
                plt.tight_layout()
                plt.savefig("{}/feedbacks/network_{}_{}/{}/feedbacks_Tlim{}_Tpeak{}_tconv{}_{:.2f}.pdf".format(long_save_name,
			    kk[0], kk[1], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength))
                #plt.show()
                plt.clf()
                plt.close()
                """

                # Plot Kendall tau for autocorrelation
                fig, ax = plt.subplots()
                ax.set_title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                ax.set_ylabel("Detrending window")
                ax.set_xlabel("Filtering bandwidth")
                X, Y = np.meshgrid(bws, detrend_windows)
                CS = ax.contourf(X, Y, tau_GIS_autocorr)
                cbar = fig.colorbar(CS)
                cbar.set_label(r"Kendall $\tau$ at $T_{crit}$")
                fig.tight_layout()
                fig.savefig("{}/feedbacks/network_{}_{}/{}/AC_sensitivity_Tlim{}_Tpeak{}_tconv{}_{:.2f}.pdf".format(long_save_name,
			        kk[0], kk[1], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength))
                #plt.show()
                fig.clf()
                plt.close()
                
                """ 
                 # Plot variance
                    fig, ax1 = plt.subplots(1,1)
                    ax2 = ax1.twinx()
                    ax1.grid(True)
                    ax2.grid(False)
                    fig.suptitle("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                    ax1.scatter(time, ann_variance_GIS, label="GIS", color='c', s=2)
                    #ax1.scatter(time, ann_variance_THC, label="THC", color='b', s=2)
                    #ax1.scatter(time, ann_variance_WAIS, label="WAIS", color='k', s=2)
                    #ax1.scatter(time, ann_variance_AMAZ, label="AMAZ", color='g', s=2)
                    ax2.plot(time, tau_GIS_variance, color='r')
                    ax1.axvline(tip_t, color='k', linestyle='--')
                    ax1.set_xlabel("Time [yr]")
                    ax1.set_ylabel(r"Variance [a.u.$^2$]")
                    ax2.set_ylabel(r"Kendall $\tau$ correlation")
                    ax1.legend(loc='best')  # , ncol=5)
                    fig.tight_layout()
                    fig.savefig("{}/feedbacks/network_{}_{}/{}/var_sensitivity_Tlim{}_Tpeak{}_tconv{}_{:.2f}.pdf".format(long_save_name,
			        kk[0], kk[1], str(mc_dir).zfill(4), T_lim, T_peak, t_conv, strength))
                    #plt.show()
                    fig.clf()
                    plt.close()
                    """
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

