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
plus_minus_include = False      # from Kriegler, 2009: Unclear links; if False all unclear links are set to off state and only network "0-0" is computed
stochasticity = True            # gaussian noise is included in the tipping element evolution  
scaled_noise = False            # noise levels for each tipping element is scaled by the respective timescale    
ews_calculate = True            # early warning signals are calculated; requires stochasticity == True
#####################################################################
duration = 50000.               # actual real simulation years
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
coupling_strength = np.array([0.0])
#drive global mean temperature (GMT) above pre-industrial
T_end = 3.5                 # GMT increase to end on [deg] 
T_rates = [1./10000, 1./200, 1./100]  # GMT rate to increase by [deg/yr]

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
    noise = 0.005 # noise level (can be changed; from Laeo Crnkovic-Rubsamen: 0.01)
    if scaled_noise == True:
        sigma = np.diag([1./gis_time, 1./thc_time, 1./wais_time, 1./amaz_time])*noise # diagonal uncorrelated noise
    else:
        n = 4 # number of investigated tipping elements
        sigma = np.diag([1]*n)*noise # diagonal uncorrelated noise
else:
    sigma = None
    if ews_calculate == True:
        ews_calculate = False
        print("Early warning signals are not analysed as there is no stochasticity in the tipping elements")

# Set parameters for Early Warning Signal analysis
if ews_calculate == True:
    min_point = 0			# minimum number of states after which to start calculating EWS
    detrend_window = 15000 		# the length of the detrending window
    step_size = 10			# the number of states between each EWS calculation
    bw = 200                            # the bandwidth for filtering timeseries
    min_len = 25			# smallest sample number to start evaluating
    
    tip_t_gis = np.abs(GMT_array-limits_gis).argmin() # find t where GIS reaches tipping element
    tip_t_thc = np.abs(GMT_array-limits_thc).argmin() # find t where THC reaches tipping element
    tip_t_wais = np.abs(GMT_array-limits_wais).argmin() # find t where WAIS reaches tipping element
    tip_t_amaz = np.abs(GMT_array-limits_amaz).argmin() # find t where AMAZ reaches tipping element

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
for kk in plus_minus_links:
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
        for T_rate in GMT_rates: 
            print("T_rate: {}°C/yr".format(T_rate))
            
            output = []
            
            t = 0
            effective_GMT = 0
            
            # set up the network of the Earth System
            net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

            # initialize state
            initial_state = [-1, -1, -1, -1]
            ev = evolve(net, initial_state)
            
            # evolve
            ev.integrate( timestep, t_end, initial_state, sigma=sigma)
            
            # set the autocorrelation and variance of the tipping elements to NaN
            ac_gis = np.nan; var_gis = np.nan
            ac_thc = np.nan; var_thc = np.nan
            ac_wais = np.nan; var_wais = np.nan
            ac_amaz = np.nan; var_amaz = np.nan
            
            # set the last point for calculating autocorrelation and variance to zero
            lp_gis = lp_thc = lp_wais = lp_amaz = 0

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
                            [net.get_tip_states(ev.get_timeseries()[1][-1])[3]].count(True),
                            ac_gis, ac_thc, ac_wais, ac_amaz,
                            var_gis, var_thc, var_wais, var_amaz
                            ])

            while effective_GMT <= T_end:
                
                t += 1
                effective_GMT += T_rate

                #get back the network of the Earth system
                net = earth_system.earth_network(effective_GMT, strength, kk[0], kk[1])

                # initialize state
                initial_state = [ev.get_timeseries()[1][-1, 0], ev.get_timeseries()[1][-1, 1], ev.get_timeseries()[1][-1, 2], ev.get_timeseries()[1][-1, 3]]
                ev = evolve(net, initial_state)
                ev.integrate( timestep, t_end, initial_state, sigma=sigma)
                
                if ews_calculate == True:
                    
                    # save all states for ews analysis (relevant if t_end > timestep)
                    # if t !=0, do not count the initial state, which is the last state of the previous year
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
                        
                        # Calculate autocorrelation and variance

                        autocorr_GIS, last_point0, ann_AC_GIS = calc_autocorrelation(states_GIS, last_point, autocorr_GIS,
                                                                        detrend_window, min_point, step_size, bw)
                        autocorr_THC, last_point0, ann_AC_THC = calc_autocorrelation(states_THC, last_point, autocorr_THC,
                                                                        detrend_window, min_point, step_size, bw)
                        autocorr_WAIS, last_point0, ann_AC_WAIS = calc_autocorrelation(states_WAIS, last_point, autocorr_WAIS,
                                                                        detrend_window, min_point, step_size, bw)
                        autocorr_AMAZ, last_point0, ann_AC_AMAZ = calc_autocorrelation(states_AMAZ, last_point, autocorr_AMAZ,
                                                                        detrend_window, min_point, step_size, bw)
                        
                        variance_GIS, last_point0, ann_var_GIS = calc_variance(states_GIS, last_point, variance_GIS,
                                                                        detrend_window, min_point, step_size, bw)
                        variance_THC, last_point0, ann_var_THC = calc_variance(states_THC, last_point, variance_THC,
                                                                        detrend_window, min_point, step_size, bw)
                        variance_WAIS, last_point0, ann_var_WAIS = calc_variance(states_WAIS, last_point, variance_WAIS,
                                                                        detrend_window, min_point, step_size, bw)
                        variance_AMAZ, last_point0, ann_var_AMAZ = calc_variance(states_AMAZ, last_point, variance_AMAZ,
                                                                        detrend_window, min_point, step_size, bw)
                        
                        # Save the last point to start up again in next round
                        last_point = last_point0 

                        # Store the variance and autocorrelation calculated for this year
                        ann_autocorr_GIS = np.append(ann_autocorr_GIS, ann_AC_GIS)
                        ann_autocorr_THC = np.append(ann_autocorr_THC, ann_AC_THC)
                        ann_autocorr_WAIS = np.append(ann_autocorr_WAIS, ann_AC_WAIS)
                        ann_autocorr_AMAZ = np.append(ann_autocorr_AMAZ, ann_AC_AMAZ)
                        ann_variance_GIS = np.append(ann_variance_GIS, ann_var_GIS)
                        ann_variance_THC = np.append(ann_variance_THC, ann_var_THC)
                        ann_variance_WAIS = np.append(ann_variance_WAIS, ann_var_WAIS)
                        ann_variance_AMAZ = np.append(ann_variance_AMAZ, ann_var_AMAZ)

                    else:

                        # If no autocorrelation and variance was calculated this year, append NaN
                        ann_autocorr_GIS = np.append(ann_autocorr_GIS, np.nan)
                        ann_autocorr_THC = np.append(ann_autocorr_THC, np.nan)
                        ann_autocorr_WAIS = np.append(ann_autocorr_WAIS, np.nan)
                        ann_autocorr_AMAZ = np.append(ann_autocorr_AMAZ, np.nan)
                        ann_variance_GIS = np.append(ann_variance_GIS, np.nan)
                        ann_variance_THC = np.append(ann_variance_THC, np.nan)
                        ann_variance_WAIS = np.append(ann_variance_WAIS, np.nan)
                        ann_variance_AMAZ = np.append(ann_variance_AMAZ, np.nan)
                    
                    # If the autocorrelation and variance timeseries is long enough, start calculating Kendall tau correlation

                    if len(autocorr_GIS) > min_len:
                        
                        Tau_AC, p_value = kendalltau(autocorr_GIS, np.arange(len(autocorr_GIS)))
                        Tau_var, p_value = kendalltau(variance_GIS, np.arange(len(variance_GIS)))
                        tau_autocorr_GIS = np.append(tau_autocorr_GIS, Tau_AC)
                        tau_variance_GIS = np.append(tau_variance_GIS, Tau_var)

                        Tau_AC, p_value = kendalltau(autocorr_THC, np.arange(len(autocorr_THC)))
                        Tau_var, p_value = kendalltau(variance_THC, np.arange(len(variance_THC)))
                        tau_autocorr_THC = np.append(tau_autocorr_THC, Tau_AC)
                        tau_variance_THC = np.append(tau_variance_THC, Tau_var)

                        Tau_AC, p_value = kendalltau(autocorr_WAIS, np.arange(len(autocorr_WAIS)))
                        Tau_var, p_value = kendalltau(variance_WAIS, np.arange(len(variance_WAIS)))
                        tau_autocorr_WAIS = np.append(tau_autocorr_WAIS, Tau_AC)
                        tau_variance_WAIS = np.append(tau_variance_WAIS, Tau_var)

                        Tau_AC, p_value = kendalltau(autocorr_AMAZ, np.arange(len(autocorr_AMAZ)))
                        Tau_var, p_value = kendalltau(variance_AMAZ, np.arange(len(variance_AMAZ)))
                        tau_autocorr_AMAZ = np.append(tau_autocorr_AMAZ, Tau_AC)
                        tau_variance_AMAZ = np.append(tau_variance_AMAZ, Tau_var)

                    else:
                        tau_autocorr_GIS = np.append(tau_autocorr_GIS, np.nan)
                        tau_variance_GIS = np.append(tau_variance_GIS, np.nan)
                        
                        tau_autocorr_THC = np.append(tau_autocorr_THC, np.nan)
                        tau_variance_THC = np.append(tau_variance_THC, np.nan)
                        
                        tau_autocorr_WAIS = np.append(tau_autocorr_WAIS, np.nan)
                        tau_variance_WAIS = np.append(tau_variance_WAIS, np.nan)
                        
                        tau_autocorr_AMAZ = np.append(tau_autocorr_AMAZ, np.nan)
                        tau_variance_AMAZ = np.append(tau_variance_AMAZ, np.nan)
                
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
                np.savetxt("{}/feedbacks/network_{}_{}/{}/feedbacks_Tstart{}_Tend{}_tdur{}_{:.2f}.txt".format(long_save_name, 
			        kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength), data)
                time = data.T[0]
                state_gis = data.T[1]
                state_thc = data.T[2]
                state_wais = data.T[3]
                state_amaz = data.T[4]
                
                # Plotting structure
                fig = plt.figure()
                plt.grid(True)
                plt.title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                plt.plot(time, state_amaz, label="AMAZ", color='g')
                plt.plot(time, gaussian_filter1d(state_amaz, bw), color = 'grey')
                plt.plot(time, state_thc, label="THC", color='b')
                plt.plot(time, gaussian_filter1d(state_thc, bw), color = 'grey')
                plt.plot(time, state_wais, label="WAIS", color='k')
                plt.plot(time, gaussian_filter1d(state_wais, bw), color = 'grey')
                plt.plot(time, state_gis, label="GIS", color='c')
                plt.plot(time, gaussian_filter1d(state_gis, bw), color = 'grey')
                plt.xlabel("Time [yr]")
                plt.ylabel("system feature f [a.u.]")
                plt.legend(loc='best')  # , ncol=5)
                ax2 = plt.gca().twinx()
                ax2.plot(time, GMT[:int(duration)], color='r')
                ax2.grid(False)
                ax2.set_ylabel("$\Delta$GMT")
                ax2.yaxis.label.set_color('r')
                plt.tight_layout()
                if scaled_noise == True:
                    fig.savefig("{}/feedbacks/network_{}_{}/{}/feedbacks_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                else:
                    fig.savefig("{}/feedbacks/network_{}_{}/{}/feedbacks_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                #plt.show()
                plt.clf()
                plt.close()

                if ews_calculate == True:
                    
                    # GIS

                    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8,7), sharex=True)
                    if scaled_noise == True:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (scaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    else:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (unscaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    # Plot residual
                    ax1.grid(True)
                    ax1.plot(time, state_gis - gaussian_filter1d(state_gis, bw), label="GIS",color = 'c')
                    ax1.axvline(tip_t_gis, color='k', linestyle='--')
                    ax1.set_ylabel("System residual [a.u.]")

                    # Plot autocorrelation
                    ax2.grid(True)
                    ax2r = ax2.twinx()
                    ax2r.grid(False)
                    ax2.scatter(time, ann_autocorr_GIS, label="GIS", color='c', s=2)
                    ax2r.plot(time, tau_autocorr_GIS, color='r')
                    ax2.axvline(tip_t_gis, color='k', linestyle='--')
                    ax2.set_ylabel("Autocorrelation")
                    ax2r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax2r.yaxis.label.set_color('r')

                    # Plot variance
                    ax3.grid(True)
                    ax3r= ax3.twinx()
                    ax3r.grid(False)
                    ax3.scatter(time, ann_variance_GIS, label="GIS", color='c', s=2)
                    ax3r.plot(time, tau_variance_GIS, color='r')
                    ax3.axvline(tip_t_gis, color='k', linestyle='--')
                    ax3.set_xlabel("Time [yr]")
                    ax3.set_ylabel(r"Variance [a.u.$^2$]")
                    ax3r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax3r.yaxis.label.set_color('r')

                    fig.tight_layout()
                    if scaled_noise == True:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_GIS_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    else:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_GIS_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    #plt.show()
                    fig.clf()
                    plt.close()
                    
                    # THC

                    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8,7), sharex=True)
                    if scaled_noise == True:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (scaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    else:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (unscaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    # Plot residual
                    ax1.grid(True)
                    ax1.plot(time, state_thc - gaussian_filter1d(state_thc, bw), label="THC",color = 'b')
                    ax1.axvline(tip_t_thc, color='k', linestyle='--')
                    ax1.set_ylabel("System residual [a.u.]")

                    # Plot autocorrelation
                    ax2.grid(True)
                    ax2r = ax2.twinx()
                    ax2r.grid(False)
                    ax2.scatter(time, ann_autocorr_THC, label="THC", color='b', s=2)
                    ax2r.plot(time, tau_autocorr_THC, color='r')
                    ax2.axvline(tip_t_thc, color='k', linestyle='--')
                    ax2.set_ylabel("Autocorrelation")
                    ax2r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax2r.yaxis.label.set_color('r')

                    # Plot variance
                    ax3.grid(True)
                    ax3r= ax3.twinx()
                    ax3r.grid(False)
                    ax3.scatter(time, ann_variance_THC, label="THC", color='b', s=2)
                    ax3r.plot(time, tau_variance_THC, color='r')
                    ax3.axvline(tip_t_thc, color='k', linestyle='--')
                    ax3.set_xlabel("Time [yr]")
                    ax3.set_ylabel(r"Variance [a.u.$^2$]")
                    ax3r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax3r.yaxis.label.set_color('r')

                    fig.tight_layout()
                    if scaled_noise == True:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_THC_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    else:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_THC_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    #plt.show()
                    fig.clf()
                    
                    # WAIS

                    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8,7), sharex=True)
                    if scaled_noise == True:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (scaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    else:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (unscaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    # Plot residual
                    ax1.grid(True)
                    ax1.plot(time, state_wais - gaussian_filter1d(state_wais, bw), label="WAIS",color = 'k')
                    ax1.axvline(tip_t_wais, color='k', linestyle='--')
                    ax1.set_ylabel("System residual [a.u.]")

                    # Plot autocorrelation
                    ax2.grid(True)
                    ax2r = ax2.twinx()
                    ax2r.grid(False)
                    ax2.scatter(time, ann_autocorr_WAIS, label="WAIS", color='k', s=2)
                    ax2r.plot(time, tau_autocorr_WAIS, color='r')
                    ax2.axvline(tip_t_wais, color='k', linestyle='--')
                    ax2.set_ylabel("Autocorrelation")
                    ax2r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax2r.yaxis.label.set_color('r')

                    # Plot variance
                    ax3.grid(True)
                    ax3r= ax3.twinx()
                    ax3r.grid(False)
                    ax3.scatter(time, ann_variance_WAIS, label="WAIS", color='k', s=2)
                    ax3r.plot(time, tau_variance_WAIS, color='r')
                    ax3.axvline(tip_t_wais, color='k', linestyle='--')
                    ax3.set_xlabel("Time [yr]")
                    ax3.set_ylabel(r"Variance [a.u.$^2$]")
                    ax3r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax3r.yaxis.label.set_color('r')

                    fig.tight_layout()
                    if scaled_noise == True:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_WAIS_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    else:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_WAIS_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    #plt.show()
                    fig.clf()

                    # AMAZ

                    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8,7), sharex=True)
                    if scaled_noise == True:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (scaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    else:
                        fig.suptitle("Coupling strength: {}\n Wais to Thc:{} Thc to Amaz:{} \n Noise level:{} (unscaled)".format(np.round(strength, 2), kk[0], kk[1], noise))
                    # Plot residual
                    ax1.grid(True)
                    ax1.plot(time, state_amaz - gaussian_filter1d(state_amaz, bw), label="AMAZ",color = 'g')
                    ax1.axvline(tip_t_amaz, color='k', linestyle='--')
                    ax1.set_ylabel("System residual [a.u.]")

                    # Plot autocorrelation
                    ax2.grid(True)
                    ax2r = ax2.twinx()
                    ax2r.grid(False)
                    ax2.scatter(time, ann_autocorr_AMAZ, label="AMAZ", color='g', s=2)
                    ax2r.plot(time, tau_autocorr_AMAZ, color='r')
                    ax2.axvline(tip_t_amaz, color='k', linestyle='--')
                    ax2.set_ylabel("Autocorrelation")
                    ax2r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax2r.yaxis.label.set_color('r')

                    # Plot variance
                    ax3.grid(True)
                    ax3r= ax3.twinx()
                    ax3r.grid(False)
                    ax3.scatter(time, ann_variance_AMAZ, label="AMAZ", color='g', s=2)
                    ax3r.plot(time, tau_variance_AMAZ, color='r')
                    ax3.axvline(tip_t_amaz, color='k', linestyle='--')
                    ax3.set_xlabel("Time [yr]")
                    ax3.set_ylabel(r"Variance [a.u.$^2$]")
                    ax3r.set_ylabel(r"Kendall $\tau$ correlation")
                    ax3r.yaxis.label.set_color('r')

                    fig.tight_layout()
                    if scaled_noise == True:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_AMAZ_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}scaled.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    else:
                        fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_AMAZ_Tstart{}_Tend{}_tdur{}_{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_start, T_end, t_dur, strength, noise))
                    #plt.show()
                    fig.clf()
                    

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

