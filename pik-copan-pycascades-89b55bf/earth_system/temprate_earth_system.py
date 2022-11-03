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
from EWS_functions_fixed import autocorrelation, calc_autocorrelation, calc_variance

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
long_save_name = "results"

#######################GLOBAL VARIABLES##############################
#drive coupling strength
coupling_strength = np.array([0.35])
#drive global mean temperature (GMT) above pre-industrial
T_end = 3.5                             # final GMT increase [deg] 
T_rates = [1./10000]    # GMT increase rate [deg/yr]
durations = []                           # duration of GMT increase [yr]
for T_rate in T_rates: durations.append(int(T_end/T_rate)) 

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
    min_point = 0			            # minimum number of states after which to start calculating EWS
    step_size = 10                                  # EWS sampling frequency; the number of states between each EWS calculation
    detrend_window = [7000, 7000, 7000, 7000] 	    # the length of the detrending window
    bandwidth = [200, 200, 200, 200]                # the bandwidth for filtering timeseries
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
        for T_ind in range(len(T_rates)):
            
            T_rate = T_rates[T_ind]
            duration = durations[T_ind]
            print("Temperature rate: {}°C/yr".format(T_rate))
            print("Length of simulation: {}yr".format(duration))
            t = 0               # Start time
            effective_GMT = 0   # Start temperature
            
            output = []

            # initialize intermediate storage arrays for EWS analysis
            states = np.empty((n, duration)) 
            autocorr = np.empty((n, duration))
            variance = np.empty((n, duration))
            Tau_autocorr = [np.nan, np.nan, np.nan, np.nan]
            Tau_variance = [np.nan, np.nan, np.nan, np.nan]
            state_tipped = [False, False, False, False]
            start_point = [0, 0, 0, 0]
            tip_t = [np.nan, np.nan, np.nan, np.nan]

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
                # add NaN for no calculated autocorrelation and variance
                autocorr[elem, t] = np.nan
                variance[elem, t] = np.nan

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
                           [net.get_tip_states(ev.get_timeseries()[1][-1])[3]].count(True),
                           autocorr[0,t], autocorr[1,t], autocorr[2,t], autocorr[3,t],
                           variance[0,t], variance[1,t], variance[2,t], variance[3,t]
                           ])

            for t in range(1, duration):
                
                # increase GMT by set rate
                effective_GMT += T_rate
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
                        tip_t[elem] = t-1

                        # calculate Kendall Tau correlation of autocorrelation and variance
                        Tau_autocorr[elem], p_value = kendalltau(autocorr[elem,:t], np.arange(t), nan_policy="omit")
                        Tau_variance[elem], p_value = kendalltau(variance[elem,:t], np.arange(t), nan_policy="omit")
                  
                    
                    # if element has not tipped and timeseries is long enough, calculate autocorrelation and variance
                    if state_tipped[elem] == False and t+1 >= max(min_point,start_point[elem]) + detrend_window[elem]:

                        autocorr[elem,t], next_point = calc_autocorrelation(states[elem,:t+1], start_point[elem],
                                                                        detrend_window[elem], bandwidth[elem],
                                                                        min_point, step_size)
                        
                        variance[elem,t], next_point = calc_variance(states[elem,:t+1], start_point[elem], 
                                                                        detrend_window[elem], bandwidth[elem],
                                                                        min_point, step_size)
                        
                        # save the point to start up again in next round
                        start_point[elem] = next_point 

                    # if no autocorrelation and variance was calculated this year, append NaN
                    else:

                        autocorr[elem, t] = np.nan
                        variance[elem, t] = np.nan
                
                print(start_point) 
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
                               autocorr[0,t], autocorr[1,t], autocorr[2,t], autocorr[3,t],
                               variance[0,t], variance[1,t], variance[2,t], variance[3,t]
                               ])
            
            #necessary for break condition
            if len(output) != 0:
                #saving structure
                data = np.array(output)
                np.savetxt("{}/feedbacks/network_{}_{}/{}/feedbacks_Tend{}_Trate{}_d{:.2f}_n{}.txt".format(long_save_name, 
			        kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise), data)
                Tau_analysis = np.array([state_tipped, Tau_autocorr, Tau_variance])
                np.savetxt("{}/feedbacks/network_{}_{}/{}/Tau_Tend{}_Trate{}_d{:.2f}_n{}.txt".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise), Tau_analysis)
                 
                # Plotting structure
                time = data.T[0]
                colors = ['c','b','k','g']
                labels = ['GIS', 'THC', 'WAIS', 'AMAZ']

                fig = plt.figure()
                plt.grid(True)
                plt.title("Coupling strength: {}\n  Wais to Thc:{} Thc to Amaz:{}".format(np.round(strength, 2), kk[0], kk[1]))
                for elem in range(n):
                    plt.plot(time, data.T[elem+1], label=labels[elem], color=colors[elem])
                    plt.plot(time, gaussian_filter1d(data.T[elem+1], bandwidth[elem]), color = 'grey')
                plt.xlabel("Time [yr]")
                plt.ylabel("System feature f [a.u.]")
                plt.legend(loc='best')
                ax2 = plt.gca().twinx()
                ax2.plot(time, np.linspace(0,T_end,duration), color='r')
                ax2.grid(False)
                ax2.set_ylabel("$\Delta$GMT")
                ax2.yaxis.label.set_color('r')
                plt.tight_layout()
                fig.savefig("{}/feedbacks/network_{}_{}/{}/feedbacks_Tend{}_Trate{}_d{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), T_end, T_rate, strength, noise))
                plt.clf()
                plt.close()

                for elem in np.where(state_tipped)[0]:
                    
                    fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(8,7), sharex=True)
                    fig.suptitle(labels[elem]+"\n Coupling strength: {}, Wais to Thc:{} Thc to Amaz:{} \n Noise level:{}".format(np.round(strength, 2), kk[0], kk[1], noise))

                    # Plot residual
                    ax1.grid(True)
                    ax1.plot(time[:tip_t[elem]], 
                            (data.T[elem+1,:] - gaussian_filter1d(data.T[elem+1,:], bandwidth[elem]))[:tip_t[elem]], c=colors[elem])
                    ax1.set_ylabel("System residual [a.u.]")

                    # Plot autocorrelation
                    ax2.grid(True)
                    ax2.scatter(time, autocorr[elem,:], label=labels[elem], c=colors[elem], s=2)
                    ax2.set_ylabel("Autocorrelation")
                    ax2.text(0.1, 0.1, "Kendall tau: {}".format(Tau_autocorr[elem]))

                    # Plot variance
                    ax3.grid(True)
                    ax3.scatter(time, variance[elem,:], label=labels[elem], c=colors[elem], s=2)
                    ax3.set_xlabel("Time [yr]")
                    ax3.set_ylabel(r"Variance [a.u.$^2$]")
                    ax3.text(0.1, 0.1, "Kendall tau: {}".format(Tau_variance[elem]))

                    fig.tight_layout()
                    fig.savefig("{}/feedbacks/network_{}_{}/{}/EWS_elem{}_Tend{}_Trate{}_d{:.2f}_n{}.pdf".format(long_save_name,
                                kk[0], kk[1], str(mc_dir).zfill(4), elem, T_end, T_rate, strength, noise))
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

